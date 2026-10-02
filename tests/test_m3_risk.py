import re
from pathlib import Path

import pandas as pd
import pytest

from tbot.config import load_risk
from tbot.exec.broker import SimBroker
from tbot.exec.router import OrderRouter
from tbot.risk.killswitch import KillSwitch
from tbot.risk.limits import RiskEngine
from tbot.risk.sizing import kelly_fraction, position_shares
from tbot.risk.types import AccountState, OrderIntent, Position

NOW = pd.Timestamp("2026-10-01 15:00", tz="UTC")


def account(equity=100_000, positions=None, day_start=None, peak=None, trades=0, open_=True):
    return AccountState(
        equity=equity,
        cash=equity,
        positions=positions or {},
        day_start_equity=day_start or equity,
        peak_equity=peak or equity,
        trades_today=trades,
        now=NOW,
        market_open=open_,
    )


def buy(qty=10, price=500.0, stop=490.0, symbol="SPY"):
    return OrderIntent(symbol=symbol, side="buy", qty=qty, ref_price=price, stop_price=stop, take_profit=520.0, reason="test")


@pytest.fixture
def engine(tmp_path):
    return RiskEngine(load_risk(), KillSwitch(tmp_path), universe={"SPY", "QQQ"}, mode="paper")


def test_small_order_approved(engine):
    assert engine.check(buy(qty=1), account()).status == "approved"


def test_position_over_10pct_rejected(engine):
    d = engine.check(buy(qty=21), account())  # 21 × 500 = 10.500 > 10% de 100.000
    assert d.status == "rejected" and "posición" in " ".join(d.reasons)


def test_gross_exposure_over_50pct_rejected(engine):
    pos = {f"S{i}": Position(f"S{i}", 90, 500.0) for i in range(1)} | {"QQQ": Position("QQQ", 90, 500.0)}
    d = engine.check(buy(qty=19), account(positions=pos))  # 45k + 45k + 9.5k = 99,5% > 50%
    assert d.status == "rejected" and any("exposición" in r for r in d.reasons)


def test_daily_loss_halts_new_entries_but_allows_exits(engine):
    acct = account(equity=97_900, day_start=100_000, positions={"SPY": Position("SPY", 5, 500.0)})
    assert engine.check(buy(qty=1), acct).status == "rejected"
    exit_ = OrderIntent("SPY", "sell", 5, 500.0, None, None, "salida", is_exit=True)
    assert engine.check(exit_, acct).status == "approved"


def test_drawdown_fires_kill_switch(engine, tmp_path):
    events = engine.on_equity(account(equity=89_000, peak=100_000))
    assert engine.killswitch.active
    assert any(e.kind == "kill" for e in events)
    assert engine.check(buy(qty=1), account()).status == "rejected"
    assert KillSwitch(tmp_path).active  # persiste en disco: sobrevive a un reinicio


def test_kill_switch_reset_requires_explicit_confirmation(tmp_path):
    ks = KillSwitch(tmp_path)
    ks.fire("prueba")
    with pytest.raises(PermissionError):
        ks.reset("si")
    ks.reset("REACTIVAR")
    assert not ks.active


def test_trades_per_day_limit(engine):
    assert engine.check(buy(qty=1), account(trades=10)).status == "rejected"


def test_entry_without_stop_rejected(engine):
    assert engine.check(buy(qty=1, stop=None), account()).status == "rejected"


def test_market_closed_and_unknown_symbol_rejected(engine):
    assert engine.check(buy(qty=1), account(open_=False)).status == "rejected"
    assert engine.check(buy(qty=1, symbol="TSLA"), account()).status == "rejected"


def test_big_order_needs_manual_approval(engine):
    assert engine.check(buy(qty=3), account()).status == "needs_approval"  # 1.500 > 1.000


def test_exit_cannot_be_bigger_than_position(engine):
    acct = account(positions={"SPY": Position("SPY", 5, 500.0)})
    assert engine.check(OrderIntent("SPY", "sell", 6, 500.0, None, None, "x", is_exit=True), acct).status == "rejected"


def test_live_mode_blocked_without_golive_check(tmp_path):
    e = RiskEngine(load_risk(), KillSwitch(tmp_path), universe={"SPY"}, mode="live", golive_ok=False)
    assert e.check(buy(qty=1), account()).status == "rejected"


def test_kelly_quarter_and_caps():
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(0.40, 1.0) == 0.0  # sin ventaja: cero
    assert kelly_fraction(0.6, 1.5) == pytest.approx(0.6 - 0.4 / 1.5)
    limits = load_risk()
    # p alta pero sin calibración verificada: tamaño mínimo fijo
    q_uncal = position_shares(p=0.7, b=1.5, stop_distance=10.0, price=500.0, equity=100_000, limits=limits, calibrated=False, p_min=0.55)
    assert q_uncal == int(100_000 * limits.min_position_pct / 500)
    q_cal = position_shares(p=0.7, b=1.5, stop_distance=10.0, price=500.0, equity=100_000, limits=limits, calibrated=True, p_min=0.55)
    assert q_cal * 500 <= 100_000 * limits.max_position_pct
    assert position_shares(p=0.5, b=1.5, stop_distance=10.0, price=500.0, equity=100_000, limits=limits, calibrated=True, p_min=0.55) == 0


def test_router_is_the_only_way_to_send_orders(tmp_path, engine):
    broker = SimBroker()
    router = OrderRouter(broker, engine)
    res = router.submit(buy(qty=1), account())
    assert res.status == "approved" and len(broker.orders) == 1
    res = router.submit(buy(qty=50), account())
    assert res.status == "rejected" and len(broker.orders) == 1
    res = router.submit(buy(qty=3), account())
    assert res.status == "needs_approval" and len(broker.orders) == 1
    assert router.approve(res.pending_id, account()).status == "approved" and len(broker.orders) == 2


def test_expired_approval_is_not_executed(engine):
    broker = SimBroker()
    router = OrderRouter(broker, engine)
    res = router.submit(buy(qty=3), account())
    later = account()
    later = AccountState(**{**later.__dict__, "now": NOW + pd.Timedelta(minutes=30)})
    assert router.approve(res.pending_id, later).status == "rejected"
    assert len(broker.orders) == 0


def test_kill_switch_flattens_everything(engine):
    broker = SimBroker()
    broker.set_position("SPY", 10, 500.0)
    broker.set_position("QQQ", 4, 400.0)
    router = OrderRouter(broker, engine)
    router.kill("test", account(positions=broker.positions()))
    assert engine.killswitch.active
    assert broker.positions() == {}


def test_no_other_module_calls_the_broker_directly():
    src = Path(__file__).resolve().parents[1] / "src" / "tbot"
    allowed = {"exec/router.py", "exec/broker.py", "exec/alpaca_broker.py"}
    offenders = []
    for f in src.rglob("*.py"):
        rel = f.relative_to(src).as_posix()
        if rel in allowed:
            continue
        if re.search(r"\.(submit_order|_send|close_all_positions|submit_raw)\(", f.read_text()):
            offenders.append(rel)
    assert offenders == []
