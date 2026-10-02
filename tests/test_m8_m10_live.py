import base64

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tbot.alerts.telegram import TelegramBot
from tbot.config import load_risk
from tbot.dashboard.app import create_app
from tbot.data.synthetic import make_bars
from tbot.exec.paper_sim import PaperSimBroker
from tbot.exec.router import OrderRouter
from tbot.ledger.db import Ledger
from tbot.live.runner import Runner
from tbot.report.daily import build_report
from tbot.risk.killswitch import KillSwitch
from tbot.risk.limits import RiskEngine
from tbot.strategy.signal import Signal
from tbot.strategy.spec import StrategySpec


class FakeNotifier:
    def __init__(self):
        self.messages, self.approvals = [], []

    def notify(self, kind, text):
        self.messages.append((kind, text))

    def request_approval(self, pending_id, text):
        self.approvals.append((pending_id, text))


class FakeScorer:
    def score(self, x):
        return {
            "regime": {"trend": 0.6, "range": 0.3, "high_vol": 0.1},
            "direction": {"up": 0.5, "down": 0.2, "flat": 0.3},
            "pressure_real": {"no": 0.4, "yes": 0.6},
            "setup_quality": {"fail": 0.4, "success": 0.6},
            "risk_state": {"calm": 0.6, "elevated": 0.3, "stress": 0.1},
        }


@pytest.fixture
def bars():
    return make_bars(days=12, seed=2)


def make_runner(tmp_path, bars, decide=None, limits=None, spec=None):
    limits = limits or load_risk()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    broker = PaperSimBroker(cash=100_000)
    risk = RiskEngine(limits, KillSwitch(tmp_path / "state"), universe={"SPY"}, mode="paper")
    note = FakeNotifier()
    router = OrderRouter(broker, risk)
    spec = spec or StrategySpec(name="t", template="trend", symbols=["SPY"], flatten_eod=True)
    fetch = lambda sym, now: bars[bars["end"] <= now]  # noqa: E731
    r = Runner(spec=spec, scorer=FakeScorer(), router=router, broker=broker, ledger=ledger, notifier=note, fetch_recent=fetch, limits=limits, state_dir=tmp_path / "state", decide=decide)
    return r, broker, ledger, note


def drive(runner, broker, bars, start=80, end=None):
    for i in range(start, end or len(bars)):
        row = bars.iloc[i]
        broker.on_bar("SPY", row)  # el broker simulado ejecuta brackets con la vela
        runner.on_bar_close(row["end"])


def test_ledger_roundtrip(tmp_path):
    led = Ledger(tmp_path / "l.sqlite3")
    did = led.log_decision(pd.Timestamp("2026-10-01 15:00", tz="UTC"), "SPY", 500.0, {"a": 1.0}, {"q": {"x": 1.0}}, 0.8, True, "entry_submitted", "", "cid1", 1, "fp")
    led.log_event("kill", "test")
    assert led.decisions(limit=5)[0]["id"] == did
    assert led.events(limit=5)[0]["kind"] == "kill"


def test_every_bar_logs_a_decision_with_probabilities_and_latency(tmp_path, bars):
    runner, broker, ledger, _ = make_runner(tmp_path, bars)
    drive(runner, broker, bars, start=80, end=120)
    rows = ledger.decisions(limit=1000)
    assert len(rows) == 40
    assert all(r["probs"] and r["latency_ms"] is not None for r in rows)


def test_forced_signal_goes_through_router_and_brackets(tmp_path, bars):
    fired = {"n": 0}

    def decide(sym, i, feats, probs):
        if fired["n"] == 0:
            fired["n"] += 1
            return Signal(0.6, 1.5, "forzada")
        return None

    runner, broker, ledger, note = make_runner(tmp_path, bars, decide=decide)
    drive(runner, broker, bars, start=80, end=200)
    orders = ledger.orders()
    entry = [o for o in orders if not o["is_exit"]]
    assert len(entry) == 1 and entry[0]["stop"] is not None and entry[0]["tp"] is not None
    assert any(k == "fill" for k, _ in note.messages)
    assert len(ledger.trades()) == 1  # se cerró por stop, take profit, tiempo o fin del día
    assert broker.positions() == {}


def test_big_trade_waits_for_telegram_approval(tmp_path, bars):
    limits = load_risk()
    calls = {"n": 0}

    def decide(sym, i, feats, probs):
        calls["n"] += 1
        return Signal(0.9, 1.5, "grande") if calls["n"] == 1 else None

    runner, broker, ledger, note = make_runner(tmp_path, bars, decide=decide, limits=limits)
    runner.calibrated = lambda: True  # Kelly completo: la operación supera US$1.000
    drive(runner, broker, bars, start=80, end=81)
    assert note.approvals, "tenía que pedir aprobación"
    assert broker.submitted == []
    pid = note.approvals[0][0]
    runner.approve(pid)  # aprobada antes de que venza (10 minutos)
    assert len(broker.submitted) == 1


def test_unanswered_approval_expires_at_next_bar(tmp_path, bars):
    calls = {"n": 0}

    def decide(sym, i, feats, probs):
        calls["n"] += 1
        return Signal(0.9, 1.5, "grande") if calls["n"] == 1 else None

    runner, broker, ledger, note = make_runner(tmp_path, bars, decide=decide)
    runner.calibrated = lambda: True
    drive(runner, broker, bars, start=80, end=82)  # la vela siguiente llega 15 min después
    assert runner.router.pending == {} and broker.submitted == []
    assert any("Venció" in t for _, t in note.messages)


def test_drawdown_kills_flattens_and_alerts(tmp_path, bars):
    runner, broker, ledger, note = make_runner(tmp_path, bars)
    broker.set_position("SPY", 100, 400.0)
    broker.cash = 100_000 - 40_000
    broker.mark("SPY", 300.0)  # cae el capital: 60k + 30k = 90k → −10% desde el pico de 100k
    runner.state.peak_equity = 100_000
    runner.on_bar_close(bars["end"].iloc[100])
    assert runner.router.risk.killswitch.active
    assert broker.positions() == {}
    assert any(k == "kill" for k, _ in note.messages)


def test_reconciliation_detects_unknown_position(tmp_path, bars):
    runner, broker, ledger, note = make_runner(tmp_path, bars)
    broker.set_position("SPY", 7, 400.0)  # posición que el bot no abrió
    runner.on_bar_close(bars["end"].iloc[100])
    assert any(k == "reconcile" for k, _ in note.messages)


def test_strategy_invalidation_disables_entries(tmp_path, bars):
    runner, broker, ledger, note = make_runner(tmp_path, bars)
    for _ in range(30):
        ledger.log_trade("SPY", bars["end"].iloc[0], bars["end"].iloc[1], 1, 100.0, 99.0, -1.0, -0.01, "stop", 0.6)
    runner.on_bar_close(bars["end"].iloc[100])
    assert runner.strategy_disabled()
    assert any(k == "invalidation" for k, _ in note.messages)


def test_macro_event_day_blocks_entries(tmp_path, bars):
    day = bars["start"].iloc[100].tz_convert("America/New_York").date()
    runner, broker, ledger, note = make_runner(tmp_path, bars, decide=lambda *a: Signal(0.6, 1.5, "x"), spec=StrategySpec(name="t", template="trend", symbols=["SPY"], flatten_before_events=True))
    runner.events = [(pd.Timestamp(f"{day} 14:00").tz_localize("America/New_York"), "FOMC")]
    runner.on_bar_close(bars["end"].iloc[100])
    assert ledger.orders() == []
    assert ledger.decisions(limit=1)[0]["action"] == "blocked_event"


def test_dashboard_requires_password(tmp_path, bars):
    led = Ledger(tmp_path / "l.sqlite3")
    app = create_app(led, state_dir=tmp_path, password="clave-larga-123")
    c = TestClient(app)
    assert c.get("/api/summary").status_code == 401
    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:clave-larga-123").decode()}
    assert c.get("/api/summary", headers=auth).status_code == 200
    assert c.get("/", headers=auth).status_code == 200
    with pytest.raises(ValueError):
        create_app(led, state_dir=tmp_path, password="")


def test_telegram_only_obeys_owner_and_handles_approvals():
    approved, killed = [], []
    bot = TelegramBot(token="x", chat_id="111", on_approve=approved.append, on_reject=lambda p: None, on_kill=killed.append, on_status=lambda: "ok", http=None)
    bot.handle_update({"message": {"chat": {"id": 999}, "text": "/kill"}})
    assert killed == []
    bot.handle_update({"message": {"chat": {"id": 111}, "text": "/kill pánico"}})
    assert killed == ["pánico"]
    bot.handle_update({"callback_query": {"id": "1", "data": "approve:abc", "message": {"chat": {"id": 111}}}})
    assert approved == ["abc"]


def test_daily_report_has_required_fields(tmp_path, bars):
    led = Ledger(tmp_path / "l.sqlite3")
    t0 = bars["end"].iloc[100]
    led.log_trade("SPY", t0, t0 + pd.Timedelta(minutes=30), 10, 100.0, 102.0, 20.0, 0.02, "take_profit", 0.6)
    led.log_trade("SPY", t0, t0 + pd.Timedelta(minutes=45), 10, 100.0, 99.0, -10.0, -0.01, "stop", 0.6)
    led.log_decision(t0, "SPY", 100.0, {}, {}, 0.7, False, "none", "", None, 1, "fp")
    r = build_report(led, t0.tz_convert("America/New_York").date(), opus_cost_usd=0.5)
    for k in ("trades", "pnl", "win_rate", "largest_loss", "avg_latency_ms", "cost_per_decision_usd", "calibration"):
        assert k in r.data
    assert r.data["pnl"] == pytest.approx(10.0) and r.data["win_rate"] == 0.5 and r.data["largest_loss"] == -10.0
    assert "P&L" in r.markdown


def test_observe_mode_never_sends_orders(tmp_path, bars):
    runner, broker, ledger, note = make_runner(tmp_path, bars, decide=lambda *a: Signal(0.6, 1.5, "x"))
    runner.observe = True
    drive(runner, broker, bars, start=80, end=100)
    assert broker.submitted == [] and ledger.orders() == []
    assert any(d["action"] == "observe_only" for d in ledger.decisions(limit=100))


def test_next_bar_close():
    from tbot.live.service import next_bar_close

    t = pd.Timestamp("2026-10-01 10:07", tz="America/New_York").tz_convert("UTC")
    assert next_bar_close(t, 15).tz_convert("America/New_York").strftime("%H:%M") == "10:15"


def test_fills_are_not_recorded_twice_after_restart(tmp_path, bars):
    from tbot.exec.broker import Fill

    runner, broker, ledger, note = make_runner(tmp_path, bars)
    t = bars["end"].iloc[100]
    entry = Fill(t, "cid-1", "SPY", "buy", 5, 400.0, "entry")
    exit_ = Fill(t + pd.Timedelta(minutes=30), "cid-1-tp", "SPY", "sell", 5, 404.0, "take_profit")
    broker._fills = [entry, exit_]
    runner._process_fills()
    broker._fills = [entry, exit_]  # Alpaca vuelve a informar los mismos fills
    runner._process_fills()
    assert len(ledger.trades()) == 1
