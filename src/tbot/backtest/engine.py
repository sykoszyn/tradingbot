"""Backtest vela por vela con EL MISMO código que en vivo: state engine, estrategia, sizing y motor de riesgo.

Supuestos conservadores:
- se decide al cierre de la vela y se entra a la apertura de la siguiente (con slippage);
- si en una vela se tocan el stop y el take profit, se asume el stop;
- si el precio abre más allá del stop, se sale a la apertura (gap);
- el take profit es una orden límite: se llena al precio límite, sin slippage;
- las operaciones que piden aprobación manual se cuentan como aprobadas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RiskLimits, load_risk
from ..risk.killswitch import KillSwitch
from ..risk.limits import RiskEngine
from ..risk.sizing import position_shares
from ..risk.types import AccountState, OrderIntent, Position
from ..scorer.questions import QUESTIONS
from ..state.engine import FEATURES, features_frame
from ..strategy.signal import Signal
from ..strategy.spec import StrategySpec
from .costs import DEFAULT_COSTS, CostModel

__all__ = ["Signal", "Trade", "BacktestResult", "run_backtest"]

Decide = Callable[[str, int, dict, dict | None], Signal | None]


@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    ret: float
    exit_reason: str
    reason: str
    p_setup: float
    regime: str
    entry_index: int


@dataclass
class _Open:
    qty: int
    entry_price: float
    stop: float
    tp: float
    entry_ts: pd.Timestamp
    entry_index: int
    bars_held: int
    sig: Signal
    regime: str
    atr: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    events: list[tuple[pd.Timestamp, str, str]] = field(default_factory=list)
    killed: bool = False


def observed_regime(f: dict) -> str:
    """Régimen observable (solo hacia atrás), para cortar los resultados por régimen."""
    if f.get("rvol_ratio", 1) > 1.5:
        return "high_vol"
    return "trend" if abs(f.get("trend", 0)) > 0.002 else "range"


class _NoopKill(KillSwitch):
    """Kill switch en memoria para el backtest (no toca el archivo real)."""

    def __init__(self) -> None:
        self._on = False
        self._reason: str | None = None

    @property
    def active(self) -> bool:  # type: ignore[override]
        return self._on

    def reason(self) -> str | None:
        return self._reason

    def fire(self, reason: str) -> None:
        self._on, self._reason = True, reason


def run_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    spec: StrategySpec,
    *,
    decide: Decide | None = None,
    probs_by_symbol: dict[str, dict[str, np.ndarray]] | None = None,
    costs: CostModel = DEFAULT_COSTS,
    limits: RiskLimits | None = None,
    calibrated: bool = True,
    start_equity: float = 100_000.0,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] = features_frame,
    atr_override: float | None = None,
) -> BacktestResult:
    from ..strategy.templates import make_decider

    limits = limits or load_risk()
    decide = decide or make_decider(spec)
    risk = RiskEngine(limits, _NoopKill(), universe=set(bars_by_symbol), mode="paper")

    feats = {s: feature_fn(b.reset_index(drop=True)) for s, b in bars_by_symbol.items()}
    bars = {s: b.reset_index(drop=True) for s, b in bars_by_symbol.items()}
    feat_cols = {s: [c for c in f.columns if c not in ("atr", "price")] for s, f in feats.items()}
    feat_np = {s: feats[s][feat_cols[s]].to_numpy(float) for s in bars}
    atr_np = {s: feats[s]["atr"].to_numpy(float) for s in bars}
    Op = {s: b["open"].to_numpy(float) for s, b in bars.items()}
    H = {s: b["high"].to_numpy(float) for s, b in bars.items()}
    Lo = {s: b["low"].to_numpy(float) for s, b in bars.items()}
    C = {s: b["close"].to_numpy(float) for s, b in bars.items()}
    day = {s: b["start"].dt.tz_convert("America/New_York").dt.date.to_numpy() for s, b in bars.items()}
    last_of_day = {s: np.append(day[s][1:] != day[s][:-1], True) for s in bars}

    events: list[tuple[pd.Timestamp, str, str]] = []
    timeline = sorted(set().union(*[set(b["end"]) for b in bars.values()]))
    idx_at = {s: dict(zip(b["end"], range(len(b)), strict=True)) for s, b in bars.items()}

    cash = start_equity
    open_: dict[str, _Open] = {}
    pending: dict[str, tuple[Signal, float, int]] = {}
    last_close: dict[str, float] = {}
    trades: list[Trade] = []
    eq_ts, eq_vals = [], []
    day_start_eq = peak = start_equity
    cur_day = None
    trades_today = 0
    killed = False

    def equity() -> float:
        return cash + sum(p.qty * last_close.get(s, p.entry_price) for s, p in open_.items())

    def positions() -> dict[str, Position]:
        return {s: Position(s, p.qty, last_close.get(s, p.entry_price)) for s, p in open_.items()}

    def close_pos(s: str, i: int, price: float, reason: str, ts: pd.Timestamp) -> None:
        nonlocal cash
        p = open_.pop(s)
        fees = costs.sell_fees(p.qty, price)
        cash += p.qty * price - fees
        pnl = (price - p.entry_price) * p.qty - fees
        trades.append(Trade(s, p.entry_ts, ts, p.qty, p.entry_price, price, pnl, pnl / (p.entry_price * p.qty), reason, p.sig.reason, p.sig.p_setup, p.regime, p.entry_index))

    in_window = lambda ts: (start_ts is None or ts > start_ts) and (end_ts is None or ts <= end_ts)  # noqa: E731

    for ts in timeline:
        syms = [s for s in bars if ts in idx_at[s]]
        tsd = ts.tz_convert("America/New_York").date()
        if tsd != cur_day:
            cur_day, day_start_eq, trades_today = tsd, equity(), 0
        for s in syms:
            i = idx_at[s][ts]
            atr_i = atr_override or atr_np[s][i]
            # 1) entrada pendiente: se llena a la apertura
            if s in pending and not killed:
                sig, atr_d, di = pending.pop(s)
                fill = Op[s][i] + costs.market_slip(Op[s][i], atr_d)
                stop_dist = spec.sl_atr * atr_d
                eq = equity()
                qty = position_shares(p=sig.p_setup, b=sig.b, stop_distance=stop_dist, price=fill, equity=eq, limits=limits, calibrated=calibrated, p_min=0.0)
                if qty > 0:
                    intent = OrderIntent(s, "buy", qty, fill, fill - stop_dist, fill + spec.tp_atr * atr_d, sig.reason)
                    acct = AccountState(eq, cash, positions(), day_start_eq, peak, trades_today, ts, True)
                    d = risk.check(intent, acct)
                    if d.status in ("approved", "needs_approval"):
                        cash -= qty * fill
                        f = dict(zip(feat_cols[s], feat_np[s][di], strict=True))
                        open_[s] = _Open(qty, fill, fill - stop_dist, fill + spec.tp_atr * atr_d, ts, di, 0, sig, observed_regime(f), atr_d)
                        trades_today += 1
                    else:
                        events.append((ts, "rejected", f"{s}: {'; '.join(d.reasons)}"))
            # 2) salidas
            if s in open_:
                p = open_[s]
                p.bars_held += 1
                if Lo[s][i] <= p.stop:
                    px = (Op[s][i] if Op[s][i] < p.stop else p.stop) - costs.market_slip(p.stop, p.atr)
                    close_pos(s, i, px, "stop", ts)
                elif H[s][i] >= p.tp:
                    close_pos(s, i, max(Op[s][i], p.tp), "take_profit", ts)
                elif p.bars_held >= spec.max_hold_bars:
                    close_pos(s, i, C[s][i] - costs.market_slip(C[s][i], p.atr), "time", ts)
                elif spec.flatten_eod and last_of_day[s][i]:
                    close_pos(s, i, C[s][i] - costs.market_slip(C[s][i], p.atr), "eod", ts)
            last_close[s] = C[s][i]
            _ = atr_i

        eq = equity()
        peak = max(peak, eq)
        if not killed:
            for e in risk.on_equity(AccountState(eq, cash, positions(), day_start_eq, peak, trades_today, ts, True)):
                events.append((ts, e.kind, e.message))
                if e.kind == "kill":
                    killed = True
                    for s in list(open_):
                        close_pos(s, idx_at[s].get(ts, 0), last_close[s] - costs.market_slip(last_close[s], open_[s].atr), "kill", ts)
                    pending.clear()
        eq_ts.append(ts)
        eq_vals.append(equity())

        # 3) decisiones al cierre
        if killed or not in_window(ts):
            continue
        for s in syms:
            i = idx_at[s][ts]
            if s in open_ or s in pending or i + 1 >= len(bars[s]):
                continue
            if spec.flatten_eod and last_of_day[s][i]:
                continue
            row = feat_np[s][i]
            f = dict(zip(feat_cols[s], row, strict=True))
            probs = None
            if probs_by_symbol is not None:
                pr = probs_by_symbol[s]
                if not np.isfinite(pr["setup_quality"][i]).all():
                    continue
                probs = {q: dict(zip(cl, pr[q][i], strict=True)) for q, cl in QUESTIONS.items()}
            sig = decide(s, i, f, probs)
            if sig is not None:
                atr_d = atr_override or atr_np[s][i]
                if np.isfinite(atr_d) and atr_d > 0:
                    pending[s] = (sig, float(atr_d), i)

    return BacktestResult(trades, pd.Series(eq_vals, index=pd.DatetimeIndex(eq_ts), name="equity"), events, killed)


FEATURE_NAMES = FEATURES
