from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def sharpe(daily_returns: pd.Series) -> float:
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity, dtype=float)
    return float((1 - e / e.cummax()).max()) if len(e) else 0.0


def t_stat(trade_returns: np.ndarray) -> float:
    r = np.asarray(trade_returns, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def daily_returns(equity: pd.Series) -> pd.Series:
    by_day = equity.groupby(equity.index.tz_convert("America/New_York").date).last()
    return by_day.pct_change().dropna()


def compute_metrics(res: BacktestResult, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    eq = res.equity
    if start is not None:
        eq = eq[eq.index > start]
    if end is not None:
        eq = eq[eq.index <= end]
    trades = [t for t in res.trades if (start is None or t.entry_ts > start) and (end is None or t.entry_ts <= end)]
    rets = np.array([t.ret for t in trades])
    dr = daily_returns(eq) if len(eq) else pd.Series(dtype=float)
    years = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) > 1 else 0.0
    total = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    m = {
        "sharpe": sharpe(dr),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "t_stat": t_stat(rets),
        "n_trades": len(trades),
        "years": years,
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else 0.0,
        "avg_trade": float(rets.mean()) if len(rets) else 0.0,
        "largest_loss": float(min((t.pnl for t in trades), default=0.0)),
        "killed": res.killed,
    }
    by_year: dict[str, dict] = {}
    for y, g in dr.groupby(pd.to_datetime(dr.index).year):
        ty = [t.ret for t in trades if t.entry_ts.year == y]
        by_year[str(y)] = {"sharpe": sharpe(g), "return": float((1 + g).prod() - 1), "n_trades": len(ty), "hit_rate": float(np.mean(np.array(ty) > 0)) if ty else 0.0}
    by_regime: dict[str, dict] = {}
    for reg in ("trend", "range", "high_vol"):
        tr = np.array([t.ret for t in trades if t.regime == reg])
        by_regime[reg] = {"n_trades": len(tr), "hit_rate": float((tr > 0).mean()) if len(tr) else 0.0, "avg_trade": float(tr.mean()) if len(tr) else 0.0}
    m["by_year"], m["by_regime"] = by_year, by_regime
    return m
