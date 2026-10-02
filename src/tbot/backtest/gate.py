"""El filtro que decide si una estrategia puede operar. Lo calcula el arnés, nunca el modelo."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from ..strategy.spec import StrategySpec
from .engine import Decide, run_backtest
from .walkforward import walk_forward


@dataclass(frozen=True)
class GateThresholds:
    min_sharpe: float = 1.5
    max_drawdown: float = 0.15
    min_hit_rate: float = 0.55
    min_t_stat: float = 2.0
    min_years: float = 2.0
    min_trades: int = 30

    @staticmethod
    def load(path: Path = ROOT / "config" / "gate.toml") -> GateThresholds:
        return GateThresholds(**tomllib.loads(path.read_text())) if path.exists() else GateThresholds()


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures}


def evaluate_gate(m: dict, g: GateThresholds) -> GateResult:
    f = []
    if m["sharpe"] <= g.min_sharpe:
        f.append(f"Sharpe {m['sharpe']:.2f} ≤ {g.min_sharpe}")
    if m["max_drawdown"] >= g.max_drawdown:
        f.append(f"caída máxima {m['max_drawdown']:.1%} ≥ {g.max_drawdown:.0%}")
    if m["hit_rate"] <= g.min_hit_rate:
        f.append(f"hit rate {m['hit_rate']:.1%} ≤ {g.min_hit_rate:.0%}")
    if m["t_stat"] <= g.min_t_stat:
        f.append(f"t-stat {m['t_stat']:.2f} ≤ {g.min_t_stat}")
    if m["years"] < g.min_years:
        f.append(f"solo {m['years']:.1f} años fuera de muestra (mínimo {g.min_years})")
    if m["n_trades"] < g.min_trades:
        f.append(f"solo {m['n_trades']} operaciones (mínimo {g.min_trades})")
    return GateResult(not f, f)


def _perturb_after(bars: pd.DataFrame, cutoff_i: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b = bars.copy()
    n = len(b) - cutoff_i - 1
    if n <= 0:
        return b
    shock = np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    for col in ("open", "high", "low", "close"):
        b.loc[cutoff_i + 1 :, col] = b.loc[cutoff_i + 1 :, col].to_numpy() * shock
    b.loc[cutoff_i + 1 :, "volume"] = b.loc[cutoff_i + 1 :, "volume"].to_numpy() * rng.uniform(0.3, 3.0, n)
    return b


def feature_truncation_check(bars: pd.DataFrame, feature_fn, samples: int = 60, seed: int = 0) -> list[str]:
    """Para momentos al azar, las features calculadas con datos hasta ese momento tienen que ser
    idénticas a las que usa el backtest con toda la historia. Si difieren, la feature mira el futuro."""
    b = bars.reset_index(drop=True)
    full = feature_fn(b)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(200, len(b) - 5), size=min(samples, max(1, len(b) - 205)), replace=False))
    bad: set[str] = set()
    for i in idx:
        part = feature_fn(b.iloc[: i + 1]).iloc[-1]
        row = full.iloc[i]
        for col in full.columns:
            x, y = row[col], part[col]
            if not (pd.isna(x) and pd.isna(y)) and not np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.add(col)
    return sorted(bad)


def lookahead_check(bars_by_symbol: dict[str, pd.DataFrame], spec: StrategySpec, *, decide: Decide | None = None, feature_fn=None, cutoff_frac: float = 0.6, use_scorer: bool = False) -> GateResult:
    """Cambia los datos DESPUÉS de un corte y verifica que nada de lo decidido ANTES cambie.

    Si cambia, alguna parte (features, etiquetas, estrategia) está mirando el futuro y la estrategia se rechaza.
    """
    kw = {} if feature_fn is None else {"feature_fn": feature_fn}
    from ..state.engine import features_frame

    leaky = feature_truncation_check(next(iter(bars_by_symbol.values())), feature_fn or features_frame)
    if leaky:
        return GateResult(False, [f"lookahead: las features {', '.join(leaky)} cambian si se quitan datos futuros"])
    first = next(iter(bars_by_symbol.values()))
    cutoff_i = int(len(first) * cutoff_frac)
    cutoff_ts = first["end"].iloc[cutoff_i]
    perturbed = {s: _perturb_after(b.reset_index(drop=True), cutoff_i, seed=99) for s, b in bars_by_symbol.items()}
    if use_scorer:
        a = walk_forward(bars_by_symbol, spec, decide=decide, **kw).result
        b = walk_forward(perturbed, spec, decide=decide, **kw).result
    else:
        a = run_backtest(bars_by_symbol, spec, decide=decide, **kw)
        b = run_backtest(perturbed, spec, decide=decide, **kw)
    before = lambda r: [(t.symbol, t.entry_ts, t.qty, round(t.entry_price, 6), t.exit_reason) for t in r.trades if t.exit_ts <= cutoff_ts]  # noqa: E731
    ta, tb = before(a), before(b)
    if ta != tb:
        return GateResult(False, [f"lookahead: {len(set(ta) ^ set(tb))} operaciones anteriores al corte cambian cuando se alteran datos futuros"])
    return GateResult(True)
