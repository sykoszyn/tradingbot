"""Walk-forward: el scorer se entrena solo con el pasado (purgado) y se evalúa en el período siguiente."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import RiskLimits
from ..scorer.labels import label_span, make_labels
from ..scorer.model import CalibratedScorer
from ..scorer.questions import QUESTIONS
from ..state.engine import FEATURES, features_frame
from ..strategy.spec import StrategySpec
from .costs import DEFAULT_COSTS, CostModel
from .engine import BacktestResult, Decide, run_backtest
from .metrics import compute_metrics


@dataclass
class Fold:
    train_label_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    result: BacktestResult
    metrics: dict
    oos_start: pd.Timestamp
    final_scorer: CalibratedScorer | None


def _training_rows(bars, feats, labels, cutoff: pd.Timestamp, span: int):
    """Filas cuya etiqueta termina estrictamente antes de `cutoff` (purga)."""
    X, Y, last = [], [], None
    for s in bars:
        b, f, lab = bars[s], feats[s], labels[s]
        ends = b["end"].to_numpy()
        label_end_idx = np.minimum(np.arange(len(b)) + span, len(b) - 1)
        label_end = ends[label_end_idx]
        ok = np.asarray(pd.to_datetime(label_end, utc=True) < cutoff)
        ok &= f[FEATURES].notna().all(axis=1).to_numpy() & lab.notna().all(axis=1).to_numpy()
        if ok.any():
            X.append(f.loc[ok, FEATURES].to_numpy(float))
            Y.append(lab.loc[ok])
            le = pd.to_datetime(label_end[ok], utc=True).max()
            last = le if last is None else max(last, le)
    if not X:
        return None, None, None
    return np.vstack(X), pd.concat(Y, ignore_index=True), last


def walk_forward(
    bars_by_symbol: dict[str, pd.DataFrame],
    spec: StrategySpec,
    *,
    train_days: int = 250,
    test_days: int = 63,
    costs: CostModel = DEFAULT_COSTS,
    limits: RiskLimits | None = None,
    decide: Decide | None = None,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] = features_frame,
    calibrated: bool = True,
    seed: int = 0,
    fit_final: bool = False,
) -> WalkForwardResult:
    bars = {s: b.reset_index(drop=True) for s, b in bars_by_symbol.items()}
    feats = {s: feature_fn(b) for s, b in bars.items()}
    span = label_span(spec.horizon_bars, spec.max_hold_bars)
    labels = {s: make_labels(bars[s], feats[s], horizon=spec.horizon_bars, tp_atr=spec.tp_atr, sl_atr=spec.sl_atr, max_hold=spec.max_hold_bars) for s in bars}

    days = sorted(set().union(*[set(b["start"].dt.tz_convert("America/New_York").dt.date) for b in bars.values()]))
    if len(days) < train_days + test_days:
        raise ValueError(f"Hacen falta al menos {train_days + test_days} días de datos (hay {len(days)})")
    to_ts = lambda d: pd.Timestamp(d).tz_localize("America/New_York").tz_convert("UTC")  # noqa: E731

    probs = {s: {q: np.full((len(b), len(c)), np.nan) for q, c in QUESTIONS.items()} for s, b in bars.items()}
    folds: list[Fold] = []
    k = train_days
    while k < len(days):
        test_start = to_ts(days[k])
        test_end = to_ts(days[min(k + test_days, len(days) - 1)]) if k + test_days < len(days) else pd.Timestamp.max.tz_localize("UTC")
        X, Y, last = _training_rows(bars, feats, labels, test_start, span)
        if X is not None and len(X) > 500:
            sc = CalibratedScorer(seed=seed).fit(X, Y)
            for s, b in bars.items():
                m = ((b["end"] > test_start) & (b["end"] <= test_end)).to_numpy() & feats[s][FEATURES].notna().all(axis=1).to_numpy()
                if m.any():
                    out = sc.score_batch(feats[s].loc[m, FEATURES].to_numpy(float))
                    for q in QUESTIONS:
                        probs[s][q][m] = out[q]
            folds.append(Fold(last, test_start, test_end, len(X)))
        k += test_days

    oos_start = folds[0].test_start if folds else to_ts(days[train_days])
    res = run_backtest(bars, spec, decide=decide, probs_by_symbol=probs, costs=costs, limits=limits, calibrated=calibrated, start_ts=oos_start, feature_fn=feature_fn)
    metrics = compute_metrics(res, start=oos_start)
    final = None
    if fit_final:
        X, Y, _ = _training_rows(bars, feats, labels, pd.Timestamp.max.tz_localize("UTC"), span)
        final = CalibratedScorer(seed=seed).fit(X, Y) if X is not None else None
    return WalkForwardResult(folds, res, metrics, oos_start, final)
