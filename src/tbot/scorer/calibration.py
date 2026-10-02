"""Calibración: Brier, curva de confiabilidad, ECE y recalibración isotónica con fills propios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "count": int(m.sum()), "mean_pred": float(p[m].mean()) if m.any() else np.nan, "observed": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado: diferencia media (ponderada) entre lo predicho y lo observado."""
    curve = reliability_curve(p, y, bins)
    curve = curve[curve["count"] > 0]
    n = curve["count"].sum()
    return float((curve["count"] * (curve["mean_pred"] - curve["observed"]).abs()).sum() / n) if n else float("nan")


class Recalibrator:
    """Corrige probabilidades con lo observado en fills reales. Solo se activa con suficientes datos."""

    def __init__(self, min_samples: int = 200):
        self.min_samples = min_samples
        self._iso: IsotonicRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> Recalibrator:
        if len(p) >= self.min_samples:
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    @property
    def active(self) -> bool:
        return self._iso is not None

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(p, float)) if self._iso is not None else np.asarray(p, float)
