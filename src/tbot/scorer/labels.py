"""Etiquetas para entrenar la capa rápida. Miran el FUTURO a propósito: se usan solo para entrenar, y el
walk-forward descarta (purga) cualquier fila cuya ventana de etiqueta cruce el comienzo del período de test."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .questions import QUESTIONS


def label_span(horizon: int, max_hold: int) -> int:
    """Cuántas velas hacia adelante mira la etiqueta de la fila i."""
    return max(horizon, max_hold + 1)


def _forward_windows(a: np.ndarray, m: int) -> np.ndarray:
    """Matriz (n, m) con a[i+1 .. i+m]; NaN donde no hay datos."""
    n = len(a)
    out = np.full((n, m), np.nan)
    for j in range(1, m + 1):
        out[: n - j, j - 1] = a[j:]
    return out


def make_labels(bars: pd.DataFrame, feats: pd.DataFrame, *, horizon: int, tp_atr: float, sl_atr: float, max_hold: int) -> pd.DataFrame:
    c = bars["close"].to_numpy(float)
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    n = len(bars)
    logc = np.log(c)
    fwd_ret = np.full(n, np.nan)
    fwd_ret[: n - horizon] = logc[horizon:] - logc[: n - horizon]
    step = np.diff(logc, prepend=np.nan)
    steps = _forward_windows(step, horizon)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fwd_vol = np.nanstd(steps, axis=1, ddof=1) if horizon > 1 else np.abs(steps[:, 0])
        eff = np.abs(np.nansum(steps, axis=1)) / np.nansum(np.abs(steps), axis=1)
    rvol = feats["rvol_16"].to_numpy(float)
    atr = feats["atr"].to_numpy(float)
    flow = feats["flow_8"].to_numpy(float)

    out = pd.DataFrame(index=bars.index, columns=list(QUESTIONS), dtype=float)
    k = 0.25 * rvol * np.sqrt(horizon)
    out["direction"] = np.where(fwd_ret > k, 0, np.where(fwd_ret < -k, 1, 2))
    vr = fwd_vol / rvol
    out["regime"] = np.where(vr > 1.6, 2, np.where(eff > 0.35, 0, 1))
    out["risk_state"] = np.where(vr > 1.5, 2, np.where(vr < 0.8, 0, 1))
    half = max(1, horizon // 2)
    fwd_half = np.full(n, np.nan)
    fwd_half[: n - half] = logc[half:] - logc[: n - half]
    out["pressure_real"] = ((np.sign(fwd_half) == np.sign(flow)) & (flow != 0)).astype(float)

    # setup: entrada larga a la apertura de la vela siguiente, ¿toca TP antes que SL en max_hold velas?
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    tp = entry + tp_atr * atr
    sl = entry - sl_atr * atr
    H = _forward_windows(h, max_hold)
    L = _forward_windows(low, max_hold)
    hit_sl = L <= sl[:, None]
    hit_tp = (H >= tp[:, None]) & ~hit_sl  # si tocan los dos en la misma vela, cuenta el stop
    big = max_hold + 10
    first_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), big)
    first_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), big)
    out["setup_quality"] = (first_tp < first_sl).astype(float)

    span = label_span(horizon, max_hold)
    bad = np.zeros(n, dtype=bool)
    bad[n - span :] = True
    bad |= ~np.isfinite(rvol) | ~np.isfinite(atr) | ~np.isfinite(flow) | ~np.isfinite(fwd_ret)
    out.loc[bad, :] = np.nan
    return out
