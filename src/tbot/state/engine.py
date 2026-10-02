"""State engine: empaqueta el mercado en un snapshot numérico compacto.

Reglas:
- Solo ventanas móviles de tamaño fijo hacia atrás (nada de EMAs con memoria infinita), así el snapshot
  calculado en vivo con unas pocas velas es idéntico al del backtest.
- Solo velas cerradas en o antes del momento de decisión: el que llama pasa `BarHistory.upto(t)`.
- "Flujo de órdenes" y "spread" se aproximan con datos de la vela (volumen con signo, posición del cierre,
  rango), porque es lo único que existe igual en el histórico y en vivo. Si el modelo usara en vivo algo que
  el backtest no tiene, el backtest mentiría.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",  # retorno log de la última vela
    "ret_4",
    "ret_16",
    "rvol_16",  # volatilidad realizada (desvío de retornos log)
    "rvol_ratio",  # volatilidad corta / larga
    "trend",  # (media 16 − media 48) / precio
    "slope",  # pendiente de la media 16 en 4 velas
    "range_pos",  # posición del cierre en el rango de 16 velas (0 = mínimo, 1 = máximo)
    "dist_hi",  # distancia al máximo de 16 velas
    "clv",  # posición del cierre dentro de la última vela (−1 a 1)
    "clv_4",
    "flow_8",  # volumen con signo / volumen total, 8 velas (presión compradora)
    "vol_z",  # volumen relativo (z-score de 16 velas)
    "rsi_14",
    "atr_pct",  # ATR(14) / precio
    "spread_proxy",  # rango medio de la vela / precio
    "tod",  # hora del día (0 = apertura, 1 = cierre)
]
WARMUP_BARS = 70  # la ventana más larga es 64 velas


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    ts: pd.Timestamp  # cierre de la última vela usada
    price: float
    atr: float  # en dólares
    values: tuple[float, ...]

    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURES, self.values, strict=True))


def features_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Calcula las features para cada vela usando solo esa vela y las anteriores."""
    o, h, low, c, v = (bars[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    logret = np.log(c / c.shift(1))
    sma16 = c.rolling(16).mean()
    sma48 = c.rolling(48).mean()
    hi16 = h.rolling(16).max()
    lo16 = low.rolling(16).min()
    rng = (h - low).replace(0, np.nan)
    clv = (((c - low) - (h - c)) / rng).fillna(0.0)
    signed = np.sign(logret.fillna(0.0)) * v
    prev_c = c.shift(1)
    tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    et = bars["start"].dt.tz_convert("America/New_York")
    tod = ((et.dt.hour * 60 + et.dt.minute) - 570) / 390.0

    out = pd.DataFrame(
        {
            "ret_1": logret,
            "ret_4": np.log(c / c.shift(4)),
            "ret_16": np.log(c / c.shift(16)),
            "rvol_16": logret.rolling(16).std(),
            "rvol_ratio": logret.rolling(16).std() / logret.rolling(64).std(),
            "trend": (sma16 - sma48) / c,
            "slope": (sma16 - sma16.shift(4)) / c,
            "range_pos": ((c - lo16) / (hi16 - lo16).replace(0, np.nan)).fillna(0.5),
            "dist_hi": c / hi16 - 1,
            "clv": clv,
            "clv_4": clv.rolling(4).mean(),
            "flow_8": signed.rolling(8).sum() / v.rolling(8).sum(),
            "vol_z": (v - v.rolling(16).mean()) / v.rolling(16).std(),
            "rsi_14": rsi.fillna(50.0) / 100.0,
            "atr_pct": tr.rolling(14).mean() / c,
            "spread_proxy": (rng.fillna(0.0) / c).rolling(4).mean(),
            "tod": tod,
        },
        index=bars.index,
    )
    out["atr"] = tr.rolling(14).mean()
    out["price"] = c
    out.iloc[: WARMUP_BARS - 1] = np.nan
    return out


def snapshot(symbol: str, bars: pd.DataFrame) -> Snapshot | None:
    """Snapshot al cierre de la última vela de `bars` (que ya tiene que ser point-in-time)."""
    if len(bars) < WARMUP_BARS:
        return None
    window = bars.iloc[-WARMUP_BARS:]
    row = features_frame(window).iloc[-1]
    vals = row[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        return None
    return Snapshot(symbol, bars["end"].iloc[-1], float(row["price"]), float(row["atr"]), tuple(float(x) for x in vals))
