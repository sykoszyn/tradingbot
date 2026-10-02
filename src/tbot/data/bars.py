"""Velas OHLCV y una vista "punto en el tiempo" que nunca devuelve datos del futuro."""

from __future__ import annotations

import pandas as pd

COLUMNS = ["start", "end", "open", "high", "low", "close", "volume"]


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if df["start"].dt.tz is None or df["end"].dt.tz is None:
        raise ValueError("Los timestamps tienen que tener zona horaria (UTC)")
    if not df["start"].is_monotonic_increasing or df["start"].duplicated().any():
        raise ValueError("Las velas tienen que estar ordenadas y sin duplicados")
    if (df["end"] <= df["start"]).any():
        raise ValueError("Cada vela tiene que terminar después de empezar")
    return df


class BarHistory:
    """Historia de un símbolo. `upto(t)` devuelve solo velas cerradas en o antes de t."""

    def __init__(self, symbol: str, bars: pd.DataFrame):
        self.symbol = symbol
        self._bars = validate_bars(bars.reset_index(drop=True))

    def upto(self, t: pd.Timestamp) -> pd.DataFrame:
        n = int(self._bars["end"].searchsorted(t, side="right"))
        return self._bars.iloc[:n]

    def __len__(self) -> int:
        return len(self._bars)
