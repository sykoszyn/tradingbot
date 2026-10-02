from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """Una entrada larga propuesta por la estrategia. El tamaño lo decide el código de riesgo."""

    p_setup: float  # probabilidad calibrada de tocar el take profit antes que el stop
    b: float  # take profit / stop
    reason: str
    probs: dict | None = None
