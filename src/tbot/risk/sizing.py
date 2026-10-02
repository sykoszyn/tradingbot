"""Tamaño de posición: un cuarto de Kelly sobre la probabilidad calibrada, con topes duros."""

from __future__ import annotations

import math

from ..config import RiskLimits


def kelly_fraction(p: float, b: float) -> float:
    """Fracción de Kelly para una apuesta que gana b por cada 1 que arriesga, con probabilidad p."""
    if b <= 0 or not 0 <= p <= 1:
        return 0.0
    return max(0.0, p - (1 - p) / b)


def position_shares(
    *,
    p: float,
    b: float,
    stop_distance: float,
    price: float,
    equity: float,
    limits: RiskLimits,
    calibrated: bool,
    p_min: float,
) -> int:
    """Cantidad de acciones. Cero por debajo del umbral; tamaño mínimo fijo si la calibración no está verificada."""
    if p < p_min or price <= 0 or stop_distance <= 0 or equity <= 0:
        return 0
    cap_shares = math.floor(equity * limits.max_position_pct / price)
    if not calibrated:
        return min(cap_shares, math.floor(equity * limits.min_position_pct / price))
    f = limits.kelly_fraction * kelly_fraction(p, b)
    if f <= 0:
        return 0
    # Kelly dice cuánto del capital arriesgar (lo que se pierde si salta el stop)
    risk_dollars = f * equity
    return max(0, min(cap_shares, math.floor(risk_dollars / stop_distance)))
