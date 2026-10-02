"""Reglas de entrada por plantilla + combinación explícita de las probabilidades de la capa rápida."""

from __future__ import annotations

import math
from collections.abc import Callable

from .signal import Signal
from .spec import StrategySpec

Feats = dict[str, float]
Probs = dict[str, dict[str, float]]


def _trend(f: Feats, p: dict[str, float]) -> bool:
    return f["trend"] > p["trend_min"] and f["slope"] > p["slope_min"]


def _meanrev(f: Feats, p: dict[str, float]) -> bool:
    return f["rsi_14"] < p["rsi_max"] and f["range_pos"] < p["range_pos_max"]


def _breakout(f: Feats, p: dict[str, float]) -> bool:
    return f["dist_hi"] > p["dist_hi_min"] and f["vol_z"] > p["vol_z_min"] and f["flow_8"] > p["flow_min"]


RULES: dict[str, Callable[[Feats, dict[str, float]], bool]] = {"trend": _trend, "meanrev": _meanrev, "breakout": _breakout}


def make_decider(spec: StrategySpec, bars_per_day: int = 26) -> Callable[[str, int, Feats, Probs | None], Signal | None]:
    rule = RULES[spec.template]
    w_total = sum(spec.weights.values())

    def decide(symbol: str, i: int, feats: Feats, probs: Probs | None) -> Signal | None:
        if probs is None or not all(math.isfinite(v) for v in feats.values()):
            return None
        bar_of_day = round(feats["tod"] * bars_per_day)
        if bar_of_day < spec.no_entry_first_bars or bar_of_day >= bars_per_day - spec.no_entry_last_bars:
            return None
        if not rule(feats, spec.params):
            return None
        if probs["risk_state"]["stress"] > spec.max_stress:
            return None
        combined = 0.0
        for q, th in spec.thresholds.items():
            v = probs[q][th.cls]
            if v < th.min:
                return None
            combined += spec.weights[q] * v
        if combined / w_total < spec.combined_min:
            return None
        return Signal(p_setup=probs["setup_quality"]["success"], b=spec.payoff, reason=f"{spec.template}: combinado {combined / w_total:.2f}", probs=probs)

    return decide
