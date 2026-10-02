"""Especificación de estrategia: lo único que Opus puede proponer. Esquema cerrado y validado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..scorer.questions import QUESTIONS

Template = Literal["trend", "meanrev", "breakout"]

TEMPLATE_REGIME = {"trend": "trend", "meanrev": "range", "breakout": "trend"}

DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "trend": {"trend_min": 0.001, "slope_min": 0.0},
    "meanrev": {"rsi_max": 0.35, "range_pos_max": 0.25},
    "breakout": {"dist_hi_min": -0.0005, "vol_z_min": 1.0, "flow_min": 0.1},
}


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cls: str
    min: float = Field(ge=0, le=1)


class Invalidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rolling_trades: int = Field(30, ge=10, le=200)
    min_hit_rate: float = Field(0.45, ge=0, le=1)
    max_strategy_drawdown: float = Field(0.08, gt=0, le=0.15)


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=60)
    template: Template
    symbols: list[str] = Field(min_length=1, max_length=10)
    timeframe_minutes: int = 15
    horizon_bars: int = Field(8, ge=2, le=64)
    tp_atr: float = Field(1.5, ge=0.3, le=6)
    sl_atr: float = Field(1.0, ge=0.3, le=4)
    max_hold_bars: int = Field(16, ge=1, le=130)
    params: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, Threshold] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    combined_min: float = Field(0.4, ge=0, le=1)
    max_stress: float = Field(0.35, ge=0, le=1)
    flatten_eod: bool = True
    no_entry_first_bars: int = Field(1, ge=0, le=10)
    no_entry_last_bars: int = Field(2, ge=0, le=10)
    flatten_before_events: bool = False
    invalidation: Invalidation = Field(default_factory=Invalidation)
    notes: str = Field("", max_length=2000)
    version: int = 1

    @model_validator(mode="after")
    def _fill_and_check(self) -> StrategySpec:
        self.params = {**DEFAULT_PARAMS[self.template], **self.params}
        unknown = set(self.params) - set(DEFAULT_PARAMS[self.template])
        if unknown:
            raise ValueError(f"Parámetros desconocidos para {self.template}: {sorted(unknown)}")
        b = self.tp_atr / self.sl_atr
        breakeven = 1 / (1 + b)
        defaults = {
            "setup_quality": Threshold(cls="success", min=round(breakeven + 0.05, 3)),
            "direction": Threshold(cls="up", min=0.34),
            "regime": Threshold(cls=TEMPLATE_REGIME[self.template], min=0.30),
            "pressure_real": Threshold(cls="yes", min=0.45),
        }
        self.thresholds = {**defaults, **self.thresholds}
        for q, th in self.thresholds.items():
            if q not in QUESTIONS or th.cls not in QUESTIONS[q]:
                raise ValueError(f"Umbral inválido: {q}/{th.cls}")
        if self.thresholds["setup_quality"].min <= breakeven:
            raise ValueError(f"El umbral de setup ({self.thresholds['setup_quality'].min}) tiene que superar el punto de equilibrio {breakeven:.3f}")
        self.weights = {q: float(self.weights.get(q, 1.0)) for q in self.thresholds}
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Los pesos tienen que ser positivos")
        if self.timeframe_minutes not in (5, 15, 30, 60):
            raise ValueError("timeframe_minutes tiene que ser 5, 15, 30 o 60")
        return self

    @property
    def payoff(self) -> float:
        return self.tp_atr / self.sl_atr

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path) -> StrategySpec:
        return StrategySpec.model_validate(json.loads(Path(path).read_text()))


def render_markdown(spec: StrategySpec, metrics: dict | None = None, gate: dict | None = None) -> str:
    th = spec.thresholds
    rule = {
        "trend": f"tendencia > {spec.params['trend_min']:.4f} (media 16 vs 48) y pendiente > {spec.params['slope_min']:.4f}",
        "meanrev": f"RSI(14) < {spec.params['rsi_max'] * 100:.0f} y cierre en el {spec.params['range_pos_max']:.0%} inferior del rango de 16 velas",
        "breakout": f"cierre a menos de {abs(spec.params['dist_hi_min']):.2%} del máximo de 16 velas, volumen z > {spec.params['vol_z_min']:.1f} y flujo > {spec.params['flow_min']:.2f}",
    }[spec.template]
    lines = [
        f"# Estrategia: {spec.name} (v{spec.version})",
        "",
        f"- **Plantilla:** {spec.template} · **Símbolos:** {', '.join(spec.symbols)} · **Timeframe:** {spec.timeframe_minutes} minutos",
        f"- **Entrada (largo):** al cierre de la vela, si {rule}, y **cada** probabilidad supera su umbral:",
    ]
    for q, t in th.items():
        lines.append(f"  - `{q}` = `{t.cls}` ≥ {t.min:.2f} (peso {spec.weights[q]:g})")
    lines += [
        f"  - puntaje combinado (promedio ponderado) ≥ {spec.combined_min:.2f}, y `risk_state` = `stress` ≤ {spec.max_stress:.2f}",
        f"  - no entra en las primeras {spec.no_entry_first_bars} ni en las últimas {spec.no_entry_last_bars} velas del día",
        "- **Ejecución:** orden a la apertura de la vela siguiente, con stop y take profit en el broker (bracket).",
        f"- **Stop:** {spec.sl_atr:g} × ATR(14) debajo de la entrada.",
        f"- **Take profit:** {spec.tp_atr:g} × ATR(14) arriba de la entrada (relación {spec.payoff:.2f}).",
        f"- **Salida por tiempo:** después de {spec.max_hold_bars} velas" + (", y siempre antes del cierre del día." if spec.flatten_eod else "."),
        "- **Tamaño:** ¼ de Kelly sobre la probabilidad calibrada de `setup_quality`, con los topes de config/risk.toml; tamaño mínimo fijo hasta verificar la calibración con fills propios.",
        f"- **Invalidación (se desactiva sola y avisa):** hit rate < {spec.invalidation.min_hit_rate:.0%} en las últimas {spec.invalidation.rolling_trades} operaciones, o caída de la estrategia > {spec.invalidation.max_strategy_drawdown:.0%}.",
    ]
    if spec.flatten_before_events:
        lines.append("- **Eventos macro:** cierra todo antes de los eventos de config/events.csv y no entra ese día.")
    if metrics:
        lines += ["", "## Resultado fuera de muestra (walk-forward, con costos)", "", "| Métrica | Valor |", "|---|---|"]
        for k in ("sharpe", "max_drawdown", "hit_rate", "t_stat", "n_trades", "years", "cagr"):
            if k in metrics:
                v = metrics[k]
                lines.append(f"| {k} | {v:.1%} |" if k in ("max_drawdown", "hit_rate", "cagr") else f"| {k} | {v:.2f} |")
    if gate:
        lines += ["", f"**Filtro:** {'PASÓ' if gate.get('passed') else 'NO PASÓ'}"]
        lines += [f"- {f}" for f in gate.get("failures", [])]
    if spec.notes:
        lines += ["", "## Notas", "", spec.notes]
    return "\n".join(lines) + "\n"
