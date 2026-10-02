"""Preguntas de resultado fijo que la capa rápida responde en cada vela (una sola llamada).

Cada pregunta aísla un factor; el código las combina con pesos explícitos definidos en strategy.md.
"""

QUESTIONS: dict[str, tuple[str, ...]] = {
    "regime": ("trend", "range", "high_vol"),  # régimen de las próximas velas
    "direction": ("up", "down", "flat"),  # dirección en el horizonte
    "pressure_real": ("no", "yes"),  # ¿la presión compradora/vendedora actual continúa?
    "setup_quality": ("fail", "success"),  # ¿una entrada larga ahora toca el take profit antes que el stop?
    "risk_state": ("calm", "elevated", "stress"),  # volatilidad que viene vs. la reciente
}
