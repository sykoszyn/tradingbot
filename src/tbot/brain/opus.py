"""Cliente de Opus 5.5 para el cerebro lento: una llamada por pedido, salida JSON con esquema, tope de gasto."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
PRICES = {"claude-opus-5-5": (4.0, 20.0)}  # US$ por millón de tokens (entrada, salida)


class BudgetExceeded(RuntimeError):
    pass


class OpusRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class OpusResponse:
    data: dict
    cost_usd: float


class Brain(Protocol):
    def ask(self, system: str, user: str, schema: dict, max_usd: float) -> OpusResponse: ...


class OpusClient:
    def __init__(self, api_key: str, model: str = MODEL, effort: str = "high", max_tokens: int = 32000):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.spent = 0.0

    def ask(self, system: str, user: str, schema: dict, max_usd: float) -> OpusResponse:
        p_in, p_out = PRICES.get(self.model, (5.0, 25.0))
        worst_case = (len(system) + len(user)) / 3.0 / 1e6 * p_in + self.max_tokens / 1e6 * p_out
        if self.spent + worst_case > max_usd:
            raise BudgetExceeded(f"Gastado US${self.spent:.2f}; la próxima llamada podría superar el tope de US${max_usd:.2f}")
        # Si el modelo declina el pedido, la API lo reintenta con el modelo alternativo recomendado.
        with self._client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        u = msg.usage
        cost = ((u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)) / 1e6 * p_in + (getattr(u, "cache_read_input_tokens", 0) or 0) / 1e6 * p_in * 0.1 + (u.output_tokens or 0) / 1e6 * p_out
        self.spent += cost
        if msg.stop_reason == "refusal":
            raise OpusRefused("El modelo declinó el pedido")
        if msg.stop_reason == "max_tokens":
            raise ValueError("La respuesta de Opus se cortó por max_tokens")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return OpusResponse(json.loads(text), cost)
