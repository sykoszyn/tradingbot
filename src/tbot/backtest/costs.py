from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Costos realistas para acciones de EE.UU. en Alpaca (sin comisión, con fees regulatorios y slippage)."""

    spread_bps: float = 1.0  # spread total; se paga la mitad en cada orden a mercado
    slippage_atr_k: float = 0.05  # slippage extra proporcional a la volatilidad (fracción del ATR)
    sec_fee_rate: float = 0.0000278  # SEC, solo en ventas, sobre el monto
    taf_per_share: float = 0.000166  # FINRA TAF, solo en ventas, con tope
    taf_cap: float = 8.30

    def market_slip(self, price: float, atr: float) -> float:
        return price * self.spread_bps / 2 / 1e4 + self.slippage_atr_k * atr

    def sell_fees(self, qty: int, price: float) -> float:
        return qty * price * self.sec_fee_rate + min(self.taf_cap, qty * self.taf_per_share)

DEFAULT_COSTS = CostModel()
