"""Motor de riesgo. Todo lo que dice "no" vive acá, en código; ningún modelo lo puede saltear."""

from __future__ import annotations

from ..config import RiskLimits
from .killswitch import KillSwitch
from .types import AccountState, OrderIntent, RiskDecision, RiskEvent


class RiskEngine:
    def __init__(self, limits: RiskLimits, killswitch: KillSwitch, *, universe: set[str], mode: str, golive_ok: bool = False):
        if mode not in ("paper", "live"):
            raise ValueError("mode inválido")
        self.limits = limits
        self.killswitch = killswitch
        self.universe = universe
        self.mode = mode
        self.golive_ok = golive_ok

    def daily_loss_hit(self, a: AccountState) -> bool:
        return a.equity <= a.day_start_equity * (1 - self.limits.daily_loss_limit_pct)

    def drawdown_hit(self, a: AccountState) -> bool:
        return a.equity <= a.peak_equity * (1 - self.limits.max_drawdown_pct)

    def on_equity(self, a: AccountState) -> list[RiskEvent]:
        """Se llama con cada actualización de capital. Dispara el kill switch si hace falta."""
        events: list[RiskEvent] = []
        if self.drawdown_hit(a) and not self.killswitch.active:
            msg = f"Caída de {1 - a.equity / a.peak_equity:.1%} desde el pico (límite {self.limits.max_drawdown_pct:.0%})"
            self.killswitch.fire(msg)
            events.append(RiskEvent("kill", msg))
        elif self.daily_loss_hit(a):
            events.append(RiskEvent("halt_day", f"Pérdida diaria de {1 - a.equity / a.day_start_equity:.1%}: no hay más entradas hoy"))
        return events

    def check(self, intent: OrderIntent, a: AccountState, *, approved_manually: bool = False) -> RiskDecision:
        L = self.limits
        reasons: list[str] = []
        pos = a.positions.get(intent.symbol)

        if intent.qty <= 0:
            reasons.append("cantidad inválida")
        if not a.market_open:
            reasons.append("mercado cerrado")

        if intent.is_exit:
            # salir siempre está permitido (incluso con kill switch), pero solo para reducir
            if pos is None or intent.qty > abs(pos.qty) or (intent.side == "sell") != (pos.qty > 0):
                reasons.append("la salida no reduce una posición existente")
            return RiskDecision("rejected" if reasons else "approved", tuple(reasons), intent)

        if self.killswitch.active:
            reasons.append(f"kill switch activo: {self.killswitch.reason()}")
        if self.mode == "live" and not self.golive_ok:
            reasons.append("modo real bloqueado: el chequeo final no pasó")
        if intent.symbol not in self.universe:
            reasons.append(f"{intent.symbol} no está en el universo permitido")
        if intent.side != "buy":
            reasons.append("v1 opera solo en largo")
        if intent.stop_price is None or intent.stop_price >= intent.ref_price:
            reasons.append("toda entrada necesita un stop por debajo del precio")
        if self.daily_loss_hit(a):
            reasons.append("límite de pérdida diaria alcanzado")
        if self.drawdown_hit(a):
            reasons.append("caída máxima alcanzada")
        if a.trades_today >= L.max_trades_per_day:
            reasons.append("máximo de operaciones del día")
        current = pos.notional if pos else 0.0
        if current + intent.notional > a.equity * L.max_position_pct + 1e-6:
            reasons.append(f"tamaño de posición supera {L.max_position_pct:.0%} del capital")
        if a.gross_exposure + intent.notional > a.equity * L.max_gross_exposure_pct + 1e-6:
            reasons.append(f"exposición total supera {L.max_gross_exposure_pct:.0%} del capital")

        if reasons:
            return RiskDecision("rejected", tuple(reasons), intent)
        if intent.notional > L.manual_approval_usd and not approved_manually:
            return RiskDecision("needs_approval", (f"operación de US${intent.notional:,.0f} supera US${L.manual_approval_usd:,.0f}",), intent)
        return RiskDecision("approved", (), intent)
