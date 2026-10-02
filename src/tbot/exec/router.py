"""El único camino por el que sale una orden: siempre pasa por el motor de riesgo."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

import pandas as pd

from ..risk.limits import RiskEngine
from ..risk.types import AccountState, OrderIntent, RiskDecision
from .broker import Broker, BrokerOrder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    status: str
    reasons: tuple[str, ...]
    intent: OrderIntent
    order: BrokerOrder | None = None
    pending_id: str | None = None


@dataclass
class _Pending:
    intent: OrderIntent
    created: pd.Timestamp


class OrderRouter:
    def __init__(self, broker: Broker, risk: RiskEngine, on_event: Callable[[str, dict], None] | None = None):
        self._broker = broker
        self.risk = risk
        self.pending: dict[str, _Pending] = {}
        self.on_event = on_event or (lambda kind, data: None)

    def _send(self, decision: RiskDecision) -> RouteResult:
        order = self._broker.submit_order(decision.intent)
        self.on_event("order", {"intent": decision.intent, "order": order})
        return RouteResult("approved", decision.reasons, decision.intent, order)

    def submit(self, intent: OrderIntent, account: AccountState) -> RouteResult:
        d = self.risk.check(intent, account)
        if d.status == "approved":
            return self._send(d)
        if d.status == "needs_approval":
            pid = uuid.uuid4().hex[:10]
            self.pending[pid] = _Pending(intent, account.now)
            self.on_event("approval_needed", {"pending_id": pid, "intent": intent, "reasons": d.reasons})
            return RouteResult("needs_approval", d.reasons, intent, pending_id=pid)
        self.on_event("rejected", {"intent": intent, "reasons": d.reasons})
        return RouteResult("rejected", d.reasons, intent)

    def approve(self, pending_id: str, account: AccountState) -> RouteResult:
        p = self.pending.pop(pending_id, None)
        if p is None:
            raise KeyError("No existe esa aprobación pendiente")
        if account.now - p.created > pd.Timedelta(minutes=self.risk.limits.approval_timeout_minutes):
            return RouteResult("rejected", ("la aprobación venció",), p.intent)
        # se vuelve a chequear todo con el estado actual: aprobar no saltea ningún otro límite
        d = self.risk.check(p.intent, account, approved_manually=True)
        if d.status != "approved":
            return RouteResult("rejected", d.reasons, p.intent)
        return self._send(d)

    def reject(self, pending_id: str) -> None:
        self.pending.pop(pending_id, None)

    def kill(self, reason: str, account: AccountState) -> list[RouteResult]:
        """Dispara el kill switch y cierra todas las posiciones."""
        self.risk.killswitch.fire(reason)
        self.pending.clear()
        self.on_event("kill", {"reason": reason})
        results = []
        for pos in self._broker.positions().values():
            exit_ = OrderIntent(pos.symbol, "sell" if pos.qty > 0 else "buy", abs(pos.qty), pos.avg_price, None, None, f"kill: {reason}", is_exit=True)
            acct = replace(account, positions=self._broker.positions(), market_open=True)
            d = self.risk.check(exit_, acct)
            if d.status == "approved":
                results.append(self._send(d))
            else:
                log.error("No se pudo cerrar %s en el kill switch: %s", pos.symbol, d.reasons)
        return results
