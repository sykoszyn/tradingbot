"""Interfaz de broker. Solo `OrderRouter` la usa para mandar órdenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..risk.types import OrderIntent, Position


@dataclass(frozen=True)
class BrokerOrder:
    id: str
    client_id: str
    symbol: str
    side: str
    qty: int
    status: str


class Broker(Protocol):
    def submit_order(self, intent: OrderIntent) -> BrokerOrder: ...
    def positions(self) -> dict[str, Position]: ...


class SimBroker:
    """Broker en memoria para tests: llena todo al precio de referencia."""

    def __init__(self) -> None:
        self.orders: list[OrderIntent] = []
        self._positions: dict[str, Position] = {}
        self._seen: set[str] = set()

    def set_position(self, symbol: str, qty: int, price: float) -> None:
        self._positions[symbol] = Position(symbol, qty, price)

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        if intent.client_id in self._seen:  # idempotente, como Alpaca con client_order_id
            return BrokerOrder(intent.client_id, intent.client_id, intent.symbol, intent.side, intent.qty, "duplicate")
        self._seen.add(intent.client_id)
        self.orders.append(intent)
        signed = intent.qty if intent.side == "buy" else -intent.qty
        cur = self._positions.get(intent.symbol)
        new_qty = (cur.qty if cur else 0) + signed
        if new_qty == 0:
            self._positions.pop(intent.symbol, None)
        else:
            avg = intent.ref_price if not cur or (cur.qty > 0) != (signed > 0) else (cur.avg_price * cur.qty + intent.ref_price * signed) / new_qty
            self._positions[intent.symbol] = Position(intent.symbol, new_qty, avg)
        return BrokerOrder(intent.client_id, intent.client_id, intent.symbol, intent.side, intent.qty, "filled")
