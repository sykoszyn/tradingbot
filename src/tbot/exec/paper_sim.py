"""Broker de papel simulado en memoria: llena entradas al precio de referencia y ejecuta los brackets
(stop / take profit) con cada vela. Sirve para tests y para `tbot run --sim` sin Alpaca."""

from __future__ import annotations

import pandas as pd

from ..risk.types import OrderIntent, Position
from .broker import BrokerOrder, Fill


class PaperSimBroker:
    def __init__(self, cash: float = 100_000.0):
        self.cash = cash
        self._pos: dict[str, Position] = {}
        self._marks: dict[str, float] = {}
        self._brackets: dict[str, tuple[float, float, str]] = {}
        self._fills: list[Fill] = []
        self._seen: set[str] = set()
        self.submitted: list[OrderIntent] = []
        self.now: pd.Timestamp | None = None

    # --- utilidades de test ---
    def set_position(self, symbol: str, qty: int, price: float) -> None:
        self._pos[symbol] = Position(symbol, qty, price)
        self._marks.setdefault(symbol, price)

    def mark(self, symbol: str, price: float) -> None:
        self._marks[symbol] = price

    # --- interfaz de broker ---
    def positions(self) -> dict[str, Position]:
        return {s: Position(s, p.qty, self._marks.get(s, p.avg_price)) for s, p in self._pos.items()}

    def entry_prices(self) -> dict[str, float]:
        return {s: p.avg_price for s, p in self._pos.items()}

    def account(self) -> tuple[float, float]:
        eq = self.cash + sum(p.qty * self._marks.get(s, p.avg_price) for s, p in self._pos.items())
        return eq, self.cash

    def market_open(self, now: pd.Timestamp) -> bool:
        return True

    def drain_fills(self) -> list[Fill]:
        out, self._fills = self._fills, []
        return out

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        if intent.client_id in self._seen:
            return BrokerOrder(intent.client_id, intent.client_id, intent.symbol, intent.side, intent.qty, "duplicate")
        self._seen.add(intent.client_id)
        self.submitted.append(intent)
        price = self._marks.get(intent.symbol, intent.ref_price) if intent.is_exit else intent.ref_price
        self._fill(intent.symbol, intent.side, intent.qty, price, intent.client_id, "exit" if intent.is_exit else "entry")
        if intent.is_exit:
            self._brackets.pop(intent.symbol, None)
        elif intent.stop_price and intent.take_profit:
            self._brackets[intent.symbol] = (intent.stop_price, intent.take_profit, intent.client_id)
        return BrokerOrder(intent.client_id, intent.client_id, intent.symbol, intent.side, intent.qty, "filled")

    def on_bar(self, symbol: str, bar: pd.Series) -> None:
        self.now = bar["end"]
        if symbol in self._brackets and symbol in self._pos:
            stop, tp, cid = self._brackets[symbol]
            qty = self._pos[symbol].qty
            if bar["low"] <= stop:
                self._fill(symbol, "sell", qty, min(bar["open"], stop), cid + "-sl", "stop")
                self._brackets.pop(symbol)
            elif bar["high"] >= tp:
                self._fill(symbol, "sell", qty, max(bar["open"], tp), cid + "-tp", "take_profit")
                self._brackets.pop(symbol)
        self._marks[symbol] = float(bar["close"])

    def _fill(self, symbol: str, side: str, qty: int, price: float, cid: str, kind: str) -> None:
        signed = qty if side == "buy" else -qty
        self.cash -= signed * price
        cur = self._pos.get(symbol)
        new = (cur.qty if cur else 0) + signed
        if new == 0:
            self._pos.pop(symbol, None)
        else:
            avg = price if not cur else cur.avg_price
            self._pos[symbol] = Position(symbol, new, avg)
        self._marks[symbol] = price if kind == "entry" else self._marks.get(symbol, price)
        self._fills.append(Fill(self.now or pd.Timestamp.now(tz="UTC"), cid, symbol, side, qty, float(price), kind))
