"""Broker real: Alpaca (paper por defecto). Entradas como órdenes bracket: el stop y el take profit quedan en
el broker, así protegen la posición aunque el bot se caiga."""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..risk.types import OrderIntent, Position
from ..secrets import Secrets
from .broker import BrokerOrder, Fill

log = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, secrets: Secrets, *, paper: bool = True, allow_live: bool = False):
        from alpaca.trading.client import TradingClient

        if not paper and not allow_live:
            raise PermissionError("Modo real bloqueado: el chequeo final (tbot golive-check) no pasó")
        self._c = TradingClient(secrets.get("ALPACA_API_KEY"), secrets.get("ALPACA_SECRET_KEY"), paper=paper)
        self.paper = paper
        self._seen_fills: set[str] = set()
        self._since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)
        self._clock_cache: tuple[float, bool] = (0.0, False)

    def account(self) -> tuple[float, float]:
        a = self._c.get_account()
        return float(a.equity), float(a.cash)

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self._c.get_all_positions():
            qty = int(float(p.qty))
            price = float(p.current_price or p.avg_entry_price)
            out[p.symbol] = Position(p.symbol, qty, price)
        return out

    def market_open(self, now: pd.Timestamp) -> bool:
        t, v = self._clock_cache
        if time.time() - t > 30:
            v = bool(self._c.get_clock().is_open)
            self._clock_cache = (time.time(), v)
        return v

    def clock(self):
        return self._c.get_clock()

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        if intent.is_exit:
            # primero se cancelan las patas abiertas del bracket, si no Alpaca rechaza la venta
            for o in self._c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[intent.symbol])):
                self._c.cancel_order_by_id(o.id)
            req = MarketOrderRequest(symbol=intent.symbol, qty=intent.qty, side=side, time_in_force=TimeInForce.DAY, client_order_id=intent.client_id)
        else:
            req = MarketOrderRequest(
                symbol=intent.symbol,
                qty=intent.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(intent.take_profit, 2)),
                stop_loss=StopLossRequest(stop_price=round(intent.stop_price, 2)),
                client_order_id=intent.client_id,
            )
        try:
            o = self._c.submit_order(req)
        except APIError as e:
            if "client_order_id" in str(e):  # ya existía: idempotencia
                o = self._c.get_order_by_client_id(intent.client_id)
            else:
                raise
        return BrokerOrder(str(o.id), o.client_order_id, o.symbol, str(o.side), int(float(o.qty or 0)), str(o.status))

    def drain_fills(self) -> list[Fill]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        out: list[Fill] = []
        orders = self._c.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=self._since.to_pydatetime(), nested=True, limit=500))
        for parent in orders:
            for o in [parent, *(parent.legs or [])]:
                fq = float(o.filled_qty or 0)
                key = f"{o.id}:{fq}"
                if fq <= 0 or str(o.status).lower().endswith("partially_filled") or key in self._seen_fills:
                    continue
                self._seen_fills.add(key)
                t = str(o.type).lower()
                kind = "entry" if str(o.side).lower().endswith("buy") else ("stop" if "stop" in t else "take_profit" if "limit" in t else "exit")
                cid = o.client_order_id if o is parent else f"{parent.client_order_id}-{kind}"
                out.append(Fill(pd.Timestamp(o.filled_at), cid, o.symbol, "buy" if kind == "entry" else "sell", int(fq), float(o.filled_avg_price), kind))
        return out
