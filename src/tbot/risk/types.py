from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    ref_price: float
    stop_price: float | None
    take_profit: float | None
    reason: str
    is_exit: bool = False
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex[:20])

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.ref_price


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_price: float

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.avg_price


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    positions: dict[str, Position]
    day_start_equity: float
    peak_equity: float
    trades_today: int
    now: pd.Timestamp
    market_open: bool

    @property
    def gross_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values())


@dataclass(frozen=True)
class RiskDecision:
    status: Literal["approved", "rejected", "needs_approval"]
    reasons: tuple[str, ...]
    intent: OrderIntent


@dataclass(frozen=True)
class RiskEvent:
    kind: Literal["halt_day", "kill"]
    message: str
