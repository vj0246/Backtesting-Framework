"""Order and fill records.

Orders are created by strategies through ``BarContext`` and consumed by the
event-driven engine. Fills are produced by the fill model and applied to the
ledger. Both are plain records so they serialize to a blotter without ceremony.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import count

import pandas as pd


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    """``DAY``: cancelled if not filled at the next bar.

    ``GTC``: persists until filled, cancelled, or expired via ``Order.max_bars``.
    """

    DAY = "day"
    GTC = "gtc"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


_order_ids = count(1)


@dataclass
class Order:
    """A request to change a position.

    Either ``quantity`` (signed units, negative = sell) or ``target_weight``
    (fraction of equity) is given. Target-weight orders are sized at fill
    time from the equity and price at execution, which is how a rebalance is
    actually executed and what lets the two engines agree exactly.
    """

    symbol: str
    quantity: float = float("nan")
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    target_weight: float | None = None
    created_at: pd.Timestamp | None = None
    created_bar: int = -1
    max_bars: int | None = None
    id: int = field(default_factory=lambda: next(_order_ids))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.target_weight is None:
            if not math.isfinite(self.quantity) or self.quantity == 0:
                raise ValueError("order quantity must be a non-zero finite number")
        else:
            if not math.isfinite(self.target_weight):
                raise ValueError("target_weight must be finite")
            if self.order_type is not OrderType.MARKET:
                raise ValueError("target-weight orders must be market orders")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders need limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop orders need stop_price")

    @property
    def is_target(self) -> bool:
        return self.target_weight is not None

    @property
    def remaining(self) -> float:
        if math.isnan(self.quantity):
            return float("nan")
        return self.quantity - self.filled_quantity

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)


@dataclass(frozen=True)
class Fill:
    """An execution against an order.

    ``price`` is what was paid per unit including slippage; ``reference_price``
    is the bar price the fill was derived from (open for next-open market
    orders). ``commission`` and ``slippage_cost`` are absolute cash amounts.
    """

    order_id: int
    symbol: str
    timestamp: pd.Timestamp
    bar: int
    quantity: float
    price: float
    reference_price: float
    commission: float
    slippage_cost: float
    participation: float

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.reference_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost


def fills_to_frame(fills: list[Fill]) -> pd.DataFrame:
    columns = [
        "timestamp",
        "bar",
        "order_id",
        "symbol",
        "quantity",
        "price",
        "reference_price",
        "commission",
        "slippage_cost",
        "participation",
    ]
    if not fills:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "timestamp": f.timestamp,
            "bar": f.bar,
            "order_id": f.order_id,
            "symbol": f.symbol,
            "quantity": f.quantity,
            "price": f.price,
            "reference_price": f.reference_price,
            "commission": f.commission,
            "slippage_cost": f.slippage_cost,
            "participation": f.participation,
        }
        for f in fills
    ]
    return pd.DataFrame(rows, columns=columns)


def orders_to_frame(orders: list[Order]) -> pd.DataFrame:
    columns = [
        "id",
        "created_at",
        "created_bar",
        "symbol",
        "quantity",
        "target_weight",
        "order_type",
        "limit_price",
        "stop_price",
        "time_in_force",
        "status",
        "filled_quantity",
        "reason",
    ]
    if not orders:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "id": o.id,
            "created_at": o.created_at,
            "created_bar": o.created_bar,
            "symbol": o.symbol,
            "quantity": o.quantity,
            "target_weight": o.target_weight,
            "order_type": o.order_type.value,
            "limit_price": o.limit_price,
            "stop_price": o.stop_price,
            "time_in_force": o.time_in_force.value,
            "status": o.status.value,
            "filled_quantity": o.filled_quantity,
            "reason": o.reason,
        }
        for o in orders
    ]
    return pd.DataFrame(rows, columns=columns)
