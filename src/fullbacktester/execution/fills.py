"""Fill model: turns open orders into fills against one bar's OHLCV.

Rules, all deliberately conservative:

* Market orders fill at the bar's open (or close, under ``FillTiming.SAME_CLOSE``).
* Limit buys fill only if the bar's low reached the limit, at ``min(open, limit)``;
  limit sells only if the high reached it, at ``max(open, limit)``.
* Stop buys trigger if the high reached the stop, filling at ``max(open, stop)``;
  stop sells trigger if the low reached it, filling at ``min(open, stop)``.
* A symbol with no bar (NaN open) cannot trade.
* ``max_participation`` caps units per fill at a fraction of bar volume; the rest
  stays open (GTC) or is cancelled (DAY).
* Target-weight orders are sized against equity marked at the execution price,
  simultaneously for every order in the bar, so a full rebalance is one decision.
* Slippage moves the price against the trade; commission is charged in cash.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fullbacktester.data.panel import Panel
from fullbacktester.execution.config import EngineConfig
from fullbacktester.execution.ledger import Ledger
from fullbacktester.execution.orders import Fill, Order, OrderStatus, OrderType, TimeInForce

_EPS = 1e-12


class FillModel:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def fill_bar(
        self,
        orders: list[Order],
        panel: Panel,
        bar: int,
        ledger: Ledger,
        marks: np.ndarray,
        *,
        at_close: bool = False,
    ) -> tuple[list[Fill], list[Order]]:
        """Attempt every open order at ``bar``.

        ``marks`` are last-known closes used to value symbols with no bar this
        period. Returns ``(fills, orders still open)``; orders that were cancelled,
        rejected, or filled are not returned but keep their final status.
        """
        config = self.config
        cost_model = config.cost_model
        timestamp = panel.timestamps[bar]
        opens = panel.open[bar]
        highs = panel.high[bar]
        lows = panel.low[bar]
        closes = panel.close[bar]
        volumes = panel.volume[bar]

        execution_marks = np.where(
            np.isfinite(closes if at_close else opens), closes if at_close else opens, marks
        )
        equity_ref = ledger.equity(execution_marks)
        positions_ref = ledger.positions.copy()

        fills: list[Fill] = []
        still_open: list[Order] = []

        for order in orders:
            if not order.is_open:
                continue
            j = ledger.index_of(order.symbol)
            reference, why = _reference_price(
                order, opens[j], highs[j], lows[j], closes[j], at_close=at_close
            )
            if reference is None:
                self._defer_or_cancel(order, bar, why, still_open)
                continue

            if order.is_target:
                assert order.target_weight is not None
                weight = order.target_weight
                if not config.allow_short and weight < 0:
                    weight = 0.0
                desired = weight * equity_ref / reference - positions_ref[j]
                if math.isnan(order.quantity):
                    order.quantity = desired
                quantity = config.round_quantity(desired)
            else:
                quantity = config.round_quantity(order.remaining)

            if abs(quantity) < _EPS:
                order.status = OrderStatus.FILLED
                order.reason = order.reason or "already at target"
                continue

            if not config.allow_short and ledger.positions[j] + quantity < -_EPS:
                order.status = OrderStatus.REJECTED
                order.reason = "short selling disabled"
                continue

            quantity, capped = self._apply_participation_cap(quantity, volumes[j])
            if abs(quantity) < _EPS:
                self._defer_or_cancel(
                    order, bar, "participation cap leaves nothing tradeable", still_open
                )
                continue

            quantity, leverage_hit = self._apply_leverage_cap(
                quantity, j, ledger, execution_marks, equity_ref
            )
            if abs(quantity) < _EPS:
                order.status = OrderStatus.REJECTED
                order.reason = "gross leverage cap"
                continue

            participation = float(cost_model.participation(quantity, volumes[j]))
            price = float(cost_model.fill_price(quantity, reference, participation))
            commission = float(cost_model.commission.commission(quantity, reference))
            slippage_cost = float(cost_model.slippage_cost(quantity, reference, participation))
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                timestamp=timestamp,
                bar=bar,
                quantity=quantity,
                price=price,
                reference_price=float(reference),
                commission=commission,
                slippage_cost=slippage_cost,
                participation=participation,
            )
            ledger.apply_fill(fill)
            fills.append(fill)
            order.filled_quantity += quantity

            fully_filled = abs(order.remaining) < _EPS or (
                not config.fractional_shares and abs(order.remaining) < config.lot_size
            )
            if fully_filled or leverage_hit:
                order.status = OrderStatus.FILLED
                if leverage_hit:
                    order.reason = "reduced to gross leverage cap"
            elif capped:
                order.status = OrderStatus.PARTIAL
                self._defer_or_cancel(order, bar, "participation cap", still_open)
            else:
                order.status = OrderStatus.FILLED

        return fills, still_open

    # ------------------------------------------------------------------ helpers

    def _defer_or_cancel(self, order: Order, bar: int, why: str, still_open: list[Order]) -> None:
        if order.time_in_force is TimeInForce.DAY:
            order.status = OrderStatus.CANCELLED
            order.reason = why
            return
        if order.max_bars is not None and bar - order.created_bar >= order.max_bars:
            order.status = OrderStatus.CANCELLED
            order.reason = f"expired after {order.max_bars} bars ({why})"
            return
        still_open.append(order)

    def _apply_participation_cap(self, quantity: float, volume: float) -> tuple[float, bool]:
        cap = self.config.max_participation
        if cap is None or not np.isfinite(volume):
            return quantity, False
        max_units = self.config.round_quantity(cap * volume)
        if abs(quantity) <= max_units:
            return quantity, False
        return math.copysign(max_units, quantity), True

    def _apply_leverage_cap(
        self,
        quantity: float,
        j: int,
        ledger: Ledger,
        marks: np.ndarray,
        equity: float,
    ) -> tuple[float, bool]:
        cap = self.config.max_gross_leverage
        if cap is None:
            return quantity, False
        current = ledger.positions[j]
        proposed = current + quantity
        if abs(proposed) <= abs(current):
            return quantity, False  # reducing exposure is always allowed
        others = ledger.gross_notional(marks) - abs(current * marks[j])
        room = cap * equity - others
        max_abs_units = max(room, 0.0) / marks[j]
        if abs(proposed) <= max_abs_units + _EPS:
            return quantity, False
        allowed_proposed = math.copysign(max_abs_units, proposed)
        capped = self.config.round_quantity(allowed_proposed - current)
        if math.copysign(1.0, capped) != math.copysign(1.0, quantity):
            return 0.0, True
        return capped, True


def _reference_price(
    order: Order,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    at_close: bool,
) -> tuple[float | None, str]:
    if not np.isfinite(open_):
        return None, "no bar for symbol"
    if order.order_type is OrderType.MARKET:
        return (close if at_close else open_), ""
    if order.order_type is OrderType.LIMIT:
        assert order.limit_price is not None
        limit = order.limit_price
        if order.is_buy:
            return (min(open_, limit), "") if low <= limit else (None, "limit not reached")
        return (max(open_, limit), "") if high >= limit else (None, "limit not reached")
    assert order.stop_price is not None
    stop = order.stop_price
    if order.is_buy:
        return (max(open_, stop), "") if high >= stop else (None, "stop not triggered")
    return (min(open_, stop), "") if low <= stop else (None, "stop not triggered")


def unfilled_summary(orders: list[Order]) -> pd.Series:
    """Count of orders by final status, for reports."""
    counts: dict[str, int] = {}
    for order in orders:
        counts[order.status.value] = counts.get(order.status.value, 0) + 1
    return pd.Series(counts, dtype="int64")
