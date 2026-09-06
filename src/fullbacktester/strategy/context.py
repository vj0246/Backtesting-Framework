"""What a strategy can see and do on one bar."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fullbacktester.data.panel import PanelView
from fullbacktester.execution.config import EngineConfig
from fullbacktester.execution.ledger import PortfolioSnapshot
from fullbacktester.execution.orders import Order, OrderStatus, OrderType, TimeInForce


class BarContext:
    """Point-in-time view of the market plus the portfolio, and an order API.

    ``view`` ends at the current bar. ``portfolio`` is marked at the current
    bar's close. Orders submitted here execute according to the engine's fill
    timing, by default at the next bar's open.
    """

    __slots__ = ("_config", "_marks", "_orders", "_view", "cancel_requested", "notes", "portfolio")

    def __init__(
        self,
        view: PanelView,
        portfolio: PortfolioSnapshot,
        config: EngineConfig,
        marks: np.ndarray,
    ) -> None:
        self._view = view
        self.portfolio = portfolio
        self._config = config
        self._marks = marks
        self._orders: list[Order] = []
        self.cancel_requested = False
        self.notes: list[str] = []

    # -------------------------------------------------------------- read side

    @property
    def view(self) -> PanelView:
        return self._view

    @property
    def now(self) -> pd.Timestamp:
        return self._view.now

    @property
    def bar(self) -> int:
        return self._view.index

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._view.symbols

    @property
    def cash(self) -> float:
        return self.portfolio.cash

    @property
    def equity(self) -> float:
        return self.portfolio.equity

    @property
    def orders(self) -> list[Order]:
        """Orders submitted on this bar, in submission order (including rejected ones)."""
        return list(self._orders)

    def position(self, symbol: str) -> float:
        return self.portfolio.position(symbol)

    def weight(self, symbol: str) -> float:
        return self.portfolio.weight(symbol)

    def price(self, symbol: str) -> float:
        """Last known close for ``symbol`` (NaN if it has never traded)."""
        return float(self._marks[self._symbol_index(symbol)])

    # ------------------------------------------------------------- write side

    def order(
        self,
        symbol: str,
        quantity: float,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        max_bars: int | None = None,
    ) -> Order:
        """Submit an order for ``quantity`` units (negative = sell).

        Quantities are rounded toward zero to the lot grid unless fractional
        shares are enabled. Orders that round to zero or would open a short
        when shorting is disabled are recorded as rejected and returned, so
        the blotter shows what the strategy asked for.
        """
        self._symbol_index(symbol)
        rounded = self._config.round_quantity(quantity)
        order = Order(
            symbol=symbol,
            quantity=rounded if rounded != 0 else math.copysign(1.0, quantity),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            max_bars=max_bars,
            created_at=self.now,
            created_bar=self.bar,
        )
        if rounded == 0:
            order.quantity = quantity
            order.status = OrderStatus.REJECTED
            order.reason = f"quantity {quantity} rounds to zero on lot size {self._config.lot_size}"
        elif not self._config.allow_short and self.position(symbol) + rounded < -1e-12:
            order.status = OrderStatus.REJECTED
            order.reason = "short selling disabled"
        self._orders.append(order)
        return order

    def order_target_quantity(self, symbol: str, target: float) -> Order | None:
        delta = target - self.position(symbol)
        if abs(delta) < 1e-12:
            return None
        return self.order(symbol, delta)

    def order_target_weight(self, symbol: str, weight: float) -> Order | None:
        """Rebalance ``symbol`` to ``weight`` of equity, sized at execution.

        NaN weight means "no opinion" and submits nothing. Negative weights are
        clipped to zero (with a note) when shorting is disabled. Symbols with
        no price yet are rejected because a weight cannot be turned into units.
        """
        j = self._symbol_index(symbol)
        if weight is None or math.isnan(weight):
            return None
        if not self._config.allow_short and weight < 0:
            self.notes.append(
                f"{symbol}: target weight {weight:.4f} clipped to 0, shorting disabled"
            )
            weight = 0.0
        order = Order(
            symbol=symbol,
            target_weight=float(weight),
            created_at=self.now,
            created_bar=self.bar,
        )
        if not np.isfinite(self._marks[j]):
            order.status = OrderStatus.REJECTED
            order.reason = "no price yet for symbol"
        self._orders.append(order)
        return order

    def order_target_weights(self, weights: Mapping[str, float] | pd.Series) -> list[Order]:
        """Rebalance every listed symbol. Symbols not listed are left alone."""
        submitted: list[Order] = []
        for symbol, weight in weights.items():
            order = self.order_target_weight(str(symbol), float(weight))
            if order is not None:
                submitted.append(order)
        return submitted

    def cancel_all(self) -> None:
        """Cancel every order still open from earlier bars before new orders are placed."""
        self.cancel_requested = True

    # ---------------------------------------------------------------- helpers

    def _symbol_index(self, symbol: str) -> int:
        try:
            return self._view.symbols.index(symbol)
        except ValueError:
            raise KeyError(
                f"unknown symbol {symbol!r}; panel symbols: {self._view.symbols}"
            ) from None
