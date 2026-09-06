"""Bar-by-bar engine with explicit orders, fills, and a cash ledger.

Per bar ``i``:

1. Fill orders left open from earlier bars against bar ``i`` (next-open timing).
2. Update last-known closes and mark the portfolio at bar ``i``'s close.
3. Hand the strategy a ``BarContext`` whose view ends at bar ``i``.
4. Queue the strategy's orders for bar ``i + 1`` (or fill market orders now
   under ``SAME_CLOSE`` timing and re-mark).

Orders queued on the final bar can never fill and are reported as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fullbacktester.data.panel import Panel
from fullbacktester.execution.config import EngineConfig, FillTiming
from fullbacktester.execution.fills import FillModel
from fullbacktester.execution.ledger import Ledger
from fullbacktester.execution.orders import (
    Order,
    OrderStatus,
    OrderType,
    fills_to_frame,
    orders_to_frame,
)
from fullbacktester.flags import Flag, Severity
from fullbacktester.result import BacktestResult
from fullbacktester.strategy.base import Strategy
from fullbacktester.strategy.context import BarContext


class EventDrivenEngine:
    name = "event_driven"

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config if config is not None else EngineConfig()

    def run(self, strategy: Strategy, panel: Panel) -> BacktestResult:
        config = self.config
        n_bars, n_symbols = len(panel), panel.n_symbols
        ledger = Ledger(panel.symbols, config.initial_cash)
        fill_model = FillModel(config)
        strategy.reset()

        equity = np.empty(n_bars)
        cash = np.empty(n_bars)
        positions = np.empty((n_bars, n_symbols))
        marks = np.full(n_symbols, np.nan)
        pending: list[Order] = []
        all_orders: list[Order] = []
        notes: set[str] = set()
        flags: list[Flag] = []

        if config.fill_timing is FillTiming.SAME_CLOSE:
            flags.append(
                Flag(
                    source="engine",
                    severity=Severity.WARN,
                    message=(
                        "fill_timing=SAME_CLOSE executes market orders at the close the "
                        "strategy has already observed; results are optimistic unless the "
                        "strategy genuinely trades the closing auction"
                    ),
                )
            )

        for i in range(n_bars):
            if pending:
                _, pending = fill_model.fill_bar(pending, panel, i, ledger, marks)

            bar_close = panel.close[i]
            marks = np.where(np.isfinite(bar_close), bar_close, marks)

            if i >= strategy.warmup:
                snapshot = ledger.snapshot(panel.timestamps[i], i, marks)
                ctx = BarContext(panel.view(i + 1), snapshot, config, marks)
                strategy.on_bar(ctx)
                notes.update(ctx.notes)

                if ctx.cancel_requested:
                    for order in pending:
                        order.status = OrderStatus.CANCELLED
                        order.reason = "cancelled by strategy"
                    pending = []

                new_orders = ctx.orders
                all_orders.extend(new_orders)
                live = [o for o in new_orders if o.is_open]
                if config.fill_timing is FillTiming.SAME_CLOSE:
                    now = [o for o in live if o.order_type is OrderType.MARKET]
                    later = [o for o in live if o.order_type is not OrderType.MARKET]
                    if now:
                        _, still_open = fill_model.fill_bar(
                            now, panel, i, ledger, marks, at_close=True
                        )
                        later.extend(still_open)
                    pending.extend(later)
                else:
                    pending.extend(live)

            equity[i] = ledger.equity(marks)
            cash[i] = ledger.cash
            positions[i] = ledger.positions

        if pending:
            for order in pending:
                order.status = OrderStatus.CANCELLED
                order.reason = "panel ended before fill"
            flags.append(
                Flag(
                    source="engine",
                    severity=Severity.INFO,
                    message=f"{len(pending)} order(s) placed on the final bar never filled",
                )
            )
        rejected = [o for o in all_orders if o.status is OrderStatus.REJECTED]
        if rejected:
            reasons = sorted({o.reason or "rejected" for o in rejected})
            flags.append(
                Flag(
                    source="engine",
                    severity=Severity.INFO,
                    message=f"{len(rejected)} order(s) rejected: {'; '.join(reasons)}",
                )
            )
        for note in sorted(notes):
            flags.append(Flag(source="engine", severity=Severity.INFO, message=note))

        index = panel.timestamps
        symbols = list(panel.symbols)
        equity_series = pd.Series(equity, index=index, name="equity")
        positions_frame = pd.DataFrame(positions, index=index, columns=symbols)
        close_ffill = pd.DataFrame(panel.close, index=index, columns=symbols).ffill()
        weights = positions_frame * close_ffill
        weights = weights.div(equity_series, axis=0).fillna(0.0)

        return BacktestResult(
            engine=self.name,
            strategy_name=strategy.name,
            config=config,
            equity=equity_series,
            cash=pd.Series(cash, index=index, name="cash"),
            positions=positions_frame,
            weights=weights,
            fills=fills_to_frame(ledger.fills),
            orders=orders_to_frame(all_orders),
            bars_per_year=panel.bars_per_year,
            flags=flags,
        )
