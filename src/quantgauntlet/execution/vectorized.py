"""Weight-based engine: batched signals, ledger-free accounting.

Accepts ``WeightStrategy`` and ``BatchWeightStrategy`` only. Imperative
strategies depend on fills and are refused with a ``TypeError`` rather than
approximated.

Accounting is a scan over bars with NumPy vectors over symbols (see
DECISIONS.md D-007). Each step does exactly what the ledger does: mark to the
execution price, rebalance to target notional, pay costs from cash, mark to
close. Under ``EngineConfig.idealized()`` this engine and the event-driven one
agree to floating-point precision, which the test suite enforces.

Positions are continuous (no lot rounding), shorting is always allowed, and
leverage and participation caps are ignored: this is the idealized twin. The
gap to the event-driven result under realistic settings is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantgauntlet.data.panel import Panel
from quantgauntlet.execution.config import EngineConfig, FillTiming
from quantgauntlet.execution.orders import Fill, fills_to_frame
from quantgauntlet.flags import Flag, Severity
from quantgauntlet.result import BacktestResult
from quantgauntlet.strategy.base import BatchWeightStrategy, Strategy, WeightStrategy


class VectorizedEngine:
    name = "vectorized"

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config if config is not None else EngineConfig()

    def run(self, strategy: Strategy, panel: Panel) -> BacktestResult:
        if not isinstance(strategy, WeightStrategy):
            raise TypeError(
                f"{type(strategy).__name__} is an imperative Strategy; the vectorized "
                "engine runs WeightStrategy / BatchWeightStrategy only. Use EventDrivenEngine."
            )
        config = self.config
        strategy.reset()
        targets = self.collect_targets(strategy, panel)
        flags: list[Flag] = []
        if not config.is_idealized:
            flags.append(
                Flag(
                    source="engine",
                    severity=Severity.INFO,
                    message=(
                        "vectorized engine ignores lot sizes, short restrictions, leverage "
                        "and participation caps; compare with EventDrivenEngine for realism"
                    ),
                )
            )
        if config.fill_timing is FillTiming.SAME_CLOSE:
            flags.append(
                Flag(
                    source="engine",
                    severity=Severity.WARN,
                    message=(
                        "fill_timing=SAME_CLOSE executes at the close the strategy has already "
                        "observed; results are optimistic"
                    ),
                )
            )
        equity, cash, positions, fills = _scan(targets, panel, config)

        index = panel.timestamps
        symbols = list(panel.symbols)
        equity_series = pd.Series(equity, index=index, name="equity")
        positions_frame = pd.DataFrame(positions, index=index, columns=symbols)
        close_ffill = pd.DataFrame(panel.close, index=index, columns=symbols).ffill()
        weights = (positions_frame * close_ffill).div(equity_series, axis=0).fillna(0.0)

        return BacktestResult(
            engine=self.name,
            strategy_name=strategy.name,
            config=config,
            equity=equity_series,
            cash=pd.Series(cash, index=index, name="cash"),
            positions=positions_frame,
            weights=weights,
            fills=fills_to_frame(fills),
            orders=pd.DataFrame(),
            bars_per_year=panel.bars_per_year,
            flags=flags,
        )

    @staticmethod
    def collect_targets(strategy: WeightStrategy, panel: Panel) -> np.ndarray:
        """Target weights decided at each bar's close, shape (bars, symbols); NaN = no opinion."""
        n_bars, n_symbols = len(panel), panel.n_symbols
        targets = np.full((n_bars, n_symbols), np.nan)
        columns = list(panel.symbols)
        if isinstance(strategy, BatchWeightStrategy):
            frame = strategy.target_weights_batch(panel.view(n_bars))
            aligned = frame.reindex(index=panel.timestamps, columns=columns)
            targets[:] = aligned.to_numpy(dtype=np.float64)
            if strategy.warmup > 0:
                targets[: strategy.warmup] = np.nan
            return targets
        for i in range(strategy.warmup, n_bars):
            weights = strategy.target_weights(panel.view(i + 1))
            series = pd.Series(weights, dtype="float64")
            unknown = [s for s in series.index if s not in panel.symbols]
            if unknown:
                raise KeyError(f"weights for symbols not in panel: {unknown}")
            targets[i] = series.reindex(columns).to_numpy(dtype=np.float64)
        return targets


def _scan(
    targets: np.ndarray, panel: Panel, config: EngineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Fill]]:
    n_bars, n_symbols = targets.shape
    cost_model = config.cost_model
    at_close = config.fill_timing is FillTiming.SAME_CLOSE

    cash = config.initial_cash
    units = np.zeros(n_symbols)
    marks = np.full(n_symbols, np.nan)
    equity_out = np.empty(n_bars)
    cash_out = np.empty(n_bars)
    units_out = np.empty((n_bars, n_symbols))
    fills: list[Fill] = []

    def rebalance(target: np.ndarray, bar: int, exec_price: np.ndarray) -> None:
        nonlocal cash, units
        exec_marks = np.where(np.isfinite(exec_price), exec_price, marks)
        held = units != 0
        equity_ref = cash + float(np.sum(np.where(held, units * exec_marks, 0.0)))
        tradeable = np.isfinite(exec_price) & np.isfinite(target)
        desired = np.where(
            tradeable, target * equity_ref / np.where(tradeable, exec_price, 1.0), units
        )
        delta = desired - units
        delta[np.abs(delta) < 1e-12] = 0.0
        active = delta != 0
        if not active.any():
            return
        volume = panel.volume[bar]
        participation = cost_model.participation(delta, volume)
        price = cost_model.fill_price(delta, exec_price, participation)
        commission = cost_model.commission.commission(delta, exec_price)
        slippage = cost_model.slippage_cost(delta, exec_price, participation)
        cash -= float(np.sum(np.where(active, delta * price + commission, 0.0)))
        units = np.where(active, desired, units)
        timestamp = panel.timestamps[bar]
        for j in np.flatnonzero(active):
            fills.append(
                Fill(
                    order_id=0,
                    symbol=panel.symbols[j],
                    timestamp=timestamp,
                    bar=bar,
                    quantity=float(delta[j]),
                    price=float(price[j]),
                    reference_price=float(exec_price[j]),
                    commission=float(commission[j]),
                    slippage_cost=float(slippage[j]),
                    participation=float(participation[j]),
                )
            )

    for i in range(n_bars):
        if not at_close and i > 0 and np.isfinite(targets[i - 1]).any():
            rebalance(targets[i - 1], i, panel.open[i])
        close = panel.close[i]
        marks = np.where(np.isfinite(close), close, marks)
        if at_close and np.isfinite(targets[i]).any():
            rebalance(targets[i], i, close)
        held = units != 0
        equity_out[i] = cash + float(np.sum(np.where(held, units * marks, 0.0)))
        cash_out[i] = cash
        units_out[i] = units

    return equity_out, cash_out, units_out, fills
