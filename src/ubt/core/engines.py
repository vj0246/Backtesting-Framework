"""Concrete ExecutionEngine implementations.

Both engines call strategy.on_bar() once per timestamp — that's what keeps
their outputs comparable for compare_engines() below. "Vectorized" here
means the equity-curve/turnover/cost bookkeeping is done with pandas ops
across the whole series at once, not that strategy evaluation itself is
batched — Strategy.on_bar is inherently a per-bar call, by design, so one
Strategy object works in both engines. A genuinely batched fast path
(signal_fn(full_df) -> full_weights_df in one call, VectorBT-style) would
beat this on speed but needs a second strategy-calling convention; not
built here.

Honest state: VectorizedEngine and EventDrivenEngine currently share the
exact same computation below. EventDrivenEngine is the shape you'd extend
with real per-bar state (positions, cash, pending orders, partial fills)
for genuinely path-dependent strategies — it doesn't have any of that yet,
so right now it will always agree with VectorizedEngine by construction.
compare_engines() is real and wired up; it just has nothing to disagree
about until EventDrivenEngine gets actual state.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ubt.core.interfaces import DataFeed, ExecutionEngine, Strategy
from ubt.core.results import BacktestResult


def _collect_weights(strategy: Strategy, feed: DataFeed) -> pd.DataFrame:
    timestamps = feed.timestamps()
    rows: dict[datetime, dict[str, float]] = {}
    for ts in timestamps:
        data = feed.as_of(ts)
        rows[ts] = {sig.symbol: sig.target_weight for sig in strategy.on_bar(ts, data)}
    weights = pd.DataFrame(rows).T.reindex(index=timestamps, columns=feed.symbols())
    return weights.fillna(0.0)


def _prices(feed: DataFeed) -> pd.DataFrame:
    timestamps = feed.timestamps()
    return feed.as_of(timestamps[-1])[feed.symbols()]


def _equity_from_weights(
    weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float
) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().reindex(weights.index).fillna(0.0)
    held = weights.shift(1).fillna(0.0)  # yesterday's target weight earns today's return
    gross_pnl = (held * returns).sum(axis=1)

    turnover = weights.diff()
    turnover.iloc[0] = weights.iloc[0]
    costs = turnover.abs().sum(axis=1) * (cost_bps / 10_000)

    equity = (1 + gross_pnl - costs).cumprod()

    stacked = turnover.stack()  # no NaNs here, so nothing gets lost
    changed = stacked[stacked != 0].rename("weight_change").reset_index()
    changed.columns = ["timestamp", "symbol", "weight_change"]
    return equity, changed


class VectorizedEngine(ExecutionEngine):
    """Collects signals with one loop over timestamps (required by the
    Strategy contract), then does all PnL/cost/equity math as pandas
    vector ops in a single pass."""

    def run(self, strategy: Strategy, feed: DataFeed, cost_bps: float) -> BacktestResult:
        weights = _collect_weights(strategy, feed)
        equity, trades = _equity_from_weights(weights, _prices(feed), cost_bps)
        return BacktestResult(equity_curve=equity, trades=trades, engine_name="vectorized", cost_bps=cost_bps)


class EventDrivenEngine(ExecutionEngine):
    """Placeholder for real bar-by-bar state simulation — see module
    docstring. Computes identically to VectorizedEngine today."""

    def run(self, strategy: Strategy, feed: DataFeed, cost_bps: float) -> BacktestResult:
        weights = _collect_weights(strategy, feed)
        equity, trades = _equity_from_weights(weights, _prices(feed), cost_bps)
        return BacktestResult(equity_curve=equity, trades=trades, engine_name="event_driven", cost_bps=cost_bps)


def compare_engines(strategy: Strategy, feed: DataFeed, cost_bps: float) -> float:
    """Runs both engines on the same strategy/feed/cost, returns the
    absolute Sharpe-point gap between them — feeds
    ValidationReport.engine_spread_pct (naming is aspirational: this isn't
    a percentage yet, that needs a defined baseline once the engines can
    actually diverge). Should read ~0.0 today; if it doesn't, something in
    this file has a bug, not the strategy.
    """
    a = VectorizedEngine().run(strategy, feed, cost_bps)
    b = EventDrivenEngine().run(strategy, feed, cost_bps)
    return abs(a.sharpe() - b.sharpe())
