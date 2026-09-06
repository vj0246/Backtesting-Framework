"""The shared output of every engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from quantgauntlet.execution.config import EngineConfig
from quantgauntlet.flags import Flag
from quantgauntlet.metrics.performance import PerformanceMetrics, compute_metrics, drawdown_series


@dataclass
class BacktestResult:
    """Equity path, positions, and blotter of one strategy on one engine.

    Attributes:
        engine: ``"event_driven"`` or ``"vectorized"``.
        strategy_name: For reports.
        config: The execution assumptions the run used.
        equity: Portfolio value at each bar close.
        cash: Cash at each bar close.
        positions: Units held per symbol at each bar close.
        weights: Position notional / equity at each bar close.
        fills: One row per execution (see ``orders.fills_to_frame``).
        orders: Every order submitted, with final status.
        flags: Findings the engine raised during the run.
        bars_per_year: Annualization factor inherited from the panel.
    """

    engine: str
    strategy_name: str
    config: EngineConfig
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    bars_per_year: float
    flags: list[Flag] = field(default_factory=list)

    @property
    def initial_equity(self) -> float:
        return self.config.initial_cash

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0

    @cached_property
    def returns(self) -> pd.Series:
        """Bar returns; the first bar is measured against initial cash."""
        previous = np.concatenate([[self.initial_equity], self.equity.to_numpy()[:-1]])
        return pd.Series(
            self.equity.to_numpy() / previous - 1.0, index=self.equity.index, name="return"
        )

    @cached_property
    def turnover(self) -> pd.Series:
        """Traded notional per bar as a fraction of that bar's equity."""
        notional = pd.Series(0.0, index=self.equity.index)
        if len(self.fills):
            traded = (
                (self.fills["quantity"].abs() * self.fills["reference_price"])
                .groupby(self.fills["timestamp"])
                .sum()
            )
            notional = notional.add(traded, fill_value=0.0).reindex(self.equity.index).fillna(0.0)
        return (notional / self.equity).rename("turnover")

    @cached_property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1).rename("gross_exposure")

    @property
    def total_costs(self) -> float:
        if not len(self.fills):
            return 0.0
        return float(self.fills["commission"].sum() + self.fills["slippage_cost"].sum())

    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    def metrics(self, rf_annual: float = 0.0) -> PerformanceMetrics:
        return compute_metrics(
            self.returns,
            self.bars_per_year,
            rf_annual=rf_annual,
            turnover=self.turnover,
            gross_exposure=self.gross_exposure,
            n_fills=len(self.fills),
            total_costs=self.total_costs,
            initial_equity=self.initial_equity,
        )

    def summary(self, rf_annual: float = 0.0) -> pd.Series:
        series = self.metrics(rf_annual).to_series()
        series["engine"] = self.engine
        series["strategy"] = self.strategy_name
        return series
