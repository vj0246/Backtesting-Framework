"""Run several strategies through the same gauntlet and compare them honestly.

Every entry gets the same panel, the same execution assumptions, the same
cost model, the same validation checks, and a Deflated Sharpe Ratio that
accounts for how many entries there are. The number you tried is part of the
statistics, not a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantgauntlet.data.panel import Panel
from quantgauntlet.data.quality import check_data_quality
from quantgauntlet.execution.compare import EngineComparison, compare_results
from quantgauntlet.execution.config import EngineConfig
from quantgauntlet.execution.event_driven import EventDrivenEngine
from quantgauntlet.execution.vectorized import VectorizedEngine
from quantgauntlet.flags import Flag, Severity
from quantgauntlet.metrics.overfitting import (
    CSCVResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quantgauntlet.result import BacktestResult
from quantgauntlet.strategy.base import Strategy, WeightStrategy
from quantgauntlet.validation.perturbation import FutureLeakTester
from quantgauntlet.validation.report import ValidationReport
from quantgauntlet.validation.static_scanner import StaticScanner


@dataclass
class ArenaEntry:
    name: str
    strategy: Strategy
    event_driven: BacktestResult
    vectorized: BacktestResult | None
    comparison: EngineComparison | None
    validation: ValidationReport

    @property
    def result(self) -> BacktestResult:
        """The realistic (event-driven) result; the one to report."""
        return self.event_driven


@dataclass
class ArenaResult:
    entries: dict[str, ArenaEntry]
    data_flags: list[Flag]
    rf_annual: float
    table: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.table = self._build_table()

    @property
    def n_trials(self) -> int:
        return len(self.entries)

    def returns_matrix(self) -> pd.DataFrame:
        return pd.DataFrame({name: e.result.returns for name, e in self.entries.items()})

    def pbo(self, n_blocks: int = 16) -> CSCVResult:
        """Probability of backtest overfitting across the entries (needs >= 2)."""
        return probability_of_backtest_overfitting(self.returns_matrix(), n_blocks=n_blocks)

    def best(self, by: str = "sharpe") -> str:
        ranked = self.table[by].dropna()
        if ranked.empty:
            raise ValueError(f"no finite values for {by!r}")
        return str(ranked.idxmax())

    def summary(self) -> str:
        columns = [
            "cagr",
            "sharpe",
            "deflated_sharpe",
            "max_drawdown",
            "avg_turnover",
            "cost_drag",
            "engine_sharpe_gap",
            "validation_score",
        ]
        lines = [self.table[columns].to_string(float_format=lambda v: f"{v:.3f}")]
        if self.data_flags:
            lines.append("data:")
            lines.extend(f"  {flag}" for flag in self.data_flags)
        for name, entry in self.entries.items():
            high = [f for f in entry.validation.flags if f.severity is Severity.HIGH]
            if high:
                lines.append(f"{name}: {len(high)} HIGH-severity finding(s)")
                lines.extend(f"  {flag}" for flag in high)
        return "\n".join(lines)

    def _build_table(self) -> pd.DataFrame:
        rows = {}
        per_bar_sharpes = {}
        for name, entry in self.entries.items():
            metrics = entry.result.metrics(self.rf_annual)
            r = entry.result.returns
            std = r.std(ddof=1)
            per_bar_sharpes[name] = float(r.mean() / std) if std > 0 else math.nan
            row = metrics.as_dict()
            row["final_equity"] = entry.result.final_equity
            row["engine_sharpe_gap"] = (
                entry.comparison.sharpe_gap if entry.comparison is not None else math.nan
            )
            row["engine_equity_gap"] = (
                entry.comparison.max_equity_gap if entry.comparison is not None else math.nan
            )
            row["validation_score"] = entry.validation.score
            row["high_flags"] = sum(
                1 for f in entry.validation.flags if f.severity is Severity.HIGH
            )
            rows[name] = row
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "strategy"

        n = len(self.entries)
        finite = [s for s in per_bar_sharpes.values() if math.isfinite(s)]
        variance = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
        table["deflated_sharpe"] = [
            deflated_sharpe_ratio(
                per_bar_sharpes[name],
                int(table.loc[name, "n_bars"]),
                n,
                variance,
                float(table.loc[name, "skew"]) if math.isfinite(table.loc[name, "skew"]) else 0.0,
                float(table.loc[name, "excess_kurtosis"])
                if math.isfinite(table.loc[name, "excess_kurtosis"])
                else 0.0,
            )
            for name in table.index
        ]
        table["probabilistic_sharpe"] = [
            probabilistic_sharpe_ratio(
                per_bar_sharpes[name],
                0.0,
                int(table.loc[name, "n_bars"]),
                float(table.loc[name, "skew"]) if math.isfinite(table.loc[name, "skew"]) else 0.0,
                float(table.loc[name, "excess_kurtosis"])
                if math.isfinite(table.loc[name, "excess_kurtosis"])
                else 0.0,
            )
            for name in table.index
        ]
        return table.sort_values("sharpe", ascending=False)


class Arena:
    """Collect strategies, run them all under identical assumptions, rank them.

    Args:
        panel: Shared market data.
        config: Shared execution assumptions (defaults to ``EngineConfig()``).
        validate: Run static, perturbation, and engine-comparison checks per entry.
        perturbation_samples: Bars tested by the future-noise check per entry.
        rf_annual: Risk-free rate for Sharpe and Sortino.
        seed: For the perturbation tester.
    """

    def __init__(
        self,
        panel: Panel,
        config: EngineConfig | None = None,
        *,
        validate: bool = True,
        perturbation_samples: int = 4,
        rf_annual: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.panel = panel
        self.config = config if config is not None else EngineConfig()
        self.validate = validate
        self.perturbation_samples = perturbation_samples
        self.rf_annual = rf_annual
        self.seed = seed
        self._strategies: dict[str, Strategy] = {}

    def add(self, strategy: Strategy, name: str | None = None) -> Arena:
        label = name or strategy.name
        if label in self._strategies:
            raise ValueError(f"duplicate strategy name {label!r}; pass name=")
        self._strategies[label] = strategy
        return self

    def run(self) -> ArenaResult:
        if not self._strategies:
            raise ValueError("no strategies added")
        data_flags = check_data_quality(self.panel)
        event_engine = EventDrivenEngine(self.config)
        vector_engine = VectorizedEngine(self.config)
        scanner = StaticScanner()
        tester = FutureLeakTester(n_samples=self.perturbation_samples, seed=self.seed)

        entries: dict[str, ArenaEntry] = {}
        for name, strategy in self._strategies.items():
            event_result = event_engine.run(strategy, self.panel)
            vector_result = None
            comparison = None
            flags: list[Flag] = [f for f in event_result.flags if f.severity is not Severity.INFO]
            if isinstance(strategy, WeightStrategy):
                vector_result = vector_engine.run(strategy, self.panel)
                comparison = compare_results(vector_result, event_result)
                flags += comparison.flags()
            if self.validate:
                flags += scanner.scan(strategy)
                flags += tester.run(strategy, self.panel)
            entries[name] = ArenaEntry(
                name=name,
                strategy=strategy,
                event_driven=event_result,
                vectorized=vector_result,
                comparison=comparison,
                validation=ValidationReport(flags=flags, comparison=comparison),
            )
        return ArenaResult(entries=entries, data_flags=data_flags, rf_annual=self.rf_annual)
