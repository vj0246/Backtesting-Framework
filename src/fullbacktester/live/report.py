"""Comparing a live paper record against what a backtest claims it should be.

Two different questions, and it matters which one you are asking.

**Replay divergence.** Re-run the strategy through the backtester over exactly
the bars the session recorded, and compare with what the session actually did.
These should agree almost exactly, because the paper session applies the same
fill model and cost model. When they do not, the live loop and the backtest have
drifted apart, or the session skipped bars, or the strategy is not deterministic.
This is a correctness check on the machinery.

**Expectation gap.** Compare the live record against a backtest run over an
earlier period, the one that convinced you to trade the strategy. These are not
expected to match, because the market moved on. A large negative gap is the
signal worth having: the edge you measured is not the edge you are getting.

Only the first is a bug. The second is the entire reason to paper trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fullbacktester.data.panel import Panel
from fullbacktester.execution.config import EngineConfig
from fullbacktester.execution.event_driven import EventDrivenEngine
from fullbacktester.flags import Flag, Severity
from fullbacktester.metrics.performance import PerformanceMetrics
from fullbacktester.result import BacktestResult
from fullbacktester.strategy.base import Strategy


@dataclass(frozen=True)
class ReplayCheck:
    """Live record versus a backtest over the identical recorded bars."""

    live: PerformanceMetrics
    replayed: PerformanceMetrics
    max_equity_gap: float
    final_equity_gap: float
    n_bars: int

    def agrees(self, tolerance: float = 1e-6) -> bool:
        return self.max_equity_gap <= tolerance

    def flags(self, tolerance: float = 1e-6) -> list[Flag]:
        if self.agrees(tolerance):
            return [
                Flag(
                    source="live_replay",
                    severity=Severity.INFO,
                    message=(
                        f"live record reproduces exactly when replayed through the backtester "
                        f"over the same {self.n_bars} bars"
                    ),
                )
            ]
        return [
            Flag(
                source="live_replay",
                severity=Severity.HIGH,
                message=(
                    f"replaying the recorded bars through the backtester gives a different "
                    f"equity path (max gap {self.max_equity_gap:.3%}). The live loop and the "
                    "backtest disagree, or bars were skipped; the paper record is not "
                    "reproducible until this is explained"
                ),
            )
        ]


@dataclass(frozen=True)
class ExpectationGap:
    """Live performance versus the backtest that justified trading the strategy."""

    live: PerformanceMetrics
    expected: PerformanceMetrics
    sharpe_gap: float
    cagr_gap: float
    live_bars: int
    expected_bars: int

    def flags(self, sharpe_tolerance: float = 0.5) -> list[Flag]:
        out: list[Flag] = []
        if self.live_bars < 60:
            out.append(
                Flag(
                    source="live_expectation",
                    severity=Severity.INFO,
                    message=(
                        f"only {self.live_bars} live bars so far; the gap below is not yet "
                        "distinguishable from noise"
                    ),
                )
            )
        if math.isfinite(self.sharpe_gap) and self.sharpe_gap < -sharpe_tolerance:
            out.append(
                Flag(
                    source="live_expectation",
                    severity=Severity.WARN,
                    message=(
                        f"live Sharpe is {abs(self.sharpe_gap):.2f} below the backtest "
                        f"({self.live.sharpe:.2f} vs {self.expected.sharpe:.2f}). Either the "
                        "backtest was optimistic or the regime changed; both are reasons to "
                        "stop adding capital, not to retune"
                    ),
                )
            )
        return out


def replay_check(
    live_result: BacktestResult,
    strategy: Strategy,
    panel: Panel,
    config: EngineConfig,
) -> ReplayCheck:
    """Re-run ``strategy`` over the session's own recorded bars and compare.

    ``panel`` must be the bars the session traded on, as first seen. Only the
    overlapping timestamps are compared, since the session begins recording
    equity from its first processed bar rather than from the panel's start.
    """
    replayed = EventDrivenEngine(config).run(strategy, panel)
    shared = live_result.equity.index.intersection(replayed.equity.index)
    if len(shared) < 2:
        raise ValueError("live record and replay share fewer than two bars")
    live_equity = live_result.equity.loc[shared].to_numpy()
    replay_equity = replayed.equity.loc[shared].to_numpy()
    # Both start from the same capital, so compare the paths directly.
    gaps = np.abs(live_equity - replay_equity) / np.abs(replay_equity)
    return ReplayCheck(
        live=live_result.metrics(),
        replayed=replayed.metrics(),
        max_equity_gap=float(np.nanmax(gaps)),
        final_equity_gap=float((live_equity[-1] - replay_equity[-1]) / replay_equity[-1]),
        n_bars=len(shared),
    )


def expectation_gap(live_result: BacktestResult, expected: BacktestResult) -> ExpectationGap:
    """Compare a live record against the backtest it was expected to reproduce."""
    live_metrics = live_result.metrics()
    expected_metrics = expected.metrics()
    return ExpectationGap(
        live=live_metrics,
        expected=expected_metrics,
        sharpe_gap=live_metrics.sharpe - expected_metrics.sharpe,
        cagr_gap=live_metrics.cagr - expected_metrics.cagr,
        live_bars=live_metrics.n_bars,
        expected_bars=expected_metrics.n_bars,
    )


def summary_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """One row per strategy in the session, ranked by Sharpe."""
    rows = {}
    for label, result in results.items():
        metrics = result.metrics()
        row = metrics.as_dict()
        row["final_equity"] = result.final_equity
        row["total_return"] = result.total_return
        rows[label] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    return table.sort_values("sharpe", ascending=False)
