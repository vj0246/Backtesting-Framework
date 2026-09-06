"""Cross-engine comparison: the implementation-risk diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fullbacktester.data.panel import Panel
from fullbacktester.execution.config import EngineConfig
from fullbacktester.execution.event_driven import EventDrivenEngine
from fullbacktester.execution.vectorized import VectorizedEngine
from fullbacktester.flags import Flag, Severity
from fullbacktester.result import BacktestResult
from fullbacktester.strategy.base import WeightStrategy


@dataclass(frozen=True)
class EngineComparison:
    """How far the idealized (vectorized) and realistic (event-driven) results sit apart.

    Attributes:
        max_equity_gap: max over bars of |E_vec - E_evt| / E_evt.
        final_equity_gap: (E_vec - E_evt) / E_evt at the last bar.
        sharpe_gap: annualized Sharpe(vectorized) - Sharpe(event-driven).
        tracking_error: annualized std of the per-bar return difference.
        max_weight_gap: max over bars and symbols of |w_vec - w_evt|.
    """

    vectorized: BacktestResult
    event_driven: BacktestResult
    max_equity_gap: float
    final_equity_gap: float
    sharpe_gap: float
    tracking_error: float
    max_weight_gap: float

    def agrees(self, tolerance: float = 1e-9) -> bool:
        return self.max_equity_gap <= tolerance and self.max_weight_gap <= tolerance

    def flags(self, *, equity_gap_warn: float = 0.01, sharpe_gap_warn: float = 0.25) -> list[Flag]:
        """Findings when the engines disagree materially under the same configuration."""
        out: list[Flag] = []
        if self.max_equity_gap > equity_gap_warn:
            out.append(
                Flag(
                    source="engine_comparison",
                    severity=Severity.WARN,
                    message=(
                        "idealized and realistic equity paths differ by up to "
                        f"{self.max_equity_gap:.2%}; the strategy is sensitive to "
                        "execution assumptions"
                    ),
                )
            )
        if math.isfinite(self.sharpe_gap) and abs(self.sharpe_gap) > sharpe_gap_warn:
            out.append(
                Flag(
                    source="engine_comparison",
                    severity=Severity.WARN,
                    message=(
                        f"vectorized Sharpe exceeds event-driven Sharpe by {self.sharpe_gap:+.2f}; "
                        "the idealized number overstates what execution constraints allow"
                    ),
                )
            )
        return out


def compare_results(vectorized: BacktestResult, event_driven: BacktestResult) -> EngineComparison:
    e_vec = vectorized.equity.to_numpy()
    e_evt = event_driven.equity.to_numpy()
    if e_vec.shape != e_evt.shape:
        raise ValueError("results cover different numbers of bars")
    rel = np.abs(e_vec - e_evt) / e_evt
    diff = vectorized.returns.to_numpy() - event_driven.returns.to_numpy()
    tracking = (
        float(np.std(diff, ddof=1) * math.sqrt(vectorized.bars_per_year))
        if len(diff) > 1
        else math.nan
    )
    weights_gap = float(
        np.nanmax(np.abs(vectorized.weights.to_numpy() - event_driven.weights.to_numpy()))
    )
    sharpe_vec = vectorized.metrics().sharpe
    sharpe_evt = event_driven.metrics().sharpe
    return EngineComparison(
        vectorized=vectorized,
        event_driven=event_driven,
        max_equity_gap=float(rel.max()),
        final_equity_gap=float((e_vec[-1] - e_evt[-1]) / e_evt[-1]),
        sharpe_gap=sharpe_vec - sharpe_evt,
        tracking_error=tracking,
        max_weight_gap=weights_gap,
    )


def compare_engines(
    strategy: WeightStrategy, panel: Panel, config: EngineConfig | None = None
) -> EngineComparison:
    """Run both engines on the same strategy, panel, and config."""
    config = config if config is not None else EngineConfig()
    vectorized = VectorizedEngine(config).run(strategy, panel)
    event_driven = EventDrivenEngine(config).run(strategy, panel)
    return compare_results(vectorized, event_driven)
