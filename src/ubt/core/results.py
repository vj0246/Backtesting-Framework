"""Shared output shapes. Both engines return a BacktestResult; both leakage
checks contribute LeakageFlags to one ValidationReport. Reporting and the CLI
consume these without caring which engine or which check produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    engine_name: str  # "vectorized" | "event_driven" | "reference"
    cost_bps: float

    def sharpe(self) -> float:
        r = self.equity_curve.pct_change().dropna()
        return float((r.mean() / r.std()) * (252**0.5)) if r.std() else 0.0


@dataclass
class LeakageFlag:
    source: str  # "static" | "perturbation" | "purge_embargo"
    severity: str  # "info" | "warn" | "high"
    message: str
    file: str | None = None
    line: int | None = None


@dataclass
class ValidationReport:
    leakage_flags: list[LeakageFlag] = field(default_factory=list)
    deflated_sharpe: float | None = None
    probability_of_backtest_overfitting: float | None = None
    engine_spread_pct: float | None = None  # max pairwise divergence vs. the reference calc

    @property
    def leakage_score(self) -> float:
        """0-100. Not pass/fail — read the flags for what was actually checked
        and by which method; a clean score means nothing the scanner and the
        perturbation test know how to look for was found, not "no leakage"."""
        penalty = sum({"info": 1, "warn": 5, "high": 20}[f.severity] for f in self.leakage_flags)
        return max(0.0, 100.0 - penalty)
