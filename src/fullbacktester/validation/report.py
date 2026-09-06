"""Aggregate every check into one report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from fullbacktester.data.panel import Panel
from fullbacktester.data.quality import check_data_quality
from fullbacktester.execution.compare import EngineComparison, compare_engines
from fullbacktester.execution.config import EngineConfig
from fullbacktester.flags import Flag, Severity, score
from fullbacktester.strategy.base import Strategy, WeightStrategy
from fullbacktester.validation.perturbation import FutureLeakTester
from fullbacktester.validation.static_scanner import StaticScanner


@dataclass
class ValidationReport:
    """Flags from every check plus the engine comparison, with a 0-100 confidence score.

    The score is not pass/fail. Read the flags: a clean score means nothing the
    checks know how to find was found.
    """

    flags: list[Flag] = field(default_factory=list)
    comparison: EngineComparison | None = None

    @property
    def score(self) -> float:
        return score(self.flags)

    @property
    def has_high(self) -> bool:
        return any(f.severity is Severity.HIGH for f in self.flags)

    def by_source(self) -> dict[str, list[Flag]]:
        grouped: dict[str, list[Flag]] = defaultdict(list)
        for flag in self.flags:
            grouped[flag.source].append(flag)
        return dict(grouped)

    def summary(self) -> str:
        lines = [f"validation score: {self.score:.0f}/100 ({len(self.flags)} flags)"]
        for source, flags in sorted(self.by_source().items()):
            lines.append(f"  {source}:")
            lines.extend(f"    {flag}" for flag in flags)
        if self.comparison is not None:
            c = self.comparison
            lines.append(
                "  engines: max equity gap "
                f"{c.max_equity_gap:.3%}, Sharpe gap {c.sharpe_gap:+.3f}, "
                f"tracking error {c.tracking_error:.3%}"
            )
        return "\n".join(lines)


def validate(
    strategy: Strategy,
    panel: Panel,
    *,
    config: EngineConfig | None = None,
    static: bool = True,
    perturbation: bool = True,
    data: bool = True,
    engines: bool = True,
    n_samples: int = 6,
    seed: int = 0,
) -> ValidationReport:
    """Run every applicable check on ``strategy`` against ``panel``.

    ``engines`` compares the vectorized and event-driven engines under
    ``config`` for weight strategies (imperative strategies have no vectorized
    twin). ``data`` runs the panel quality checks.
    """
    flags: list[Flag] = []
    comparison: EngineComparison | None = None
    if data:
        flags += check_data_quality(panel)
    if static:
        flags += StaticScanner().scan(strategy)
    if perturbation:
        flags += FutureLeakTester(n_samples=n_samples, seed=seed).run(strategy, panel)
    if engines and isinstance(strategy, WeightStrategy):
        comparison = compare_engines(strategy, panel, config)
        flags += comparison.flags()
        flags += [f for f in comparison.event_driven.flags if f.severity is not Severity.INFO]
    return ValidationReport(flags=flags, comparison=comparison)
