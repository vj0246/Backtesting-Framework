"""Diagnostic flags shared by data quality checks, leakage checks, and engines.

A flag is a finding, not a verdict. Every producer documents what its flags do
and do not cover; a report with zero flags means nothing the producers know how
to look for was found.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    HIGH = "high"

    @property
    def penalty(self) -> int:
        return _PENALTY[self]


_PENALTY: dict[Severity, int] = {Severity.INFO: 1, Severity.WARN: 5, Severity.HIGH: 20}


@dataclass(frozen=True)
class Flag:
    """One finding.

    Attributes:
        source: Producer identifier, e.g. ``"static"``, ``"perturbation"``,
            ``"data"``, ``"engine"``.
        severity: How much the finding should lower confidence in a result.
        message: Human-readable description with enough detail to act on.
        file: Source file the finding points at, when it concerns user code.
        line: 1-based line number in ``file``.
        symbol: Instrument the finding concerns, when it concerns data.
    """

    source: str
    severity: Severity
    message: str
    file: str | None = None
    line: int | None = None
    symbol: str | None = None

    def __str__(self) -> str:
        location = ""
        if self.file is not None:
            location = f" ({self.file}:{self.line})" if self.line is not None else f" ({self.file})"
        elif self.symbol is not None:
            location = f" [{self.symbol}]"
        return f"[{self.severity.value}] {self.source}: {self.message}{location}"


def score(flags: list[Flag]) -> float:
    """0-100 confidence score. 100 means no flags; each flag subtracts its severity penalty."""
    penalty = sum(flag.severity.penalty for flag in flags)
    return max(0.0, 100.0 - penalty)
