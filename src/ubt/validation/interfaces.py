"""Both leakage-detection methods implement LeakageCheck, so the report
aggregator never needs to know which one produced a given flag — and the
docstrings below are the actual reason to build both together rather than
one now, one later: each one is blind to what the other catches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ubt.core.interfaces import DataFeed, Strategy
from ubt.core.results import LeakageFlag


class LeakageCheck(ABC):
    @abstractmethod
    def run(self, strategy: Strategy, feed: DataFeed) -> list[LeakageFlag]: ...


class PerturbationTester(LeakageCheck):
    """Reruns the strategy on truncated data windows and diffs signals at
    fixed timestamps — generalizes freqtrade's lookahead-analysis beyond
    its own strategy convention, onto anything built on Strategy.on_bar.

    Behavioral, not structural: catches leakage the static scanner has no
    pattern for, at the cost of being provably incomplete too — a strategy
    whose leakage doesn't happen to change behavior under truncation will
    pass clean here, the documented freqtrade false-negative case.
    """

    def run(self, strategy: Strategy, feed: DataFeed) -> list[LeakageFlag]:
        raise NotImplementedError

    # candidate v0 approach: rerun on_bar() at each timestamp with feed
    # windows truncated to N, 2N, 4N bars of lookback; flag any Signal
    # whose target_weight changes at a fixed timestamp as lookback grows
