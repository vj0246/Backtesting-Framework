"""Core contracts shared by both execution engines and all three strategy shapes.

VectorizedEngine and EventDrivenEngine are both written against ExecutionEngine.
RuleBasedStrategy, EventDrivenStrategy (subclass Strategy directly), and MLStrategy
are all written against Strategy. Neither engine needs to know which strategy shape
it got, and nothing needs to know which engine ran it. That's what makes building
both engines together tractable instead of two codebases that quietly diverge.

DataFeed.as_of() is the single chokepoint for time. Both engines and the
perturbation tester call it instead of raw indexing — that's what makes
look-ahead structurally hard to introduce by accident, not just checked for
after the fact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

from ubt.core.results import BacktestResult


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"


@runtime_checkable
class DataFeed(Protocol):
    """Point-in-time market data access — one contract for equities and crypto.

    Fundamentals/corporate-action adapters implement as_of with a
    restatement-aware lookup; plain OHLCV adapters implement it with a slice.
    Either way: rows with timestamp > `timestamp` must never be reachable
    through this object.
    """

    asset_class: AssetClass

    def as_of(self, timestamp: datetime) -> pd.DataFrame:
        """Return all rows with index <= timestamp. Nothing later, ever."""
        ...

    def symbols(self) -> list[str]: ...

    def timestamps(self) -> list[datetime]:
        """Every bar the feed has, in order. Added when implementing the
        engines — as_of()/symbols() alone gave no way to enumerate bars."""
        ...


@dataclass(frozen=True)
class Signal:
    """Normalized output of any strategy adapter, at one timestamp.
    Both engines consume this — never raw broker-style orders."""

    timestamp: datetime
    symbol: str
    target_weight: float  # -1..1


class Strategy(ABC):
    """Common base for all three adapter shapes below."""

    @abstractmethod
    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> list[Signal]:
        """Called once per bar by both engines — batched by the vectorized
        engine, one timestamp at a time by the event-driven engine. The
        strategy can't tell which one is calling it, which is exactly what
        keeps the two engines' outputs comparable for the implementation-risk
        cross-check."""
        ...

    def scan_targets(self) -> list:
        """Callables the static leakage scanner should read. Defaults to
        on_bar itself — right for a hand-written subclass, wrong for the
        two adapters below, since their on_bar is plumbing, not user logic.
        Added when building StaticScanner; found the gap building it."""
        return [self.on_bar]


class RuleBasedStrategy(Strategy):
    """Adapter for a plain vectorized signal function: dataframe -> weights."""

    def __init__(self, signal_fn) -> None:
        self._signal_fn = signal_fn

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> list[Signal]:
        weights = self._signal_fn(data)  # expects a symbol -> weight mapping
        return [
            Signal(timestamp, symbol, float(w))
            for symbol, w in weights.items()
            if w != 0
        ]

    def scan_targets(self) -> list:
        return [self._signal_fn]


class MLStrategy(Strategy):
    """Adapter for an sklearn-style estimator (.fit / .predict).

    purge_embargo_bars is read by the CV helper during training, not by
    on_bar — keeping it here means "how much of a gap did you leave" travels
    with the strategy object instead of living in someone's memory.
    """

    def __init__(self, model, feature_fn, purge_embargo_bars: int = 0) -> None:
        self._model = model
        self._feature_fn = feature_fn
        self.purge_embargo_bars = purge_embargo_bars

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> list[Signal]:
        features = self._feature_fn(data)
        preds = self._model.predict(features)  # expects symbol -> predicted weight
        return [
            Signal(timestamp, symbol, float(w))
            for symbol, w in preds.items()
            if w != 0
        ]

    def scan_targets(self) -> list:
        return [self._feature_fn]


class ExecutionEngine(ABC):
    """VectorizedEngine and EventDrivenEngine both implement this. Same
    inputs, same BacktestResult shape — required for the zero-cost
    cross-engine sanity check that feeds the implementation-risk score."""

    @abstractmethod
    def run(self, strategy: Strategy, feed: DataFeed, cost_bps: float) -> BacktestResult: ...
