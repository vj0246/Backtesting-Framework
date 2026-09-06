"""Cash and position accounting for the event-driven engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fullbacktester.execution.orders import Fill


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Read-only portfolio state handed to strategies through ``BarContext``.

    ``prices`` are the marks used to compute ``equity``: the last known close of
    each symbol at the snapshot time (NaN for symbols that have never traded).
    """

    timestamp: pd.Timestamp
    bar: int
    cash: float
    equity: float
    positions: Mapping[str, float]
    prices: Mapping[str, float]

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def price(self, symbol: str) -> float:
        return self.prices.get(symbol, float("nan"))

    def notional(self, symbol: str) -> float:
        units = self.position(symbol)
        return 0.0 if units == 0 else units * self.price(symbol)

    def weight(self, symbol: str) -> float:
        if self.equity <= 0:
            return float("nan")
        return self.notional(symbol) / self.equity

    @property
    def weights(self) -> dict[str, float]:
        return {s: self.weight(s) for s in self.positions}

    @property
    def gross_exposure(self) -> float:
        if self.equity <= 0:
            return float("nan")
        return sum(abs(self.notional(s)) for s in self.positions) / self.equity

    @property
    def net_exposure(self) -> float:
        if self.equity <= 0:
            return float("nan")
        return sum(self.notional(s) for s in self.positions) / self.equity


class Ledger:
    """Positions in units per symbol plus cash. Fills are the only way state changes."""

    def __init__(self, symbols: Sequence[str], initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.symbols: tuple[str, ...] = tuple(symbols)
        self._index: dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions = np.zeros(len(self.symbols), dtype=np.float64)
        self.fills: list[Fill] = []

    def index_of(self, symbol: str) -> int:
        try:
            return self._index[symbol]
        except KeyError:
            raise KeyError(f"unknown symbol {symbol!r}") from None

    def position(self, symbol: str) -> float:
        return float(self.positions[self.index_of(symbol)])

    def apply_fill(self, fill: Fill) -> None:
        """Cash pays ``quantity * price`` (price includes slippage) plus commission."""
        j = self.index_of(fill.symbol)
        self.positions[j] += fill.quantity
        if abs(self.positions[j]) < 1e-12:
            self.positions[j] = 0.0
        self.cash -= fill.quantity * fill.price + fill.commission
        self.fills.append(fill)

    def equity(self, marks: np.ndarray) -> float:
        """Cash plus positions marked at ``marks`` (NaN allowed only for flat symbols)."""
        held = self.positions != 0
        if np.any(held & ~np.isfinite(marks)):
            bad = [self.symbols[j] for j in np.flatnonzero(held & ~np.isfinite(marks))]
            raise ValueError(f"position in {bad} but no price to mark it")
        return self.cash + float(np.sum(np.where(held, self.positions * marks, 0.0)))

    def gross_notional(self, marks: np.ndarray) -> float:
        held = self.positions != 0
        return float(np.sum(np.where(held, np.abs(self.positions * marks), 0.0)))

    def snapshot(self, timestamp: pd.Timestamp, bar: int, marks: np.ndarray) -> PortfolioSnapshot:
        positions = {
            s: float(self.positions[j]) for s, j in self._index.items() if self.positions[j] != 0
        }
        prices = {s: float(marks[j]) for s, j in self._index.items() if np.isfinite(marks[j])}
        return PortfolioSnapshot(
            timestamp=timestamp,
            bar=bar,
            cash=self.cash,
            equity=self.equity(marks),
            positions=positions,
            prices=prices,
        )
