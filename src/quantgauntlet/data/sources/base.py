"""Data source contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import ClassVar

import pandas as pd

from quantgauntlet.flags import Flag, Severity
from quantgauntlet.markets import Frequency, Market


class DataSource(ABC):
    """A place bars come from.

    Class attributes declare what the source can do and what it cannot be
    trusted for; ``caveats`` turns the latter into flags so every report
    carries the data's limitations next to the strategy's numbers.
    """

    name: ClassVar[str]
    markets: ClassVar[frozenset[str]]
    frequencies: ClassVar[frozenset[Frequency]]
    adjusted: ClassVar[bool | None]
    survivorship_free: ClassVar[bool]
    point_in_time: ClassVar[bool] = False

    @abstractmethod
    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        """Return canonical bars for ``symbols`` with closes in ``[start, end]`` (inclusive dates).

        Must return a frame with the columns in ``schema.BAR_COLUMNS``; the
        loader validates it. Symbols with no data are simply absent.
        """

    def supports(self, market: Market, frequency: Frequency) -> bool:
        return market.code in self.markets and frequency in self.frequencies

    def caveats(self) -> list[Flag]:
        flags: list[Flag] = []
        if not self.survivorship_free:
            flags.append(
                Flag(
                    source="data",
                    severity=Severity.WARN,
                    message=(
                        f"{self.name}: universe is whatever exists today; delisted and failed "
                        "names are absent (survivorship bias). Cross-sectional results are "
                        "optimistic"
                    ),
                )
            )
        if self.adjusted is False:
            flags.append(
                Flag(
                    source="data",
                    severity=Severity.WARN,
                    message=(
                        f"{self.name}: prices are unadjusted for splits, bonuses, and dividends; "
                        "returns across corporate actions are wrong"
                    ),
                )
            )
        elif self.adjusted is None:
            flags.append(
                Flag(
                    source="data",
                    severity=Severity.INFO,
                    message=f"{self.name}: adjustment status of the prices is unknown",
                )
            )
        if not self.point_in_time:
            flags.append(
                Flag(
                    source="data",
                    severity=Severity.INFO,
                    message=(
                        f"{self.name}: not point-in-time; restatements and symbol changes are "
                        "applied retroactively"
                    ),
                )
            )
        return flags
