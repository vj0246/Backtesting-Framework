"""Data quality checks on a panel.

These do not fix anything. They tell you what the data cannot support:
missing sessions, zero-volume bars, single-bar moves that look like
unadjusted corporate actions, stale prices, and sparse symbols. Source-level
caveats (survivorship, adjustment) are appended when the source is known.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantgauntlet.data.panel import Panel
from quantgauntlet.flags import Flag, Severity
from quantgauntlet.markets import Frequency

if TYPE_CHECKING:
    from quantgauntlet.data.sources.base import DataSource

SOURCE = "data"


def check_data_quality(
    panel: Panel,
    *,
    source: DataSource | None = None,
    max_abs_return: float = 0.5,
    stale_run: int = 10,
    sparse_fraction: float = 0.5,
) -> list[Flag]:
    """Return findings about ``panel``. Empty means nothing the checks look for was found."""
    flags: list[Flag] = list(source.caveats()) if source is not None else []
    n_bars = len(panel)
    for j, symbol in enumerate(panel.symbols):
        close = panel.close[:, j]
        valid = np.isfinite(close)
        n_valid = int(valid.sum())
        if n_valid == 0:
            flags.append(Flag(SOURCE, Severity.HIGH, "no bars at all", symbol=symbol))
            continue
        if n_valid < sparse_fraction * n_bars:
            flags.append(
                Flag(
                    SOURCE,
                    Severity.INFO,
                    f"bars on only {n_valid}/{n_bars} panel timestamps",
                    symbol=symbol,
                )
            )

        valid_ts = panel.timestamps[valid]
        market = panel.market_of(symbol)
        if market is not None and panel.frequency is Frequency.DAY_1 and n_valid > 1:
            first = valid_ts[0].tz_convert(market.tz).date()
            last = valid_ts[-1].tz_convert(market.tz).date()
            expected = len(market.expected_sessions(first, last))
            missing = expected - n_valid
            if missing > 0:
                severity = Severity.WARN if market.holiday_aware else Severity.INFO
                basis = "holiday calendar" if market.holiday_aware else "weekday count"
                flags.append(
                    Flag(
                        SOURCE,
                        severity,
                        f"{missing} session(s) missing between {first} and {last} ({basis}; "
                        "install exchange_calendars for holiday-aware counts)"
                        if not market.holiday_aware
                        else f"{missing} session(s) missing between {first} and {last} ({basis})",
                        symbol=symbol,
                    )
                )

        volume = panel.volume[:, j][valid]
        zero_volume = int(np.sum(volume == 0))
        if zero_volume:
            flags.append(
                Flag(
                    SOURCE,
                    Severity.INFO,
                    f"{zero_volume} bar(s) with zero volume ({zero_volume / n_valid:.1%})",
                    symbol=symbol,
                )
            )

        prices = close[valid]
        returns = prices[1:] / prices[:-1] - 1.0
        extreme = np.flatnonzero(np.abs(returns) > max_abs_return)
        for k in extreme[:5]:
            when = valid_ts[k + 1].date()
            flags.append(
                Flag(
                    SOURCE,
                    Severity.WARN,
                    f"{returns[k]:+.1%} single-bar move on {when}: "
                    "unadjusted split/bonus or bad tick",
                    symbol=symbol,
                )
            )
        if len(extreme) > 5:
            flags.append(
                Flag(
                    SOURCE,
                    Severity.WARN,
                    f"{len(extreme) - 5} further single-bar moves beyond {max_abs_return:.0%}",
                    symbol=symbol,
                )
            )

        longest = _longest_constant_run(prices)
        if longest >= stale_run:
            flags.append(
                Flag(
                    SOURCE,
                    Severity.INFO,
                    f"close unchanged for {longest} consecutive bars: stale or illiquid",
                    symbol=symbol,
                )
            )
    return flags


def _longest_constant_run(values: np.ndarray) -> int:
    if len(values) < 2:
        return len(values)
    same = values[1:] == values[:-1]
    longest = run = 0
    for flag in same:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest + 1 if longest else 1
