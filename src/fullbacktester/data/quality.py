"""Data quality checks on a panel.

These do not fix anything. They tell you what the data cannot support:
missing sessions, zero-volume bars, single-bar moves that look like
unadjusted corporate actions, stale prices, and sparse symbols. Source-level
caveats (survivorship, adjustment) are appended when the source is known.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from fullbacktester.data.panel import Panel
from fullbacktester.flags import Flag, Severity
from fullbacktester.markets import Frequency

if TYPE_CHECKING:
    from fullbacktester.data.sources.base import DataSource

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
            # Compare the actual dates, not their counts. Subtracting counts lets an
            # unscheduled session hide a genuinely absent one: NSE's Diwali Muhurat
            # sessions are not in any standard calendar, and six of them masked three
            # real gaps in a live 2018-2024 pull.
            observed = {ts.tz_convert(market.tz).date() for ts in valid_ts}
            scheduled = {d.date() for d in market.expected_sessions(first, last)}
            absent = sorted(scheduled - observed)
            unscheduled = sorted(observed - scheduled)
            basis = "holiday calendar" if market.holiday_aware else "weekday count"

            if absent:
                hint = (
                    ""
                    if market.holiday_aware
                    else "; install exchange_calendars for holiday-aware counts"
                )
                flags.append(
                    Flag(
                        SOURCE,
                        Severity.WARN if market.holiday_aware else Severity.INFO,
                        f"{len(absent)} session(s) with no bar between {first} and {last} "
                        f"({basis}{hint}): {_sample_dates(absent)}",
                        symbol=symbol,
                    )
                )
            if unscheduled:
                flags.append(
                    Flag(
                        SOURCE,
                        Severity.INFO,
                        f"{len(unscheduled)} bar(s) on days the {basis} does not list as "
                        f"sessions: {_sample_dates(unscheduled)}. Special sessions such as "
                        "NSE Diwali Muhurat trading look like this and are genuine",
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


def _sample_dates(dates: list[date], limit: int = 4) -> str:
    shown = ", ".join(str(d) for d in dates[:limit])
    return shown if len(dates) <= limit else f"{shown}, and {len(dates) - limit} more"
