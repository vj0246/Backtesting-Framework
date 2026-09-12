"""Market definitions: timezone, session hours, trading days, and annualization.

Every bar in fullbacktester is stamped with the UTC instant at which it *closed*,
because that is the first moment its values are knowable. Daily bars therefore
need the market's session close to be converted to an instant, which is why the
market is attached to the data rather than assumed by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")


class Frequency(StrEnum):
    """Bar frequency. The value is the canonical short code used in cache paths."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

    @property
    def minutes(self) -> int:
        return _FREQUENCY_MINUTES[self]

    @property
    def timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.minutes)

    @property
    def is_intraday(self) -> bool:
        return self.minutes < _FREQUENCY_MINUTES[Frequency.DAY_1]


_FREQUENCY_MINUTES: dict[Frequency, int] = {
    Frequency.MINUTE_1: 1,
    Frequency.MINUTE_5: 5,
    Frequency.MINUTE_15: 15,
    Frequency.MINUTE_30: 30,
    Frequency.HOUR_1: 60,
    Frequency.DAY_1: 24 * 60,
    Frequency.WEEK_1: 7 * 24 * 60,
}

_WEEKDAYS = frozenset({0, 1, 2, 3, 4})
_ALL_DAYS = frozenset({0, 1, 2, 3, 4, 5, 6})


@dataclass(frozen=True)
class Market:
    """A trading venue's clock.

    Attributes:
        code: Short identifier used in cache paths and source registries.
        tz: IANA timezone of the session times.
        session_open: Local wall-clock open.
        session_close: Local wall-clock close.
        sessions_per_year: Typical number of trading sessions per calendar year.
            Used for annualization when the data frequency is daily.
        calendar_code: ``exchange_calendars`` code for holiday-aware session
            enumeration, or None when the market never closes.
        trading_weekdays: Weekday numbers (Monday=0) on which sessions occur.
        close_offset_days: Days to add to the session date to find the close
            instant. A 24/7 market's "day D" bar closes at midnight of D+1.
    """

    code: str
    tz: str
    session_open: time
    session_close: time
    sessions_per_year: int
    calendar_code: str | None
    trading_weekdays: frozenset[int] = _WEEKDAYS
    close_offset_days: int = 0

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def session_minutes(self) -> int:
        open_minutes = self.session_open.hour * 60 + self.session_open.minute
        close_minutes = self.session_close.hour * 60 + self.session_close.minute
        span = close_minutes - open_minutes + self.close_offset_days * 24 * 60
        if span <= 0:
            raise ValueError(f"market {self.code!r} has a non-positive session length")
        return span

    def session_close_utc(self, session: date | pd.Timestamp | datetime) -> pd.Timestamp:
        """UTC instant at which the given session date's bar closes."""
        day = pd.Timestamp(session).date()
        local = datetime.combine(day, self.session_close, tzinfo=self.zone)
        local = local + pd.Timedelta(days=self.close_offset_days)
        return pd.Timestamp(local).tz_convert("UTC")

    def session_open_utc(self, session: date | pd.Timestamp | datetime) -> pd.Timestamp:
        """UTC instant at which the given session date opens."""
        day = pd.Timestamp(session).date()
        local = datetime.combine(day, self.session_open, tzinfo=self.zone)
        return pd.Timestamp(local).tz_convert("UTC")

    def bar_close_utc(self, label: pd.Timestamp, frequency: Frequency) -> pd.Timestamp:
        """UTC instant at which a bar labelled by the *start* of its interval is fully known.

        Providers label bars by where they begin: Yahoo dates a weekly bar by its
        Monday, Alpaca stamps a daily bar at midnight New York time. Read literally,
        both make a bar look known days or hours before its last trade.

        Daily: the session close of the label's local date. Weekly: the close of the
        last scheduled session in the seven days from the label (holiday-aware with
        ``exchange_calendars``, otherwise the weekday mask, which can only err late).
        Intraday: label plus bar length. Every rule errs late rather than early.
        """
        stamp = pd.Timestamp(label)
        local = stamp.tz_localize(self.tz) if stamp.tzinfo is None else stamp.tz_convert(self.tz)
        if frequency.is_intraday:
            return (local + frequency.timedelta).tz_convert("UTC")
        day = local.date()
        if frequency is Frequency.WEEK_1:
            sessions = self.expected_sessions(day, day + timedelta(days=6))
            return self.session_close_utc(sessions[-1] if len(sessions) else day)
        return self.session_close_utc(day)

    def bars_per_year(self, frequency: Frequency) -> float:
        """Number of bars in a year at ``frequency``, for annualizing statistics."""
        if frequency is Frequency.WEEK_1:
            return 52.0
        if frequency is Frequency.DAY_1:
            return float(self.sessions_per_year)
        return self.sessions_per_year * (self.session_minutes / frequency.minutes)

    def expected_sessions(self, start: date, end: date) -> pd.DatetimeIndex:
        """Session dates between ``start`` and ``end`` inclusive.

        Uses ``exchange_calendars`` when installed (holiday-aware); otherwise
        falls back to the weekday mask, which over-counts holidays. Callers
        that report missing sessions should say which path was used.
        """
        calendar = _load_exchange_calendar(self.calendar_code)
        if calendar is not None:
            sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
            return pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        days = pd.date_range(start, end, freq="D")
        return days[[d.weekday() in self.trading_weekdays for d in days]]

    @property
    def holiday_aware(self) -> bool:
        return _load_exchange_calendar(self.calendar_code) is not None


@cache
def _load_exchange_calendar(code: str | None) -> Any:
    """The ``exchange_calendars`` calendar for ``code``, or None if unavailable."""
    if code is None:
        return None
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        return xcals.get_calendar(code)
    except xcals.errors.InvalidCalendarName:
        return None


US = Market(
    code="US",
    tz="America/New_York",
    session_open=time(9, 30),
    session_close=time(16, 0),
    sessions_per_year=252,
    calendar_code="XNYS",
)

INDIA = Market(
    code="IN",
    tz="Asia/Kolkata",
    session_open=time(9, 15),
    session_close=time(15, 30),
    sessions_per_year=250,
    calendar_code="XBOM",  # exchange_calendars ships BSE, not NSE; the holiday sets match
)

CRYPTO = Market(
    code="CRYPTO",
    tz="UTC",
    session_open=time(0, 0),
    session_close=time(0, 0),
    sessions_per_year=365,
    calendar_code=None,
    trading_weekdays=_ALL_DAYS,
    close_offset_days=1,
)

MARKETS: dict[str, Market] = {m.code: m for m in (US, INDIA, CRYPTO)}


def get_market(market: str | Market) -> Market:
    """Resolve a market code or instance. Raises ``KeyError`` for unknown codes."""
    if isinstance(market, Market):
        return market
    try:
        return MARKETS[market.upper()]
    except KeyError:
        known = ", ".join(sorted(MARKETS))
        raise KeyError(f"unknown market {market!r}; known markets: {known}") from None
