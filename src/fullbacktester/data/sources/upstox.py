"""Upstox historical candles for Indian equities.

No account is needed today: as of September 2026 Upstox serves historical candles
without authentication, although its documentation says a token is required. If
``UPSTOX_ANALYTICS_TOKEN`` is set it is sent with every request, so nothing breaks
if Upstox starts enforcing auth; if it does and no token is set, the error says
exactly what to do. The Analytics Token is free, lasts a year, and is read-only:
it cannot place or modify orders, so a leaked token cannot trade anyone's account.

Symbols are NSE trading symbols (``"RELIANCE"``), mapped through Upstox's public
instrument master, or full instrument keys (``"NSE_EQ|INE002A01018"``) used as-is.

Per Upstox, prices are adjusted for splits and bonuses but not dividends, so
returns are price returns. Only instruments that exist today are listed, so the
universe is not survivorship-free. Daily history starts in 2000, intraday in 2022.
"""

from __future__ import annotations

import difflib
import gzip
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

import pandas as pd

from fullbacktester.data.cache import default_cache_root
from fullbacktester.data.schema import BAR_COLUMNS
from fullbacktester.data.sources._http import (
    CredentialError,
    HttpResponse,
    HttpSession,
    default_session,
    error_message,
    get_with_retries,
)
from fullbacktester.data.sources.base import DataSource
from fullbacktester.flags import Flag, Severity
from fullbacktester.markets import Frequency, Market

TOKEN_ENV = "UPSTOX_ANALYTICS_TOKEN"
TOKEN_HELP = (
    "Generate a free, read-only Analytics Token on the Upstox Developer Apps page "
    "(Analytics tab, Generate Token); it lasts one year. "
    "See https://upstox.com/developer/api-documentation/analytics-token/"
)
_CANDLES_URL = "https://api.upstox.com/v3/historical-candle/{key}/{unit}/{interval}/{to}/{start}"
_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_MASTER_MAX_AGE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class _Spec:
    unit: str
    interval: str
    max_days: int | None  # longest span Upstox serves in one request; None = unlimited
    earliest: date


_DAILY_FROM = date(2000, 1, 1)
_INTRADAY_FROM = date(2022, 1, 1)
_SPECS: dict[Frequency, _Spec] = {
    Frequency.MINUTE_1: _Spec("minutes", "1", 28, _INTRADAY_FROM),
    Frequency.MINUTE_5: _Spec("minutes", "5", 28, _INTRADAY_FROM),
    Frequency.MINUTE_15: _Spec("minutes", "15", 28, _INTRADAY_FROM),
    Frequency.MINUTE_30: _Spec("minutes", "30", 88, _INTRADAY_FROM),
    Frequency.HOUR_1: _Spec("hours", "1", 88, _INTRADAY_FROM),
    Frequency.DAY_1: _Spec("days", "1", 3600, _DAILY_FROM),
    Frequency.WEEK_1: _Spec("weeks", "1", None, _DAILY_FROM),
}


class UpstoxSource(DataSource):
    name = "upstox"
    markets = frozenset({"IN"})
    frequencies = frozenset(_SPECS)
    adjusted: ClassVar[bool | None] = True
    survivorship_free = False

    def __init__(
        self,
        *,
        token: str | None = None,
        session: HttpSession | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._token = token
        self._session = session
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else default_cache_root() / "upstox"
        )
        self.timeout = timeout
        self._symbol_keys: dict[str, str] | None = None

    def __repr__(self) -> str:
        return f"UpstoxSource(cache_dir={str(self.cache_dir)!r})"

    def caveats(self) -> list[Flag]:
        return [
            *super().caveats(),
            Flag(
                source="data",
                severity=Severity.INFO,
                message=(
                    "upstox: prices are adjusted for splits and bonuses but not dividends, so "
                    "returns are price returns and understate total return by the dividend yield"
                ),
            ),
        ]

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        spec = _SPECS.get(frequency)
        if spec is None:
            raise ValueError(f"upstox does not serve {frequency.value} bars")
        token = self._token if self._token is not None else os.environ.get(TOKEN_ENV)
        keys = self.instrument_keys(symbols)
        headers = {"Accept": "application/json"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        sent_token = "Authorization" in headers

        frames = []
        for symbol, key in keys.items():
            rows: list[list[Any]] = []
            for lo, hi in _windows(max(start, spec.earliest), end, spec.max_days):
                url = _CANDLES_URL.format(
                    key=quote(key, safe=""),
                    unit=spec.unit,
                    interval=spec.interval,
                    to=hi.isoformat(),
                    start=lo.isoformat(),
                )
                response = get_with_retries(
                    self._http(), url, headers=headers, timeout=self.timeout
                )
                rows.extend(_candles(response, symbol, sent_token=sent_token))
            if rows:
                frames.append(_to_bars(rows, symbol, frequency, market))
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        return pd.concat(frames, ignore_index=True).loc[:, list(BAR_COLUMNS)]

    def instrument_keys(self, symbols: Sequence[str]) -> dict[str, str]:
        """Map each symbol to its Upstox instrument key; explain any that are unknown."""
        master: dict[str, str] | None = None
        resolved: dict[str, str] = {}
        unknown: list[str] = []
        for symbol in symbols:
            if "|" in symbol:
                resolved[symbol] = symbol
                continue
            if master is None:
                master = self._master()
            key = master.get(symbol.strip().upper())
            if key is None:
                unknown.append(symbol)
            else:
                resolved[symbol] = key
        if unknown:
            raise ValueError(_unknown_symbols_message(unknown, master or {}))
        return resolved

    def _master(self) -> dict[str, str]:
        if self._symbol_keys is not None:
            return self._symbol_keys
        path = self.cache_dir / "NSE.json.gz"
        stale = not path.exists() or time.time() - path.stat().st_mtime > _MASTER_MAX_AGE_SECONDS
        if stale:
            response = get_with_retries(
                self._http(), _INSTRUMENTS_URL, headers={"Accept": "*/*"}, timeout=self.timeout
            )
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            elif not path.exists():
                raise RuntimeError(
                    f"could not download the Upstox instrument master (HTTP "
                    f"{response.status_code}); symbols cannot be resolved"
                )
            # otherwise keep using yesterday's copy: listings change slowly
        payload = path.read_bytes()
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        self._symbol_keys = _symbol_map(json.loads(payload))
        return self._symbol_keys

    def _http(self) -> HttpSession:
        if self._session is None:
            self._session = default_session("upstox")
        return self._session


def _candles(response: HttpResponse, symbol: str, *, sent_token: bool) -> list[list[Any]]:
    status = response.status_code
    if status == 200:
        data = response.json().get("data") or {}
        return list(data.get("candles") or [])
    if status in (401, 403):
        reason = error_message(response)
        if sent_token:
            raise CredentialError(
                f"Upstox rejected {TOKEN_ENV} (HTTP {status}: {reason}). "
                f"Analytics tokens expire one year after generation. {TOKEN_HELP}"
            )
        raise CredentialError(
            f"Upstox now requires authentication for historical candles (HTTP {status}: "
            f"{reason}), and {TOKEN_ENV} is not set. {TOKEN_HELP} "
            f"Then set {TOKEN_ENV} for your user account."
        )
    if 400 <= status < 500:
        raise ValueError(
            f"Upstox refused candles for {symbol} (HTTP {status}: {error_message(response)})"
        )
    raise RuntimeError(
        f"Upstox candles for {symbol} still failing after retries "
        f"(HTTP {status}: {error_message(response)})"
    )


def _to_bars(
    rows: list[list[Any]], symbol: str, frequency: Frequency, market: Market
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [row[:6] for row in rows], columns=["label", "open", "high", "low", "close", "volume"]
    )
    frame["timestamp"] = [market.bar_close_utc(pd.Timestamp(v), frequency) for v in frame["label"]]
    frame["symbol"] = symbol
    frame = frame.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype("float64")
    return frame.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)


def _symbol_map(records: Any) -> dict[str, str]:
    """Trading symbol -> instrument key for NSE cash equities, preferring the EQ series."""
    out: dict[str, str] = {}
    for record in records:
        if record.get("segment") != "NSE_EQ" or not record.get("instrument_key"):
            continue
        symbol = str(record.get("trading_symbol", "")).upper()
        if symbol not in out or record.get("instrument_type") == "EQ":
            out[symbol] = str(record["instrument_key"])
    return out


def _windows(start: date, end: date, max_days: int | None) -> list[tuple[date, date]]:
    if start > end:
        return []
    if max_days is None:
        return [(start, end)]
    windows = []
    lo = start
    while lo <= end:
        hi = min(lo + timedelta(days=max_days - 1), end)
        windows.append((lo, hi))
        lo = hi + timedelta(days=1)
    return windows


def _unknown_symbols_message(unknown: list[str], master: dict[str, str]) -> str:
    hints = []
    for symbol in unknown:
        close = difflib.get_close_matches(symbol.strip().upper(), list(master), n=1)
        hints.append(f"{symbol!r} (did you mean {close[0]!r}?)" if close else repr(symbol))
    return (
        f"not NSE equity symbols on Upstox: {', '.join(hints)}. Use the NSE trading symbol, "
        "such as 'RELIANCE', or pass a full instrument key such as 'NSE_EQ|INE002A01018'."
    )
