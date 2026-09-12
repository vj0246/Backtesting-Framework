"""Alpaca historical bars for US equities, with the user's own API keys.

Setup, once: open a free Alpaca account (paper trading is enough), create API keys,
and set ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY``. Those are the names
Alpaca's own SDKs read, so an existing setup just works.

Uses the consolidated SIP feed with split and dividend adjustment. The free plan
serves SIP only for requests ending at least 15 minutes ago, so every request's end
is clamped to that and any bar not yet final by then is dropped. For end-of-day
bars this only matters if you run within 15 minutes of the close.

Alpaca stamps a daily bar at midnight New York time, the *start* of its day. Read
literally, that bar would look known sixteen hours before its last trade; every bar
is re-stamped to the instant it closes. Intraday bars include extended hours.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, ClassVar

import pandas as pd

from fullbacktester.data.schema import BAR_COLUMNS
from fullbacktester.data.sources._http import (
    CredentialError,
    HttpResponse,
    HttpSession,
    default_session,
    error_message,
    get_with_retries,
    require_credential,
)
from fullbacktester.data.sources.base import DataSource
from fullbacktester.markets import Frequency, Market

KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"
KEYS_HELP = "Create free API keys at https://app.alpaca.markets (paper-trading keys work for data)."
_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_TIMEFRAME = {
    Frequency.MINUTE_1: "1Min",
    Frequency.MINUTE_5: "5Min",
    Frequency.MINUTE_15: "15Min",
    Frequency.MINUTE_30: "30Min",
    Frequency.HOUR_1: "1Hour",
    Frequency.DAY_1: "1Day",
    Frequency.WEEK_1: "1Week",
}
_SIP_DELAY = pd.Timedelta(minutes=16)  # free plan: SIP only for data 15+ minutes old
_SYMBOLS_PER_REQUEST = 100
_PAGE_LIMIT = 10_000


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class AlpacaSource(DataSource):
    name = "alpaca"
    markets = frozenset({"US"})
    frequencies = frozenset(_TIMEFRAME)
    adjusted: ClassVar[bool | None] = True
    survivorship_free = False

    def __init__(
        self,
        *,
        key_id: str | None = None,
        secret_key: str | None = None,
        session: HttpSession | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._key_id = key_id
        self._secret_key = secret_key
        self._session = session
        self.timeout = timeout

    def __repr__(self) -> str:
        return "AlpacaSource()"

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        timeframe = _TIMEFRAME.get(frequency)
        if timeframe is None:
            raise ValueError(f"alpaca does not serve {frequency.value} bars")
        key_id = require_credential(self._key_id, KEY_ENV, provider="Alpaca", how_to_get=KEYS_HELP)
        secret = require_credential(
            self._secret_key, SECRET_ENV, provider="Alpaca", how_to_get=KEYS_HELP
        )
        headers = {
            "Accept": "application/json",
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        }
        begin = pd.Timestamp(start).tz_localize(market.tz).tz_convert("UTC")
        stop = min(market.session_close_utc(end), _utc_now() - _SIP_DELAY)
        if stop <= begin:
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        # Alpaca writes share classes with a dot (BRK.B); Yahoo users type BRK-B.
        labels = {symbol.strip().upper().replace("-", "."): symbol for symbol in symbols}
        collected: dict[str, list[dict[str, Any]]] = {}
        tickers = list(labels)
        for i in range(0, len(tickers), _SYMBOLS_PER_REQUEST):
            params: dict[str, Any] = {
                "symbols": ",".join(tickers[i : i + _SYMBOLS_PER_REQUEST]),
                "timeframe": timeframe,
                "start": _rfc3339(begin),
                "end": _rfc3339(stop),
                "adjustment": "all",
                "feed": "sip",
                "sort": "asc",
                "limit": _PAGE_LIMIT,
            }
            while True:
                response = get_with_retries(
                    self._http(), _BARS_URL, headers=headers, params=params, timeout=self.timeout
                )
                payload = _payload(response)
                for ticker, rows in (payload.get("bars") or {}).items():
                    collected.setdefault(ticker, []).extend(rows)
                token = payload.get("next_page_token")
                if not token:
                    break
                params = {**params, "page_token": token}

        frames = [
            _to_bars(rows, labels.get(ticker, ticker), frequency, market, stop)
            for ticker, rows in collected.items()
            if rows
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        return pd.concat(frames, ignore_index=True).loc[:, list(BAR_COLUMNS)]

    def _http(self) -> HttpSession:
        if self._session is None:
            self._session = default_session("alpaca")
        return self._session


def _payload(response: HttpResponse) -> dict[str, Any]:
    status = response.status_code
    if status == 200:
        body = response.json()
        return body if isinstance(body, dict) else {}
    message = error_message(response)
    if status in (401, 403):
        if "subscription" in message.lower():
            raise RuntimeError(
                f"Alpaca refused the request: {message}. The free plan serves SIP data only "
                "for requests ending 15 or more minutes ago, which this source already "
                "respects, so check that the account has market-data access."
            )
        raise CredentialError(
            f"Alpaca rejected {KEY_ENV} / {SECRET_ENV} (HTTP {status}: {message}). "
            f"Check both values and that they are a matching pair. {KEYS_HELP}"
        )
    if 400 <= status < 500:
        raise ValueError(f"Alpaca refused the bars request (HTTP {status}: {message})")
    raise RuntimeError(f"Alpaca bars still failing after retries (HTTP {status}: {message})")


def _to_bars(
    rows: list[dict[str, Any]],
    symbol: str,
    frequency: Frequency,
    market: Market,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = [market.bar_close_utc(pd.Timestamp(t), frequency) for t in frame["t"]]
    # A bar whose close falls after the data cut-off is still forming; drop it.
    frame = frame[frame["timestamp"] <= stop]
    frame = frame.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame["symbol"] = symbol
    frame = frame.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype("float64")
    return frame.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)


def _rfc3339(stamp: pd.Timestamp) -> str:
    return stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
