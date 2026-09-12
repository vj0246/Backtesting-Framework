"""Yahoo Finance via ``yfinance``. Optional dependency: ``pip install fullbacktester[yfinance]``.

Good for: free daily history for US and Indian equities (NSE ``.NS`` / BSE
``.BO`` suffixes), major crypto pairs (``BTC-USD``), quick research.

Not good for: survivorship (only currently listed tickers exist), intraday
depth (Yahoo caps 1-minute bars at about 30 days), reliability (unofficial API).
Prices are total-return adjusted when ``auto_adjust`` is True (the default),
which is what you want for returns and slightly wrong for exact fill prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from fullbacktester.data.schema import BAR_COLUMNS
from fullbacktester.data.sources.base import DataSource
from fullbacktester.markets import Frequency, Market

_INTERVAL = {
    Frequency.MINUTE_1: "1m",
    Frequency.MINUTE_5: "5m",
    Frequency.MINUTE_15: "15m",
    Frequency.MINUTE_30: "30m",
    Frequency.HOUR_1: "1h",
    Frequency.DAY_1: "1d",
    Frequency.WEEK_1: "1wk",
}
_MARKET_SUFFIX = {"IN": ".NS"}


class YFinanceSource(DataSource):
    name = "yfinance"
    markets = frozenset({"US", "IN", "CRYPTO"})
    frequencies = frozenset(_INTERVAL)
    adjusted = True
    survivorship_free = False

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust
        if not auto_adjust:
            self.adjusted = False  # type: ignore[misc]

    @staticmethod
    def ticker_for(symbol: str, market: Market) -> str:
        suffix = _MARKET_SUFFIX.get(market.code)
        if suffix and "." not in symbol and "-" not in symbol:
            return f"{symbol}{suffix}"
        return symbol

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is not installed; run: pip install 'fullbacktester[yfinance]'"
            ) from exc

        tickers = {self.ticker_for(s, market): s for s in symbols}
        raw = yf.download(
            list(tickers),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=_INTERVAL[frequency],
            auto_adjust=self.auto_adjust,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        frames = []
        for ticker, symbol in tickers.items():
            block = _extract(raw, ticker, single=len(tickers) == 1)
            if block is None or block.empty:
                continue
            block = block.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            block = block.dropna(subset=["close"])
            block.index = _close_instants(block.index, frequency, market)
            block = block.rename_axis("timestamp").reset_index()
            block.insert(1, "symbol", symbol)
            frames.append(block)
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        return pd.concat(frames, ignore_index=True).loc[:, list(BAR_COLUMNS)]


def _extract(raw: pd.DataFrame, ticker: str, *, single: bool) -> pd.DataFrame | None:
    if isinstance(raw.columns, pd.MultiIndex):
        block: object = None
        if ticker in raw.columns.get_level_values(0):
            block = raw[ticker]
        elif ticker in raw.columns.get_level_values(1):
            block = raw.xs(ticker, axis=1, level=1)
        return block if isinstance(block, pd.DataFrame) else None
    return raw if single else None


def _close_instants(index: pd.Index, frequency: Frequency, market: Market) -> pd.DatetimeIndex:
    # Yahoo labels bars by their start; a weekly bar is dated by its Monday.
    return pd.DatetimeIndex([market.bar_close_utc(ts, frequency) for ts in pd.DatetimeIndex(index)])
