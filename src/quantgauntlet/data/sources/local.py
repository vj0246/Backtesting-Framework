"""Local CSV / Parquet files.

Layout: one file per symbol under ``root`` (``AAPL.csv``, ``RELIANCE.parquet``)
or one long-format file with a ``symbol`` column. Column names are matched
case-insensitively; ``date``/``datetime``/``time`` are accepted for the
timestamp, ``vol`` for volume, and an ``adj close`` column is ignored.

Timestamps: tz-aware values are taken as bar close instants. Naive daily
values are session dates and become the market's session close. Naive
intraday values are localized to the market timezone and taken as bar
*close* times; if your file stamps bars at their open, shift them yourself,
because the loader cannot know.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd

from quantgauntlet.data.schema import BAR_COLUMNS
from quantgauntlet.data.sources.base import DataSource
from quantgauntlet.markets import Frequency, Market

_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "ticker": "symbol",
    "vol": "volume",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}
_SUFFIXES = (".parquet", ".csv")


class LocalSource(DataSource):
    name = "local"
    markets = frozenset({"US", "IN", "CRYPTO"})
    frequencies = frozenset(Frequency)
    adjusted: ClassVar[bool | None] = None
    survivorship_free = False

    def __init__(self, root: Path | str, *, adjusted: bool | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if adjusted is not None:
            self.adjusted = adjusted  # type: ignore[misc]

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        if self.root.is_file():
            frame = _read(self.root)
            if "symbol" not in frame.columns:
                raise ValueError(f"{self.root} has no symbol column; long format needed")
            frame = frame[frame["symbol"].astype(str).isin(list(symbols))]
            frames = [frame]
        else:
            frames = []
            for symbol in symbols:
                path = self._file_for(symbol)
                if path is None:
                    continue
                frame = _read(path)
                frame["symbol"] = symbol
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        bars = pd.concat(frames, ignore_index=True)
        bars["timestamp"] = _to_close_instants(bars["timestamp"], frequency, market)
        lo = market.session_open_utc(start)
        hi = market.session_close_utc(end)
        bars = bars[(bars["timestamp"] >= lo) & (bars["timestamp"] <= hi)]
        if "volume" not in bars.columns:
            bars["volume"] = 0.0
        return bars.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)

    def _file_for(self, symbol: str) -> Path | None:
        for suffix in _SUFFIXES:
            candidate = self.root / f"{symbol}{suffix}"
            if candidate.exists():
                return candidate
        return None


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        key = _ALIASES.get(key, key)
        renamed[column] = key
    frame = frame.rename(columns=renamed)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path}: no timestamp/date column found in {list(frame.columns)}")
    return frame


def _to_close_instants(values: pd.Series, frequency: Frequency, market: Market) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is not None:
        return parsed.dt.tz_convert("UTC")
    if frequency.is_intraday:
        return parsed.dt.tz_localize(market.tz).dt.tz_convert("UTC")
    return pd.Series([market.session_close_utc(d) for d in parsed], index=values.index)
