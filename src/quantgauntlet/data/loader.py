"""High-level loading: source registry + cache + validation in one call."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from quantgauntlet.data.cache import CacheKey, ParquetCache
from quantgauntlet.data.panel import Panel
from quantgauntlet.data.schema import BAR_COLUMNS, validate_bars
from quantgauntlet.data.sources import DataSource, default_source_name, get_source
from quantgauntlet.markets import Frequency, Market, get_market


def resolve_source(source: str | DataSource | None, market: Market) -> DataSource:
    if isinstance(source, DataSource):
        return source
    return get_source(source if source is not None else default_source_name(market.code))


def load_bars(
    symbols: Sequence[str],
    start: date | str,
    end: date | str,
    *,
    market: str | Market,
    frequency: Frequency = Frequency.DAY_1,
    source: str | DataSource | None = None,
    cache: ParquetCache | bool = True,
) -> pd.DataFrame:
    """Canonical bars for ``symbols`` between ``start`` and ``end`` (inclusive session dates).

    Cached symbols are served locally; the rest are fetched in one call to the
    source, validated, stored, and merged. ``cache=False`` bypasses storage
    entirely (useful for tests and one-off pulls).
    """
    resolved_market = get_market(market)
    start_date, end_date = _as_date(start), _as_date(end)
    if start_date > end_date:
        raise ValueError("start must not be after end")
    src = resolve_source(source, resolved_market)
    if not src.supports(resolved_market, frequency):
        raise ValueError(
            f"source {src.name!r} does not support market {resolved_market.code} "
            f"at {frequency.value}"
        )

    store = (
        ParquetCache() if cache is True else (cache if isinstance(cache, ParquetCache) else None)
    )
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for symbol in symbols:
        key = CacheKey(src.name, resolved_market.code, frequency, symbol)
        if store is not None and store.covers(key, start_date, end_date):
            cached = store.load(key)
            if cached is not None:
                frames.append(cached)
                continue
        missing.append(symbol)

    if missing:
        fetch_start, fetch_end = start_date, end_date
        if store is not None:
            for symbol in missing:
                fetched = store.fetched_range(
                    CacheKey(src.name, resolved_market.code, frequency, symbol)
                )
                if fetched is not None:
                    fetch_start, fetch_end = (
                        min(fetch_start, fetched[0]),
                        max(fetch_end, fetched[1]),
                    )
        fetched_bars = src.fetch(missing, fetch_start, fetch_end, frequency, resolved_market)
        fetched_bars = (
            validate_bars(fetched_bars)
            if not fetched_bars.empty
            else fetched_bars.reindex(columns=list(BAR_COLUMNS))
        )
        for symbol in missing:
            block = fetched_bars[fetched_bars["symbol"] == symbol]
            if store is not None:
                key = CacheKey(src.name, resolved_market.code, frequency, symbol)
                store.store(key, block, fetch_start, fetch_end)
            if not block.empty:
                frames.append(block)

    if not frames:
        raise ValueError(f"no bars returned for {list(symbols)} from {src.name}")
    bars = pd.concat(frames, ignore_index=True)
    lo = resolved_market.session_open_utc(start_date)
    hi = resolved_market.session_close_utc(end_date)
    bars = bars[(bars["timestamp"] >= lo) & (bars["timestamp"] <= hi)]
    return validate_bars(bars)


def load_panel(
    symbols: Sequence[str],
    start: date | str,
    end: date | str,
    *,
    market: str | Market,
    frequency: Frequency = Frequency.DAY_1,
    source: str | DataSource | None = None,
    cache: ParquetCache | bool = True,
) -> Panel:
    """``load_bars`` then ``Panel.from_bars`` with the market and frequency attached."""
    bars = load_bars(
        symbols, start, end, market=market, frequency=frequency, source=source, cache=cache
    )
    return Panel.from_bars(bars, frequency=frequency, market=get_market(market))


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)
