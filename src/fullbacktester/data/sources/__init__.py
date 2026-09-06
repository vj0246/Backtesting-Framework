"""Source registry with per-market defaults.

Register a new source with ``register``; pick one by name with ``get_source``.
Defaults favour adjusted prices (yfinance) because unadjusted prices produce
wrong returns; the official NSE archive is registered as ``"nse"`` for
survivorship-free universes and volume checks.
"""

from __future__ import annotations

from fullbacktester.data.sources.base import DataSource
from fullbacktester.data.sources.local import LocalSource
from fullbacktester.data.sources.nse import NSEBhavcopySource
from fullbacktester.data.sources.yfinance import YFinanceSource

SOURCES: dict[str, type[DataSource]] = {}
DEFAULT_SOURCE_BY_MARKET: dict[str, str] = {
    "US": "yfinance",
    "IN": "yfinance",
    "CRYPTO": "yfinance",
}


def register(cls: type[DataSource]) -> type[DataSource]:
    SOURCES[cls.name] = cls
    return cls


for _cls in (LocalSource, YFinanceSource, NSEBhavcopySource):
    register(_cls)


def available_sources() -> list[str]:
    return sorted(SOURCES)


def get_source(name: str, **kwargs: object) -> DataSource:
    try:
        cls = SOURCES[name]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; available: {available_sources()}") from None
    return cls(**kwargs)


def default_source_name(market_code: str) -> str:
    try:
        return DEFAULT_SOURCE_BY_MARKET[market_code]
    except KeyError:
        raise KeyError(f"no default source for market {market_code!r}; pass source=") from None


__all__ = [
    "DEFAULT_SOURCE_BY_MARKET",
    "SOURCES",
    "DataSource",
    "LocalSource",
    "NSEBhavcopySource",
    "YFinanceSource",
    "available_sources",
    "default_source_name",
    "get_source",
    "register",
]
