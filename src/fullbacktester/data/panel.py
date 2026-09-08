"""Point-in-time market data.

``Panel`` holds aligned OHLCV arrays for many symbols on one timeline.
``PanelView`` is the *only* thing a strategy ever sees: a window that ends at
the current bar and physically cannot index past it. Look-ahead through the
data layer is therefore a construction failure, not a runtime check.

The backing arrays are read-only, so a strategy that writes into a frame it
was handed gets a ``ValueError`` from NumPy instead of silently corrupting
the data every other strategy in an arena will be evaluated on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import numpy as np
import pandas as pd

from fullbacktester.data.schema import validate_bars
from fullbacktester.markets import Frequency, Market, get_market

_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _readonly(arr: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(arr, dtype=np.float64)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Panel:
    """Aligned OHLCV arrays, shape ``(n_bars, n_symbols)``, NaN where a symbol has no bar.

    Build with ``Panel.from_bars`` (canonical long frame) or ``Panel.from_wide``
    (one wide frame per field, convenient for tests and research notebooks).
    """

    timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    frequency: Frequency
    symbol_markets: Mapping[str, Market] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamps.tz is None:
            raise ValueError("Panel timestamps must be tz-aware UTC")
        if not self.timestamps.is_monotonic_increasing or self.timestamps.has_duplicates:
            raise ValueError("Panel timestamps must be strictly increasing")
        n_bars, n_symbols = len(self.timestamps), len(self.symbols)
        for name in _FIELDS:
            arr = getattr(self, name)
            if arr.shape != (n_bars, n_symbols):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(n_bars, n_symbols)}")
            object.__setattr__(self, name, _readonly(arr))
        if len(set(self.symbols)) != n_symbols:
            raise ValueError("symbols must be unique")
        object.__setattr__(self, "symbols", tuple(str(s) for s in self.symbols))

    # ---------------------------------------------------------------- builders

    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        *,
        frequency: Frequency = Frequency.DAY_1,
        market: str | Market | Mapping[str, str | Market] | None = None,
        symbols: Sequence[str] | None = None,
    ) -> Panel:
        """Build from a canonical long frame (validated here, so sources need not).

        ``market`` may be one market for every symbol or a per-symbol mapping.
        Timestamps become the union of every symbol's bar closes.
        """
        clean = validate_bars(bars)
        if symbols is None:
            ordered_symbols = tuple(dict.fromkeys(bars["symbol"].astype(str)))
        else:
            ordered_symbols = tuple(str(s) for s in symbols)
            clean = clean[clean["symbol"].isin(ordered_symbols)]
            present = set(clean["symbol"].unique())
            absent = [s for s in ordered_symbols if s not in present]
            if absent:
                raise ValueError(f"no bars for symbols {absent}")
        if clean.empty:
            raise ValueError("no bars")

        wide = clean.set_index(["timestamp", "symbol"])
        timestamps = pd.DatetimeIndex(wide.index.get_level_values(0).unique().sort_values())
        arrays = {}
        for name in _FIELDS:
            table = wide[name].unstack("symbol").reindex(index=timestamps, columns=ordered_symbols)
            arrays[name] = table.to_numpy(dtype=np.float64)
        return cls(
            timestamps=timestamps,
            symbols=ordered_symbols,
            frequency=frequency,
            symbol_markets=_resolve_markets(market, ordered_symbols),
            **arrays,
        )

    @classmethod
    def from_wide(
        cls,
        close: pd.DataFrame,
        *,
        open: pd.DataFrame | None = None,
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        frequency: Frequency = Frequency.DAY_1,
        market: str | Market | Mapping[str, str | Market] | None = None,
    ) -> Panel:
        """Build from wide frames indexed by tz-aware UTC bar close, columns = symbols.

        Missing OHLC fields default to ``close`` (a bar with no range) and missing
        volume defaults to ``+inf`` (no liquidity constraint). Convenient for
        synthetic data; real sources should use ``from_bars``.
        """
        index = pd.DatetimeIndex(close.index)
        if index.tz is None:
            raise ValueError("close index must be tz-aware UTC bar close times")
        index = index.tz_convert("UTC")
        symbols = tuple(str(c) for c in close.columns)

        def _align(frame: pd.DataFrame | None, default: np.ndarray) -> np.ndarray:
            if frame is None:
                return default
            aligned = frame.reindex(index=close.index, columns=close.columns)
            return aligned.to_numpy(dtype=np.float64)

        close_arr = close.to_numpy(dtype=np.float64)
        return cls(
            timestamps=index,
            symbols=symbols,
            open=_align(open, close_arr),
            high=_align(high, close_arr),
            low=_align(low, close_arr),
            close=close_arr,
            volume=_align(volume, np.full_like(close_arr, np.inf)),
            frequency=frequency,
            symbol_markets=_resolve_markets(market, symbols),
        )

    # ------------------------------------------------------------- properties

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def start(self) -> pd.Timestamp:
        return self.timestamps[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.timestamps[-1]

    @cached_property
    def bars_per_year(self) -> float:
        """Annualization factor.

        Theoretical (market sessions x bars per session) when every symbol
        shares one market; otherwise empirical: bars observed per year of span.
        """
        markets = {m.code: m for m in self.symbol_markets.values()}
        if len(markets) == 1:
            return next(iter(markets.values())).bars_per_year(self.frequency)
        if len(self) < 2:
            return float(len(self))
        span_years = (self.end - self.start) / pd.Timedelta(days=365.25)
        if span_years <= 0:
            return float(len(self))
        return len(self) / span_years

    def market_of(self, symbol: str) -> Market | None:
        return self.symbol_markets.get(symbol)

    # ----------------------------------------------------------------- access

    def index_of(self, timestamp: datetime | pd.Timestamp) -> int:
        """Number of bars closed at or before ``timestamp`` (a valid view length)."""
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            raise ValueError("timestamp must be tz-aware")
        return int(self.timestamps.searchsorted(ts.tz_convert("UTC"), side="right"))

    def as_of(self, timestamp: datetime | pd.Timestamp) -> PanelView:
        """Everything known at ``timestamp``: bars that closed at or before it."""
        return self.view(self.index_of(timestamp))

    def view(self, end: int) -> PanelView:
        """Bars ``[0, end)``. ``end`` must be in ``1..len(self)``."""
        if not 1 <= end <= len(self):
            raise IndexError(f"view end {end} outside 1..{len(self)}")
        return PanelView(self, end)

    def slice(self, start: int, end: int) -> Panel:
        """Sub-panel of bars ``[start, end)``; used for sub-period and CSCV analysis."""
        if not 0 <= start < end <= len(self):
            raise IndexError(f"slice [{start}, {end}) outside 0..{len(self)}")
        return Panel(
            timestamps=self.timestamps[start:end],
            symbols=self.symbols,
            frequency=self.frequency,
            symbol_markets=self.symbol_markets,
            **{name: getattr(self, name)[start:end] for name in _FIELDS},
        )

    def subset(self, symbols: Sequence[str]) -> Panel:
        cols = [self.symbols.index(s) for s in symbols]
        return Panel(
            timestamps=self.timestamps,
            symbols=tuple(symbols),
            frequency=self.frequency,
            symbol_markets={s: self.symbol_markets[s] for s in symbols if s in self.symbol_markets},
            **{name: getattr(self, name)[:, cols] for name in _FIELDS},
        )

    def replace_future(self, end: int, rng: np.random.Generator) -> Panel:
        """Copy with every bar at index >= ``end`` replaced by noise around the last known price.

        Used by the perturbation tester: a signal at bar ``end - 1`` that changes
        when this panel is substituted for the original was reading the future.
        """
        if not 1 <= end <= len(self):
            raise IndexError(f"end {end} outside 1..{len(self)}")
        n_future = len(self) - end
        if n_future == 0:
            return self
        last = np.nan_to_num(self.close[end - 1], nan=1.0)
        shocks = rng.standard_normal((n_future, self.n_symbols)) * 0.02
        noisy_close = last * np.exp(np.cumsum(shocks, axis=0))
        noisy_open = noisy_close * (1 + rng.standard_normal(noisy_close.shape) * 0.005)
        noisy_high = np.maximum(noisy_open, noisy_close) * (
            1 + np.abs(rng.standard_normal(noisy_close.shape)) * 0.005
        )
        noisy_low = np.minimum(noisy_open, noisy_close) * (
            1 - np.abs(rng.standard_normal(noisy_close.shape)) * 0.005
        )
        noisy_volume = np.where(
            np.isfinite(self.volume[end:]),
            self.volume[end:] * rng.uniform(0.5, 1.5, noisy_close.shape),
            self.volume[end:],
        )
        return Panel(
            timestamps=self.timestamps,
            symbols=self.symbols,
            frequency=self.frequency,
            symbol_markets=self.symbol_markets,
            open=np.vstack([self.open[:end], noisy_open]),
            high=np.vstack([self.high[:end], noisy_high]),
            low=np.vstack([self.low[:end], noisy_low]),
            close=np.vstack([self.close[:end], noisy_close]),
            volume=np.vstack([self.volume[:end], noisy_volume]),
        )

    def to_bars(self) -> pd.DataFrame:
        """Canonical long frame (drops rows where a symbol has no bar)."""
        frames = []
        for j, symbol in enumerate(self.symbols):
            frame = pd.DataFrame(
                {name: getattr(self, name)[:, j] for name in _FIELDS}, index=self.timestamps
            )
            frame = frame.dropna(subset=["close"])
            frame.insert(0, "symbol", symbol)
            frames.append(frame.rename_axis("timestamp").reset_index())
        return validate_bars(pd.concat(frames, ignore_index=True))


class PanelView:
    """Bars ``[0, end)`` of a panel. This is what strategies receive.

    Frame properties are built lazily and cached per view. They are backed by
    read-only slices of the panel arrays: no copy, no mutation, no future.
    """

    __slots__ = ("_end", "_frames", "_panel")

    def __init__(self, panel: Panel, end: int) -> None:
        self._panel = panel
        self._end = end
        self._frames: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- metadata

    @property
    def now(self) -> pd.Timestamp:
        """Close time of the most recent bar in this view."""
        return self._panel.timestamps[self._end - 1]

    @property
    def n_bars(self) -> int:
        return self._end

    @property
    def index(self) -> int:
        """Zero-based index of the current bar within the panel."""
        return self._end - 1

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._panel.symbols

    @property
    def frequency(self) -> Frequency:
        return self._panel.frequency

    @property
    def bars_per_year(self) -> float:
        return self._panel.bars_per_year

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._panel.timestamps[: self._end]

    def __len__(self) -> int:
        return self._end

    # ------------------------------------------------------------------ frames

    def _frame(self, name: str) -> pd.DataFrame:
        cached = self._frames.get(name)
        if cached is None:
            arr = getattr(self._panel, name)[: self._end]
            cached = pd.DataFrame(arr, index=self.timestamps, columns=list(self._panel.symbols))
            self._frames[name] = cached
        return cached

    @property
    def open(self) -> pd.DataFrame:
        return self._frame("open")

    @property
    def high(self) -> pd.DataFrame:
        return self._frame("high")

    @property
    def low(self) -> pd.DataFrame:
        return self._frame("low")

    @property
    def close(self) -> pd.DataFrame:
        return self._frame("close")

    @property
    def volume(self) -> pd.DataFrame:
        return self._frame("volume")

    @property
    def last_close(self) -> pd.Series:
        """Most recent close per symbol; NaN where the symbol has no bar yet."""
        return pd.Series(self._panel.close[self._end - 1], index=list(self._panel.symbols))

    def last_known_close(self) -> pd.Series:
        """Most recent *non-NaN* close per symbol (forward-filled), for marking positions."""
        arr = self._panel.close[: self._end]
        out = np.full(arr.shape[1], np.nan)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            valid = np.flatnonzero(~np.isnan(col))
            if valid.size:
                out[j] = col[valid[-1]]
        return pd.Series(out, index=list(self._panel.symbols))

    def bars(self, symbol: str) -> pd.DataFrame:
        """OHLCV frame for one symbol, rows where it had a bar."""
        j = self._panel.symbols.index(symbol)
        frame = pd.DataFrame(
            {name: getattr(self._panel, name)[: self._end, j] for name in _FIELDS},
            index=self.timestamps,
        )
        return frame.dropna(subset=["close"])


def _resolve_markets(
    market: str | Market | Mapping[str, str | Market] | None, symbols: Sequence[str]
) -> dict[str, Market]:
    if market is None:
        return {}
    if isinstance(market, Mapping):
        return {s: get_market(market[s]) for s in symbols if s in market}
    resolved = get_market(market)
    return {s: resolved for s in symbols}
