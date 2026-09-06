"""Shared fixtures: synthetic panels with a real market clock."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quantgauntlet.data.panel import Panel
from quantgauntlet.markets import US, Frequency, Market

PanelFactory = Callable[..., Panel]


def make_panel(
    n_bars: int = 260,
    n_symbols: int = 3,
    *,
    seed: int = 0,
    market: Market = US,
    start: str = "2023-01-02",
    volume: float = 1_000_000.0,
    drift: float = 0.0,
    missing: dict[str, list[int]] | None = None,
) -> Panel:
    """Geometric random walk OHLCV panel stamped at the market's session closes."""
    rng = np.random.default_rng(seed)
    # Real sessions (holiday-aware when exchange_calendars is installed, weekdays otherwise),
    # so the synthetic panel is consistent with what the quality checks expect.
    first = pd.Timestamp(start).date()
    last = (pd.Timestamp(start) + pd.Timedelta(days=int(n_bars * 1.6) + 14)).date()
    dates = market.expected_sessions(first, last)[:n_bars]
    assert len(dates) == n_bars
    stamps = pd.DatetimeIndex([market.session_close_utc(d) for d in dates])
    symbols = [f"S{i}" for i in range(n_symbols)]
    log_returns = rng.normal(drift, 0.01, size=(n_bars, n_symbols))
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=close.shape)))
    vol = np.full(close.shape, volume)
    if missing:
        for symbol, rows in missing.items():
            j = symbols.index(symbol)
            for field in (open_, high, low, close, vol):
                field[rows, j] = np.nan
    frames = {
        name: pd.DataFrame(arr, index=stamps, columns=symbols)
        for name, arr in (
            ("open", open_),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", vol),
        )
    }
    return Panel.from_wide(
        frames["close"],
        open=frames["open"],
        high=frames["high"],
        low=frames["low"],
        volume=frames["volume"],
        frequency=Frequency.DAY_1,
        market=market,
    )


@pytest.fixture
def panel_factory() -> PanelFactory:
    return make_panel


@pytest.fixture
def panel() -> Panel:
    return make_panel()
