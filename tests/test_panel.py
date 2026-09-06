"""Point-in-time guarantees of Panel and PanelView."""

import contextlib

import numpy as np
import pandas as pd
import pytest

from fullbacktester.data.panel import Panel
from fullbacktester.data.schema import SchemaError, validate_bars
from fullbacktester.markets import INDIA, US, Frequency


def test_view_cannot_see_past_its_end(panel):
    view = panel.view(50)
    assert view.n_bars == 50
    assert view.index == 49
    assert view.now == panel.timestamps[49]
    assert len(view.close) == 50
    assert view.close.index[-1] == view.now
    assert view.bars("S0").index[-1] <= view.now


def test_as_of_uses_bar_close_not_bar_date(panel):
    close_time = panel.timestamps[10]
    assert panel.as_of(close_time).n_bars == 11
    assert panel.as_of(close_time - pd.Timedelta(seconds=1)).n_bars == 10
    assert panel.as_of(close_time + pd.Timedelta(hours=5)).n_bars == 11


def test_as_of_rejects_naive_timestamps(panel):
    with pytest.raises(ValueError):
        panel.as_of(pd.Timestamp("2023-06-01"))


def test_view_bounds(panel):
    with pytest.raises(IndexError):
        panel.view(0)
    with pytest.raises(IndexError):
        panel.view(len(panel) + 1)


def test_arrays_are_read_only_and_strategy_writes_cannot_corrupt_panel(panel):
    original = panel.close.copy()
    view = panel.view(20)
    frame = view.close
    with contextlib.suppress(ValueError):  # pandas 2 raises (read-only buffer); pandas 3 copies
        frame.iloc[0, 0] = -1.0
    np.testing.assert_array_equal(panel.close, original)
    with pytest.raises(ValueError):
        panel.close[0, 0] = 1.0


def test_from_bars_aligns_symbols_with_different_calendars():
    us_days = pd.bdate_range("2024-01-02", periods=5)
    in_days = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for d in us_days:
        rows.append(
            {
                "timestamp": US.session_close_utc(d),
                "symbol": "AAPL",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        )
    for d in in_days:
        rows.append(
            {
                "timestamp": INDIA.session_close_utc(d),
                "symbol": "RELIANCE",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "volume": 1,
            }
        )
    panel = Panel.from_bars(pd.DataFrame(rows), market={"AAPL": "US", "RELIANCE": "IN"})
    assert len(panel) == 9  # interleaved clock: NSE closes before NYSE each day
    assert panel.symbols == ("AAPL", "RELIANCE")
    assert np.isnan(panel.close[:, 1]).sum() == 5
    assert np.isnan(panel.close[:, 0]).sum() == 4
    # NSE close at 15:30 IST (10:00 UTC) precedes NYSE close at 21:00 UTC on the same date
    assert panel.timestamps[0].hour == 10 and panel.timestamps[1].hour == 21


def test_bars_per_year_theoretical_for_single_market_and_empirical_otherwise(panel_factory):
    us_only = panel_factory(n_bars=100)
    assert us_only.bars_per_year == 252.0
    mixed_rows = us_only.to_bars()
    mixed_rows.loc[mixed_rows["symbol"] == "S0", "timestamp"] = [
        INDIA.session_close_utc(ts)
        for ts in mixed_rows.loc[mixed_rows["symbol"] == "S0", "timestamp"]
    ]
    mixed = Panel.from_bars(mixed_rows, market={"S0": "IN", "S1": "US", "S2": "US"})
    assert 400 < mixed.bars_per_year < 560  # two closes per weekday, empirical


def test_replace_future_keeps_prefix_identical_and_changes_suffix(panel):
    rng = np.random.default_rng(1)
    noisy = panel.replace_future(100, rng)
    np.testing.assert_array_equal(noisy.close[:100], panel.close[:100])
    np.testing.assert_array_equal(noisy.open[:100], panel.open[:100])
    assert not np.allclose(noisy.close[100:], panel.close[100:])
    assert (noisy.high[100:] >= noisy.low[100:]).all()
    assert noisy.timestamps.equals(panel.timestamps)


def test_last_known_close_forward_fills_missing_bars(panel_factory):
    panel = panel_factory(n_bars=10, missing={"S1": [8, 9]})
    view = panel.view(10)
    assert np.isnan(view.last_close["S1"])
    assert view.last_known_close()["S1"] == panel.close[7, 1]


def test_slice_and_subset(panel):
    part = panel.slice(10, 20)
    assert len(part) == 10
    assert part.timestamps[0] == panel.timestamps[10]
    sub = panel.subset(["S2", "S0"])
    assert sub.symbols == ("S2", "S0")
    np.testing.assert_array_equal(sub.close[:, 0], panel.close[:, 2])


def test_validate_bars_lists_every_problem():
    ts = US.session_close_utc(pd.Timestamp("2024-01-02"))
    bad = pd.DataFrame(
        {
            "timestamp": [ts, ts, ts],
            "symbol": ["A", "A", "B"],
            "open": [1.0, 1.0, 5.0],
            "high": [0.5, 1.0, 5.0],
            "low": [1.0, 1.0, 5.0],
            "close": [1.0, np.nan, 5.0],
            "volume": [1.0, -1.0, 1.0],
        }
    )
    with pytest.raises(SchemaError) as excinfo:
        validate_bars(bad)
    message = str(excinfo.value)
    assert "NaN close" in message
    assert "negative volume" in message
    assert "high < low" in message
    assert "share a (timestamp, symbol)" in message


def test_validate_bars_rejects_naive_timestamps():
    bad = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "symbol": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    with pytest.raises(SchemaError, match="naive"):
        validate_bars(bad)


def test_frequency_and_market_annualization():
    assert US.bars_per_year(Frequency.DAY_1) == 252
    assert INDIA.bars_per_year(Frequency.DAY_1) == 250
    assert US.bars_per_year(Frequency.MINUTE_30) == 252 * 13
    assert INDIA.bars_per_year(Frequency.MINUTE_15) == 250 * 25
    assert US.session_close_utc(pd.Timestamp("2024-01-10")).hour == 21  # EST
    assert US.session_close_utc(pd.Timestamp("2024-07-10")).hour == 20  # EDT
    assert INDIA.session_close_utc(pd.Timestamp("2024-07-10")).hour == 10
