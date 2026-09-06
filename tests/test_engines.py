"""Tests for VectorizedEngine, EventDrivenEngine, and compare_engines().

Both engines currently share the exact same computation (see engines.py
module docstring) — these tests confirm that honestly. They don't test
genuine path-dependent divergence, because there isn't any yet.
"""

import pandas as pd
import pytest

from ubt.core.engines import EventDrivenEngine, VectorizedEngine, compare_engines
from ubt.core.interfaces import AssetClass, RuleBasedStrategy


class _FixedFeed:
    asset_class = AssetClass.EQUITY

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def as_of(self, timestamp):
        return self._df.loc[:timestamp]

    def symbols(self):
        return list(self._df.columns)

    def timestamps(self):
        return list(self._df.index)


@pytest.fixture
def rising_feed():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAA": [100, 102, 104, 103, 106, 108]}, index=dates)
    return _FixedFeed(prices)


def _always_long(data: pd.DataFrame) -> dict[str, float]:
    return {"AAA": 1.0}


def test_vectorized_engine_produces_positive_pnl_on_uptrend(rising_feed):
    result = VectorizedEngine().run(RuleBasedStrategy(_always_long), rising_feed, cost_bps=0.0)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
    assert result.engine_name == "vectorized"


def test_cost_bps_reduces_final_equity(rising_feed):
    strategy = RuleBasedStrategy(_always_long)
    no_cost = VectorizedEngine().run(strategy, rising_feed, cost_bps=0.0)
    with_cost = VectorizedEngine().run(strategy, rising_feed, cost_bps=50.0)
    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]


def test_flat_strategy_has_zero_turnover_after_entry(rising_feed):
    result = VectorizedEngine().run(RuleBasedStrategy(_always_long), rising_feed, cost_bps=10.0)
    # one entry trade on day 0, nothing after — "always long 1.0" never changes weight
    assert len(result.trades) == 1


def test_vectorized_and_event_driven_agree_today(rising_feed):
    spread = compare_engines(RuleBasedStrategy(_always_long), rising_feed, cost_bps=10.0)
    assert spread == pytest.approx(0.0, abs=1e-9)  # true by construction — see engines.py docstring


def test_event_driven_engine_matches_vectorized_result(rising_feed):
    strategy = RuleBasedStrategy(_always_long)
    v = VectorizedEngine().run(strategy, rising_feed, cost_bps=25.0)
    e = EventDrivenEngine().run(strategy, rising_feed, cost_bps=25.0)
    pd.testing.assert_series_equal(v.equity_curve, e.equity_curve, check_names=False)
