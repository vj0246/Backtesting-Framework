"""Tests for what actually exists right now: the two concrete Strategy
adapters, and that the three ABCs correctly refuse direct instantiation.
No engine exists yet, so there's nothing to backtest — this only checks
that the contracts hold together.
"""

from datetime import datetime

import pytest

from ubt.core.interfaces import (
    DataFeed,
    ExecutionEngine,
    MLStrategy,
    RuleBasedStrategy,
    Signal,
    Strategy,
)

TS = datetime(2026, 1, 2)


def test_rule_based_strategy_emits_signals_for_nonzero_weights():
    strategy = RuleBasedStrategy(lambda df: {"AAPL": 0.5, "MSFT": 0.0, "BTC-USD": -0.3})

    signals = strategy.on_bar(TS, data=None)

    assert signals == [
        Signal(TS, "AAPL", 0.5),
        Signal(TS, "BTC-USD", -0.3),
    ]  # MSFT dropped: zero weight is "no position," not "position of size zero"


def test_ml_strategy_emits_signals_from_model_predict():
    class DummyModel:
        def predict(self, features):
            return {"AAPL": 0.2}

    strategy = MLStrategy(model=DummyModel(), feature_fn=lambda df: df, purge_embargo_bars=5)

    signals = strategy.on_bar(TS, data=None)

    assert signals == [Signal(TS, "AAPL", 0.2)]
    assert strategy.purge_embargo_bars == 5


@pytest.mark.parametrize("abc_cls", [Strategy, ExecutionEngine])
def test_abstract_bases_cannot_be_instantiated_directly(abc_cls):
    with pytest.raises(TypeError):
        abc_cls()


def test_datafeed_is_a_runtime_checkable_protocol():
    class FakeFeed:
        asset_class = "equity"

        def as_of(self, timestamp):
            return None

        def symbols(self):
            return ["AAPL"]

        def timestamps(self):
            return [TS]

    assert isinstance(FakeFeed(), DataFeed)  # structural check, not inheritance
