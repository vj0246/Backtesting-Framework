"""Engine behaviour and the cross-engine equivalence property."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantgauntlet.execution import (
    CostModel,
    EngineConfig,
    EventDrivenEngine,
    FillTiming,
    OrderType,
    VectorizedEngine,
    VolumeShareSlippage,
    compare_engines,
)
from quantgauntlet.execution.costs import BpsCommission
from quantgauntlet.flags import Severity
from quantgauntlet.strategy import BatchRuleStrategy, RuleBasedStrategy, Strategy
from tests.conftest import make_panel


def momentum(view):
    close = view.close
    signal = (close.iloc[-1] / close.iloc[-21] - 1.0 > 0).astype(float)
    return signal / max(signal.sum(), 1.0)


def batch_momentum(view):
    close = view.close
    signal = (close / close.shift(20) - 1.0 > 0).astype(float)
    return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)


class BuyAndHold(Strategy):
    def on_bar(self, ctx):
        if ctx.bar == 0:
            ctx.order_target_weights({s: 1.0 / len(ctx.symbols) for s in ctx.symbols})


class TableStrategy(BatchRuleStrategy):
    """Replays a fixed weight table; rows depend only on their own timestamp."""

    def __init__(self, table: pd.DataFrame) -> None:
        super().__init__(lambda view: table.loc[view.timestamps], name="table")


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    commission_bps=st.floats(0, 25),
    slippage_bps=st.floats(0, 25),
    nan_fraction=st.floats(0, 0.4),
    leverage=st.floats(0.2, 2.0),
)
def test_engines_agree_exactly_under_idealized_execution(
    seed, commission_bps, slippage_bps, nan_fraction, leverage
):
    panel = make_panel(n_bars=60, n_symbols=4, seed=seed)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(panel), panel.n_symbols))
    weights = raw / np.abs(raw).sum(axis=1, keepdims=True) * leverage
    weights[rng.random(weights.shape) < nan_fraction] = np.nan
    table = pd.DataFrame(weights, index=panel.timestamps, columns=list(panel.symbols))
    cost_model = CostModel(
        commission=BpsCommission(commission_bps),
        slippage=VolumeShareSlippage(base_bps=slippage_bps),
    )
    comparison = compare_engines(TableStrategy(table), panel, EngineConfig.idealized(cost_model))
    assert comparison.max_equity_gap < 1e-9
    assert comparison.max_weight_gap < 1e-9
    assert len(comparison.vectorized.fills) == len(comparison.event_driven.fills)


def test_per_bar_and_batch_formulations_of_one_signal_match(panel):
    config = EngineConfig.idealized(CostModel.bps(5, 5))
    per_bar = EventDrivenEngine(config).run(RuleBasedStrategy(momentum, warmup=21), panel)
    batch = EventDrivenEngine(config).run(BatchRuleStrategy(batch_momentum, warmup=21), panel)
    pd.testing.assert_series_equal(per_bar.equity, batch.equity)


def test_realistic_config_diverges_from_idealized_by_lot_rounding_only(panel):
    config = EngineConfig(cost_model=CostModel.bps(5, 5), initial_cash=50_000)
    comparison = compare_engines(RuleBasedStrategy(momentum, warmup=21), panel, config)
    assert 0 < comparison.max_equity_gap < 0.01
    positions = comparison.event_driven.positions.to_numpy()
    assert np.allclose(positions, np.round(positions))
    assert not comparison.flags()


def test_orders_on_final_bar_never_fill_and_are_reported(panel_factory):
    panel = panel_factory(n_bars=30)
    result = EventDrivenEngine(EngineConfig()).run(RuleBasedStrategy(momentum, warmup=21), panel)
    last_bar = len(panel) - 1
    final = result.orders[result.orders["created_bar"] == last_bar]
    assert (final["status"] == "cancelled").all()
    assert any("final bar" in f.message for f in result.flags)


def test_same_close_timing_fills_immediately_and_is_flagged(panel_factory):
    panel = panel_factory(n_bars=10)
    config = EngineConfig(fill_timing=FillTiming.SAME_CLOSE, fractional_shares=True)
    result = EventDrivenEngine(config).run(BuyAndHold(), panel)
    assert result.fills["bar"].iloc[0] == 0
    assert result.fills["reference_price"].iloc[0] == pytest.approx(panel.close[0, 0])
    assert any(f.severity is Severity.WARN and "SAME_CLOSE" in f.message for f in result.flags)
    vectorized = VectorizedEngine(config).run(
        RuleBasedStrategy(lambda v: {s: 1 / 3 for s in v.symbols}), panel
    )
    assert vectorized.fills["bar"].iloc[0] == 0


def test_next_open_timing_fills_at_next_bar_open(panel_factory):
    panel = panel_factory(n_bars=10)
    result = EventDrivenEngine(EngineConfig(fractional_shares=True)).run(BuyAndHold(), panel)
    first = result.fills.iloc[0]
    assert first["bar"] == 1
    assert first["reference_price"] == pytest.approx(panel.open[1, 0])
    assert result.equity.iloc[0] == pytest.approx(100_000.0)


def test_warmup_delays_first_decision(panel_factory):
    panel = panel_factory(n_bars=40)
    result = EventDrivenEngine().run(RuleBasedStrategy(momentum, warmup=30), panel)
    assert result.orders["created_bar"].min() == 30
    assert (result.positions.iloc[:31] == 0).all().all()


def test_vectorized_engine_refuses_imperative_strategies(panel):
    with pytest.raises(TypeError, match="imperative"):
        VectorizedEngine().run(BuyAndHold(), panel)


def test_imperative_strategy_with_limit_and_stop_orders(panel_factory):
    panel = panel_factory(n_bars=15, seed=3)

    class Bracket(Strategy):
        def on_bar(self, ctx):
            if ctx.bar == 0:
                ctx.order("S0", 100)
            elif ctx.bar == 1:
                price = ctx.price("S0")
                ctx.order("S0", -100, order_type=OrderType.STOP, stop_price=price * 0.5)
                ctx.order("S0", 50, order_type=OrderType.LIMIT, limit_price=price * 0.5)

    result = EventDrivenEngine().run(Bracket(), panel)
    statuses = result.orders.set_index("order_type")["status"]
    assert statuses["market"] == "filled"
    assert statuses["stop"] == "cancelled"
    assert statuses["limit"] == "cancelled"
    assert result.positions["S0"].iloc[-1] == 100


def test_missing_bars_are_marked_at_last_close_and_not_traded(panel_factory):
    panel = panel_factory(n_bars=12, missing={"S1": [5, 6]})
    result = EventDrivenEngine(EngineConfig(fractional_shares=True)).run(
        RuleBasedStrategy(lambda v: {s: 1 / 3 for s in v.symbols}), panel
    )
    fills_on_gap = result.fills[
        (result.fills["bar"].isin([5, 6])) & (result.fills["symbol"] == "S1")
    ]
    assert fills_on_gap.empty
    assert np.isfinite(result.equity).all()
    comparison = compare_engines(
        RuleBasedStrategy(lambda v: {s: 1 / 3 for s in v.symbols}), panel, EngineConfig.idealized()
    )
    assert comparison.agrees()


def test_rejected_orders_appear_in_blotter(panel_factory):
    panel = panel_factory(n_bars=5)

    class Shorty(Strategy):
        def on_bar(self, ctx):
            if ctx.bar == 0:
                ctx.order("S0", -10)
                ctx.order("S0", 0.4)

    result = EventDrivenEngine(EngineConfig(allow_short=False)).run(Shorty(), panel)
    assert (result.orders["status"] == "rejected").sum() == 2
    assert any("rejected" in f.message for f in result.flags)


def test_result_metrics_and_turnover(panel):
    result = EventDrivenEngine(EngineConfig(cost_model=CostModel.bps(10, 0))).run(
        RuleBasedStrategy(momentum, warmup=21), panel
    )
    metrics = result.metrics()
    assert metrics.n_bars == len(panel)
    assert metrics.bars_per_year == 252
    assert metrics.n_fills == len(result.fills)
    assert metrics.total_costs == pytest.approx(result.fills["commission"].sum())
    assert result.turnover.max() <= 2.0 + 1e-9  # full rotation sells 100% and buys 100%
    assert result.returns.iloc[0] == 0.0
    assert result.summary()["engine"] == "event_driven"
