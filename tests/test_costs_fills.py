"""Cost models and the fill model's rules."""

import numpy as np
import pandas as pd
import pytest

from quantgauntlet.data.panel import Panel
from quantgauntlet.execution.config import EngineConfig
from quantgauntlet.execution.costs import (
    BpsCommission,
    CostModel,
    FixedSlippage,
    PerUnitCommission,
    VolumeShareSlippage,
)
from quantgauntlet.execution.fills import FillModel
from quantgauntlet.execution.ledger import Ledger
from quantgauntlet.execution.orders import Order, OrderStatus, OrderType, TimeInForce
from quantgauntlet.markets import US


def _two_bar_panel(open_=100.0, high=105.0, low=95.0, close=102.0, volume=1_000.0) -> Panel:
    stamps = pd.DatetimeIndex(
        [
            US.session_close_utc(pd.Timestamp("2024-01-02")),
            US.session_close_utc(pd.Timestamp("2024-01-03")),
        ]
    )
    frame = lambda v: pd.DataFrame({"X": [v, v]}, index=stamps)  # noqa: E731
    return Panel.from_wide(
        frame(close),
        open=frame(open_),
        high=frame(high),
        low=frame(low),
        volume=frame(volume),
        market=US,
    )


def _fill(
    order: Order, config: EngineConfig, panel: Panel | None = None, ledger: Ledger | None = None
):
    panel = panel or _two_bar_panel()
    ledger = ledger or Ledger(panel.symbols, config.initial_cash)
    marks = panel.close[0]
    fills, still_open = FillModel(config).fill_bar([order], panel, 1, ledger, marks)
    return fills, still_open, ledger


def test_bps_commission_with_minimum():
    model = BpsCommission(bps=10, minimum=5.0)
    assert model.commission(10, 100.0) == pytest.approx(5.0)  # 1000 notional * 10bps = 1 < min
    assert model.commission(1000, 100.0) == pytest.approx(100.0)
    assert model.commission(0, 100.0) == 0.0


def test_per_unit_commission():
    assert PerUnitCommission(0.005, minimum=1.0).commission(100, 50.0) == pytest.approx(1.0)
    assert PerUnitCommission(0.005).commission(-1000, 50.0) == pytest.approx(5.0)


def test_volume_share_slippage_is_monotonic_and_capped():
    model = VolumeShareSlippage(impact_coefficient=0.1, max_participation=0.25)
    low, high, capped = model.slippage_bps(0.01), model.slippage_bps(0.1), model.slippage_bps(0.9)
    assert low < high < capped
    assert capped == model.slippage_bps(0.25)


def test_cost_model_moves_price_against_the_trade():
    model = CostModel(commission=BpsCommission(0), slippage=FixedSlippage(10))
    assert model.fill_price(10, 100.0, 0.0) == pytest.approx(100.1)
    assert model.fill_price(-10, 100.0, 0.0) == pytest.approx(99.9)
    assert model.participation(10, np.inf) == 0.0
    assert np.isinf(model.participation(10, 0.0))


def test_market_order_fills_at_next_open_with_slippage_and_commission():
    config = EngineConfig(cost_model=CostModel.bps(commission_bps=10, slippage_bps=5))
    order = Order("X", quantity=10)
    fills, still_open, ledger = _fill(order, config)
    assert len(fills) == 1 and not still_open
    fill = fills[0]
    assert fill.reference_price == 100.0
    assert fill.price == pytest.approx(100.05)
    assert fill.commission == pytest.approx(1.0)
    assert fill.slippage_cost == pytest.approx(0.5)
    assert ledger.position("X") == 10
    assert ledger.cash == pytest.approx(100_000 - 10 * 100.05 - 1.0)
    assert order.status is OrderStatus.FILLED


def test_limit_buy_only_fills_when_low_reaches_limit():
    config = EngineConfig()
    unreachable = Order("X", quantity=10, order_type=OrderType.LIMIT, limit_price=90.0)
    fills, still_open, _ = _fill(unreachable, config)
    assert not fills and not still_open
    assert unreachable.status is OrderStatus.CANCELLED and "limit" in unreachable.reason

    reachable = Order("X", quantity=10, order_type=OrderType.LIMIT, limit_price=97.0)
    fills, _, _ = _fill(reachable, config)
    assert fills[0].reference_price == 97.0  # min(open 100, limit 97)

    generous = Order("X", quantity=10, order_type=OrderType.LIMIT, limit_price=103.0)
    fills, _, _ = _fill(generous, config)
    assert fills[0].reference_price == 100.0  # opened below the limit: pay the open


def test_gtc_limit_stays_open_and_expires():
    config = EngineConfig()
    order = Order(
        "X",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=90.0,
        time_in_force=TimeInForce.GTC,
        created_bar=0,
        max_bars=5,
    )
    _, still_open, _ = _fill(order, config)
    assert still_open == [order] and order.status is OrderStatus.PENDING
    expiring = Order(
        "X",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=90.0,
        time_in_force=TimeInForce.GTC,
        created_bar=-4,
        max_bars=5,
    )
    _, still_open, _ = _fill(expiring, config)
    assert (
        not still_open and expiring.status is OrderStatus.CANCELLED and "expired" in expiring.reason
    )


def test_stop_orders_trigger_conservatively():
    config = EngineConfig()
    stop_sell = Order("X", quantity=-10, order_type=OrderType.STOP, stop_price=96.0)
    fills, _, _ = _fill(stop_sell, config)
    assert fills[0].reference_price == 96.0  # min(open 100, stop 96): sold at the stop, not better
    gap_down = Order("X", quantity=-10, order_type=OrderType.STOP, stop_price=101.0)
    fills, _, _ = _fill(gap_down, config)
    assert fills[0].reference_price == 100.0  # opened through the stop: filled at the open
    stop_buy_untriggered = Order("X", quantity=10, order_type=OrderType.STOP, stop_price=110.0)
    fills, _, _ = _fill(stop_buy_untriggered, config)
    assert not fills and stop_buy_untriggered.status is OrderStatus.CANCELLED


def test_no_bar_means_no_fill():
    config = EngineConfig()
    panel = _two_bar_panel()
    bars = panel.to_bars().iloc[:1]  # only the first bar exists
    single = Panel.from_bars(bars, market=US)
    one_more = pd.DataFrame(
        {
            "timestamp": [panel.timestamps[1]],
            "symbol": ["Y"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    both = Panel.from_bars(
        pd.concat([single.to_bars(), one_more]), market=US
    )  # X has no bar on day 2
    order = Order("X", quantity=10)
    ledger = Ledger(both.symbols, config.initial_cash)
    fills, _still_open = FillModel(config).fill_bar([order], both, 1, ledger, both.close[0])
    assert not fills and order.status is OrderStatus.CANCELLED and "no bar" in order.reason


def test_participation_cap_partially_fills_and_keeps_gtc_open():
    config = EngineConfig(max_participation=0.01, fractional_shares=True)
    order = Order("X", quantity=50, time_in_force=TimeInForce.GTC, created_bar=0)
    fills, still_open, _ledger = _fill(order, config)  # volume 1000 -> max 10 units
    assert fills[0].quantity == pytest.approx(10.0)
    assert fills[0].participation == pytest.approx(0.01)
    assert order.status is OrderStatus.PARTIAL and still_open == [order]
    assert order.remaining == pytest.approx(40.0)
    day_order = Order("X", quantity=50)
    _, still_open, _ = _fill(day_order, config)
    assert not still_open and day_order.status is OrderStatus.CANCELLED


def test_leverage_cap_limits_buying_power():
    config = EngineConfig(max_gross_leverage=1.0, initial_cash=10_000.0, fractional_shares=True)
    order = Order("X", quantity=500)  # 50,000 notional against 10,000 equity
    fills, _, ledger = _fill(order, config)
    assert fills[0].quantity == pytest.approx(100.0)
    assert ledger.cash == pytest.approx(0.0)
    assert "leverage" in order.reason
    reducing = Order("X", quantity=-50)
    fills, _, _ = _fill(reducing, config, ledger=ledger)
    assert fills[0].quantity == -50


def test_short_selling_can_be_disabled():
    config = EngineConfig(allow_short=False)
    order = Order("X", quantity=-10)
    fills, _, _ = _fill(order, config)
    assert not fills and order.status is OrderStatus.REJECTED
    target = Order("X", target_weight=-0.5)
    fills, _, _ = _fill(target, config)
    assert (
        not fills and target.status is OrderStatus.FILLED and "already at target" in target.reason
    )


def test_target_weight_order_is_sized_at_execution_equity_and_price():
    config = EngineConfig(fractional_shares=True, cost_model=CostModel.bps(commission_bps=10))
    order = Order("X", target_weight=0.5)
    fills, _, ledger = _fill(order, config)
    assert fills[0].quantity == pytest.approx(0.5 * 100_000 / 100.0)
    assert ledger.position("X") == pytest.approx(500.0)
    assert ledger.cash == pytest.approx(100_000 - 500 * 100.0 - 50.0)


def test_lot_rounding_truncates_toward_zero():
    config = EngineConfig(lot_size=10)
    assert config.round_quantity(27.9) == 20
    assert config.round_quantity(-27.9) == -20
    assert config.round_quantity(9.99) == 0
    assert EngineConfig(fractional_shares=True).round_quantity(27.9) == 27.9


def test_order_validation():
    with pytest.raises(ValueError):
        Order("X", quantity=0)
    with pytest.raises(ValueError):
        Order("X", quantity=1, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        Order("X", target_weight=0.1, order_type=OrderType.LIMIT, limit_price=1.0)
