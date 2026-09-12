"""The README's code examples, executed.

Documentation that is not run goes stale within a release or two, and a broken
example is worse than no example. Every snippet in README.md has a test here.
When you change the README, change this file; when this file fails, the README
is lying to somebody.

Panels are kept small so the suite stays fast; the shapes are what matter.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import fullbacktester as fbt


@pytest.fixture(scope="module")
def india_panel() -> fbt.Panel:
    rng = np.random.default_rng(4)
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
    sessions = fbt.INDIA.expected_sessions(date(2022, 1, 1), date(2024, 12, 31))[:150]
    stamps = pd.DatetimeIndex([fbt.INDIA.session_close_utc(d) for d in sessions])
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (len(stamps), len(symbols))), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, close.shape))

    def wide(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=stamps, columns=symbols)

    return fbt.Panel.from_wide(
        wide(close),
        open=wide(open_),
        high=wide(np.maximum(open_, close) * 1.004),
        low=wide(np.minimum(open_, close) * 0.996),
        volume=wide(np.full(close.shape, 2e6)),
        market="IN",
    )


@pytest.fixture(scope="module")
def config() -> fbt.EngineConfig:
    return fbt.EngineConfig(
        initial_cash=1_000_000,
        cost_model=fbt.CostModel(
            commission=fbt.BpsCommission(5.0),
            slippage=fbt.VolumeShareSlippage(impact_coefficient=0.1),
        ),
        max_participation=0.02,
        allow_short=False,
    )


def momentum(view: fbt.PanelView) -> dict[str, float]:
    """README: sixty-second start (lookback shortened to fit the test panel)."""
    returns = view.close.iloc[-1] / view.close.iloc[-60] - 1.0
    winners = returns.nlargest(3).index
    return {symbol: 1 / 3 for symbol in winners}


def test_sixty_second_start(india_panel, config):
    arena = fbt.Arena(india_panel, config, perturbation_samples=2)
    arena.add(fbt.RuleBasedStrategy(momentum, warmup=60), "momentum_6m")
    assert "momentum_6m" in arena.run().summary()


def test_strategy_shape_per_bar_weights(india_panel, config):
    def equal_weight_winners(view):
        signal = view.close.iloc[-1] / view.close.iloc[-21] - 1.0
        winners = signal[signal > 0].index
        if len(winners) == 0:
            return {}  # the README's "hold nothing" branch must be legal
        return {s: 1 / len(winners) for s in winners}

    result = fbt.EventDrivenEngine(config).run(
        fbt.RuleBasedStrategy(equal_weight_winners, warmup=21), india_panel
    )
    assert len(result.equity) == len(india_panel)


def test_strategy_shape_batch_weights(india_panel, config):
    def batch_momentum(view):
        signal = (view.close / view.close.shift(20) - 1.0 > 0).astype(float)
        return signal.div(signal.sum(axis=1).clip(lower=1), axis=0)

    result = fbt.EventDrivenEngine(config).run(
        fbt.BatchRuleStrategy(batch_momentum, warmup=21), india_panel
    )
    assert not result.fills.empty


def test_strategy_shape_orders_with_a_stop(india_panel, config):
    class BreakoutWithStop(fbt.Strategy):
        warmup = 20

        def on_bar(self, ctx):
            for symbol in ctx.symbols:
                bars = ctx.view.bars(symbol)
                if (
                    ctx.position(symbol) == 0
                    and bars["close"].iloc[-1] > bars["high"].iloc[-20:-1].max()
                ):
                    ctx.order_target_weight(symbol, 0.2)
                    ctx.order(
                        symbol,
                        -1,
                        order_type=fbt.OrderType.STOP,
                        stop_price=bars["close"].iloc[-1] * 0.92,
                        time_in_force=fbt.TimeInForce.GTC,
                    )

    result = fbt.EventDrivenEngine(config).run(BreakoutWithStop(), india_panel)
    assert not result.orders.empty


def test_walk_forward_ml_signature():
    class Dummy:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    strategy = fbt.WalkForwardMLStrategy(
        model_factory=Dummy,
        feature_fn=lambda view: pd.DataFrame(),
        horizon=5,
        embargo=2,
        retrain_every=20,
    )
    assert strategy.warmup == 8  # horizon + embargo + 1


def test_local_csv_source(tmp_path, india_panel):
    bars = india_panel.to_bars()
    one = bars[bars["symbol"] == "TCS"].copy()
    one["Date"] = one["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    one = one.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    one[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
        tmp_path / "TCS.csv", index=False
    )
    panel = fbt.load_panel(
        ["TCS"],
        "2022-01-01",
        "2024-12-31",
        market="IN",
        source=fbt.LocalSource(tmp_path),
        cache=False,
    )
    assert len(panel) == len(india_panel)


def test_data_quality_and_nse_surface(india_panel):
    assert isinstance(fbt.check_data_quality(india_panel), list)
    source = fbt.NSEBhavcopySource()
    assert source.supports(fbt.INDIA, fbt.Frequency.DAY_1)
    assert hasattr(source, "list_symbols")


def test_cost_sensitivity_loop(india_panel):
    strategy = fbt.RuleBasedStrategy(momentum, warmup=60)
    sharpes = []
    for bps in (0, 5, 25):
        result = fbt.EventDrivenEngine(
            fbt.EngineConfig(cost_model=fbt.CostModel.bps(bps, bps // 2))
        ).run(strategy, india_panel)
        sharpes.append(result.metrics().sharpe)
    assert np.all(np.isfinite(sharpes))
    assert sharpes[0] >= sharpes[-1]  # costs cannot help


def test_engine_comparison(india_panel, config):
    strategy = fbt.RuleBasedStrategy(momentum, warmup=60)
    assert fbt.compare_engines(strategy, india_panel, fbt.EngineConfig.idealized()).agrees()
    comparison = fbt.compare_engines(strategy, india_panel, config)
    assert np.isfinite(comparison.max_equity_gap)
    comparison.flags()


def test_validate_and_the_leak_detector(india_panel):
    clean = fbt.validate(fbt.RuleBasedStrategy(momentum, warmup=60), india_panel, n_samples=2)
    assert "validation score" in clean.summary()
    assert not clean.has_high

    def oracle(view):
        forward = view.close.shift(-1) / view.close - 1.0
        signal = (forward > 0).astype(float)
        return signal.div(signal.sum(axis=1).clip(lower=1), axis=0)

    leaky = fbt.validate(fbt.BatchRuleStrategy(oracle, warmup=1), india_panel, n_samples=2)
    assert leaky.has_high, "the README claims this leak is caught"


def test_arena_with_selection_aware_statistics(india_panel, config):
    def reversal(view):
        recent = view.close.iloc[-1] / view.close.iloc[-6] - 1.0
        return {s: 0.5 for s in recent.nsmallest(2).index}

    def equal_weight(view):
        return {s: 1 / len(view.symbols) for s in view.symbols}

    result = (
        fbt.Arena(india_panel, config, perturbation_samples=2)
        .add(fbt.RuleBasedStrategy(momentum, warmup=60), "momentum")
        .add(fbt.RuleBasedStrategy(reversal, warmup=6), "reversal")
        .add(fbt.RuleBasedStrategy(equal_weight, warmup=1), "benchmark")
        .run()
    )
    assert 0.0 <= result.pbo(n_blocks=8).pbo <= 1.0
    assert result.best("deflated_sharpe") in {"momentum", "reversal", "benchmark"}
    assert {"deflated_sharpe", "probabilistic_sharpe"} <= set(result.table.columns)


def test_paper_trading_cycle(tmp_path, india_panel, config):
    db = tmp_path / "momentum.db"

    def strategies():
        return {"momentum_6m": fbt.RuleBasedStrategy(momentum, warmup=60)}

    fbt.PaperSession.create(
        db,
        name="momentum",
        strategies=strategies(),
        symbols=india_panel.symbols,
        market="IN",
        config=config,
    )
    bars = india_panel.to_bars()
    for i in range(len(india_panel)):
        session = fbt.PaperSession.open(db, strategies=strategies(), config=config)
        visible = bars[bars["timestamp"] <= india_panel.timestamps[i]]
        session.step(now=india_panel.timestamps[i] + pd.Timedelta(hours=1), bars=visible)

    session = fbt.PaperSession.open(db, strategies=strategies(), config=config)
    result = session.result("momentum_6m")
    strategy = strategies()["momentum_6m"]

    assert fbt.replay_check(result, strategy, session.panel(), config).agrees()

    expected = fbt.EventDrivenEngine(config).run(strategy, india_panel)
    assert np.isfinite(fbt.expectation_gap(result, expected).sharpe_gap)

    with pytest.raises(fbt.SessionLockError):
        fbt.PaperSession.open(
            db,
            strategies={"momentum_6m": fbt.RuleBasedStrategy(lambda v: {}, warmup=60)},
            config=config,
        )


def test_metrics_reference_is_complete(india_panel, config):
    result = fbt.EventDrivenEngine(config).run(
        fbt.RuleBasedStrategy(momentum, warmup=60), india_panel
    )
    metrics = result.metrics()
    for name in (
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "max_drawdown_bars",
        "hit_rate",
        "best_bar",
        "worst_bar",
        "skew",
        "excess_kurtosis",
        "avg_turnover",
        "avg_gross_exposure",
        "n_fills",
        "total_costs",
        "cost_drag",
    ):
        assert hasattr(metrics, name), f"README documents {name}, which does not exist"
    # The README claims annualization follows the market, not a hard-coded 252.
    assert metrics.bars_per_year == 250.0

    for name in (
        "probabilistic_sharpe_ratio",
        "deflated_sharpe_ratio",
        "minimum_backtest_length",
        "probability_of_backtest_overfitting",
    ):
        assert hasattr(fbt.metrics, name), f"README documents fbt.metrics.{name}"


def test_broker_feeds(tmp_path, monkeypatch):
    """README: broker feeds. Offline: the HTTP layer is replaced, never the source."""
    from fullbacktester.data.sources import alpaca as alpaca_module
    from fullbacktester.data.sources import upstox as upstox_module
    from tests.test_brokers import FakeSession, alpaca_route, upstox_route

    # Keep the fake instrument master out of the user's real cache directory.
    monkeypatch.setattr(upstox_module, "default_cache_root", lambda: tmp_path)
    monkeypatch.setattr(upstox_module, "default_session", lambda extra: FakeSession(upstox_route))
    monkeypatch.setattr(alpaca_module, "default_session", lambda extra: FakeSession(alpaca_route))
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "readme-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "readme-secret")

    india = fbt.load_panel(
        ["RELIANCE", "TCS"], "2015-01-01", "2024-12-31", market="IN", source="upstox", cache=False
    )
    usa = fbt.load_panel(
        ["AAPL", "BRK-B"], "2015-01-01", "2024-12-31", market="US", source="alpaca", cache=False
    )
    assert india.symbols == ("RELIANCE", "TCS")
    assert usa.symbols == ("AAPL", "BRK-B")
