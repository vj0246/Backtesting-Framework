"""Paper trading: durability, the pre-registration lock, and replay equivalence."""

import numpy as np
import pandas as pd
import pytest

from fullbacktester.data.sources.base import DataSource
from fullbacktester.execution import CostModel, EngineConfig, EventDrivenEngine
from fullbacktester.flags import Severity
from fullbacktester.live import (
    LiveStore,
    PaperSession,
    SessionLockError,
    replay_check,
    strategy_fingerprint,
    summary_table,
)
from fullbacktester.markets import US, Frequency
from fullbacktester.strategy import RuleBasedStrategy, Strategy


class ReplaySource(DataSource):
    """Serves slices of a fixed panel, so a 'live' feed is deterministic."""

    name = "replay"
    markets = frozenset({"US", "IN", "CRYPTO"})
    frequencies = frozenset(Frequency)
    adjusted = True
    survivorship_free = False

    def __init__(self, panel):
        self.bars = panel.to_bars()

    def fetch(self, symbols, start, end, frequency, market):
        lo = market.session_open_utc(start)
        hi = market.session_close_utc(end)
        rows = self.bars
        return rows[(rows["timestamp"] >= lo) & (rows["timestamp"] <= hi)].reset_index(drop=True)


def momentum(view):
    close = view.close
    up = (close.iloc[-1] / close.iloc[-11] - 1.0 > 0).astype(float)
    return up / max(up.sum(), 1.0)


def flat(view):
    return {s: 0.0 for s in view.symbols}


def _strategy():
    return RuleBasedStrategy(momentum, warmup=11)


def _config():
    return EngineConfig(
        initial_cash=100_000, cost_model=CostModel.bps(5, 5), fractional_shares=True
    )


def _fresh_interpreter():
    """Reset the process-local order-id counter, as starting python again would.

    Paper trading spans processes by design, so any test that claims to check
    resumption has to reproduce this or it is checking nothing.
    """
    from fullbacktester.execution import orders as orders_module

    orders_module._last_order_id = 0


def _feed(panel, upto):
    """Bars up to and including index ``upto``, as a source would return them."""
    bars = panel.to_bars()
    return bars[bars["timestamp"] <= panel.timestamps[upto]].reset_index(drop=True)


# ---------------------------------------------------------------- lifecycle


def test_session_round_trips_through_disk(tmp_path, panel_factory):
    panel = panel_factory(n_bars=40)
    path = tmp_path / "s.db"
    PaperSession.create(
        path,
        name="test",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
        source=ReplaySource(panel),
    )
    reopened = PaperSession.open(
        path, strategies={"mom": _strategy()}, config=_config(), source=ReplaySource(panel)
    )
    assert reopened.spec.name == "test"
    assert reopened.spec.symbols == panel.symbols
    assert reopened.spec.market == "US"


def test_create_refuses_to_overwrite(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    path = tmp_path / "s.db"
    kwargs = dict(
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    PaperSession.create(path, **kwargs)
    with pytest.raises(FileExistsError):
        PaperSession.create(path, **kwargs)


def test_open_missing_session_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PaperSession.open(tmp_path / "nope.db", strategies={"m": _strategy()}, config=_config())


# --------------------------------------------------------- pre-registration


def test_editing_the_strategy_locks_the_session(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    path = tmp_path / "s.db"
    PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )

    def momentum_tweaked(view):  # same shape, different lookback: a silent edit
        close = view.close
        up = (close.iloc[-1] / close.iloc[-5] - 1.0 > 0).astype(float)
        return up / max(up.sum(), 1.0)

    with pytest.raises(SessionLockError, match="has changed since it was registered"):
        PaperSession.open(
            path,
            strategies={"mom": RuleBasedStrategy(momentum_tweaked, warmup=11)},
            config=_config(),
        )


def test_softening_costs_locks_the_session(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    path = tmp_path / "s.db"
    PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    cheaper = EngineConfig(
        initial_cash=100_000, cost_model=CostModel.bps(0, 0), fractional_shares=True
    )
    with pytest.raises(SessionLockError, match="execution config differs"):
        PaperSession.open(path, strategies={"mom": _strategy()}, config=cheaper)


def test_changing_the_strategy_set_locks_the_session(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    path = tmp_path / "s.db"
    PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    with pytest.raises(SessionLockError, match="strategy set differs"):
        PaperSession.open(
            path,
            strategies={"mom": _strategy(), "flat": RuleBasedStrategy(flat)},
            config=_config(),
        )


def test_fingerprint_is_stable_and_sensitive():
    a = strategy_fingerprint(_strategy())
    b = strategy_fingerprint(_strategy())
    assert a == b
    assert strategy_fingerprint(RuleBasedStrategy(flat)) != a
    # warmup is part of the identity: same code, different warmup is a different experiment
    assert strategy_fingerprint(RuleBasedStrategy(momentum, warmup=5)) != a


# --------------------------------------------------------- point-in-time bars


def test_recorded_bars_are_never_rewritten(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    store = LiveStore(tmp_path / "s.db")
    from fullbacktester.live.store import SessionSpec

    store.initialise(
        SessionSpec(
            name="t",
            market="US",
            frequency=Frequency.DAY_1,
            symbols=panel.symbols,
            source=None,
            config_hash="x",
            created_at=pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        [],
    )
    bars = panel.to_bars()
    added, revised = store.record_bars(bars, pd.Timestamp("2024-01-01", tz="UTC"))
    assert added == len(bars) and revised == 0

    restated = bars.copy()
    restated.loc[0, "close"] = restated.loc[0, "close"] * 1.5
    added2, revised2 = store.record_bars(restated, pd.Timestamp("2024-01-02", tz="UTC"))
    assert added2 == 0 and revised2 == 1

    kept = store.observed_bars()
    assert kept.loc[0, "close"] == pytest.approx(bars.loc[0, "close"])
    log = store.revisions()
    assert len(log) == 1 and log.loc[0, "field"] == "close"


def test_revision_raises_a_warn_flag(tmp_path, panel_factory):
    panel = panel_factory(n_bars=20)
    path = tmp_path / "s.db"
    session = PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    session.step(now=pd.Timestamp("2024-01-01", tz="UTC"), bars=panel.to_bars())
    restated = panel.to_bars()
    restated.loc[0, "close"] *= 1.2
    report = session.step(now=pd.Timestamp("2024-01-02", tz="UTC"), bars=restated)
    assert report.bars_revised == 1
    assert any(f.severity is Severity.WARN and "restate" not in f.message for f in report.flags)


# ------------------------------------------------------------------- stepping


def test_stepping_bar_by_bar_matches_a_backtest_exactly(tmp_path, panel_factory):
    """The point of the whole design: live and backtest must be the same computation."""
    panel = panel_factory(n_bars=60, seed=3)
    path = tmp_path / "s.db"
    session = PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    # Feed one bar at a time, exactly as a daily scheduler would.
    for i in range(len(panel)):
        session.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    live = session.result("mom")
    check = replay_check(live, _strategy(), session.panel(), _config())
    assert check.agrees(), f"max gap {check.max_equity_gap}"
    assert check.n_bars == len(panel)
    assert all(f.severity is Severity.INFO for f in check.flags())

    backtest = EventDrivenEngine(_config()).run(_strategy(), panel)
    np.testing.assert_allclose(live.equity.to_numpy(), backtest.equity.to_numpy(), rtol=1e-12)


def test_catching_up_after_missed_runs_matches_bar_by_bar(tmp_path, panel_factory):
    """Three days offline then one run must equal three daily runs."""
    panel = panel_factory(n_bars=40, seed=5)

    daily = PaperSession.create(
        tmp_path / "daily.db",
        name="daily",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    for i in range(len(panel)):
        daily.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    lumpy = PaperSession.create(
        tmp_path / "lumpy.db",
        name="lumpy",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    for i in (9, 19, 29, len(panel) - 1):
        lumpy.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    pd.testing.assert_series_equal(daily.result("mom").equity, lumpy.result("mom").equity)


def test_step_is_idempotent_when_nothing_new_arrives(tmp_path, panel_factory):
    panel = panel_factory(n_bars=25)
    session = PaperSession.create(
        tmp_path / "s.db",
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    first = session.step(now=pd.Timestamp("2024-06-01", tz="UTC"), bars=panel.to_bars())
    equity_after_first = session.result("mom").equity.copy()

    again = session.step(now=pd.Timestamp("2024-06-02", tz="UTC"), bars=panel.to_bars())
    assert first.bars_processed == len(panel)
    assert again.bars_processed == 0 and again.bars_added == 0
    pd.testing.assert_series_equal(session.result("mom").equity, equity_after_first)


def test_state_survives_a_new_process(tmp_path, panel_factory):
    """Reopening from disk must continue, not restart."""
    panel = panel_factory(n_bars=40, seed=7)
    path = tmp_path / "s.db"
    first = PaperSession.create(
        path,
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    for i in range(20):
        first.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))
    del first

    _fresh_interpreter()
    resumed = PaperSession.open(path, strategies={"mom": _strategy()}, config=_config())
    for i in range(20, len(panel)):
        resumed.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    backtest = EventDrivenEngine(_config()).run(_strategy(), panel)
    np.testing.assert_allclose(
        resumed.result("mom").equity.to_numpy(), backtest.equity.to_numpy(), rtol=1e-12
    )


def test_replay_cap_flags_a_hole_rather_than_hiding_it(tmp_path, panel_factory):
    panel = panel_factory(n_bars=40)
    session = PaperSession.create(
        tmp_path / "s.db",
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    session.max_replay_bars = 5
    report = session.step(now=pd.Timestamp("2024-06-01", tz="UTC"), bars=panel.to_bars())
    assert report.bars_processed == 5
    assert any("skipped" in f.message and f.severity is Severity.WARN for f in report.flags)


# ------------------------------------------------------------- multi-strategy


def test_several_strategies_share_one_bar_stream(tmp_path, panel_factory):
    panel = panel_factory(n_bars=45, seed=11)
    session = PaperSession.create(
        tmp_path / "s.db",
        name="arena",
        strategies={"mom": _strategy(), "flat": RuleBasedStrategy(flat)},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    for i in range(len(panel)):
        session.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    results = {label: session.result(label) for label in ("mom", "flat")}
    # Flat never trades, so its equity is exactly the starting cash throughout.
    assert results["flat"].fills.empty
    assert np.allclose(results["flat"].equity.to_numpy(), 100_000.0)
    assert not results["mom"].fills.empty

    table = summary_table(results)
    assert set(table.index) == {"mom", "flat"}
    assert "sharpe" in table.columns

    for label in ("mom", "flat"):
        check = replay_check(results[label], session.strategies[label], session.panel(), _config())
        assert check.agrees()


def test_imperative_strategy_orders_persist_across_runs(tmp_path, panel_factory):
    """A GTC limit placed on one run must still be pending on the next."""
    panel = panel_factory(n_bars=30, seed=13)

    class LateLimit(Strategy):
        warmup = 0

        def on_bar(self, ctx):
            if ctx.bar == 2:
                from fullbacktester.execution.orders import OrderType, TimeInForce

                ctx.order(
                    "S0",
                    10,
                    order_type=OrderType.LIMIT,
                    limit_price=ctx.price("S0") * 0.5,
                    time_in_force=TimeInForce.GTC,
                )

    path = tmp_path / "s.db"
    session = PaperSession.create(
        path,
        name="t",
        strategies={"lim": LateLimit()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    for i in range(5):
        session.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))

    _, _, pending = session.store.load_state("lim")
    assert len(pending) == 1

    resumed = PaperSession.open(path, strategies={"lim": LateLimit()}, config=_config())
    for i in range(5, len(panel)):
        resumed.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))
    backtest = EventDrivenEngine(_config()).run(LateLimit(), panel)
    np.testing.assert_allclose(
        resumed.result("lim").equity.to_numpy(), backtest.equity.to_numpy(), rtol=1e-12
    )


# ------------------------------------------------------------------- sources


def test_fetch_goes_through_the_configured_source(tmp_path, panel_factory):
    panel = panel_factory(n_bars=30)
    session = PaperSession.create(
        tmp_path / "s.db",
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
        source=ReplaySource(panel),
    )
    end = panel.timestamps[-1] + pd.Timedelta(hours=1)
    report = session.step(now=end)
    assert report.bars_added > 0
    assert not session.result("mom").equity.empty


def test_result_before_any_step_is_an_error(tmp_path, panel_factory):
    panel = panel_factory(n_bars=10)
    session = PaperSession.create(
        tmp_path / "s.db",
        name="t",
        strategies={"mom": _strategy()},
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )
    with pytest.raises(ValueError, match="no recorded equity"):
        session.result("mom")


def test_resuming_in_a_fresh_process_keeps_the_whole_blotter(tmp_path, panel_factory):
    """Regression: order ids restart at 1 in a new interpreter.

    They are the primary key in the orders table, so without reserving past the
    ids already stored, the second run's orders overwrite the first run's. The
    bug lost 18 of 58 orders and left the equity curve intact, so nothing else
    would have noticed.
    """
    panel = panel_factory(n_bars=30, seed=17)
    path = tmp_path / "s.db"

    def always_in(view):
        return {s: 1 / len(view.symbols) for s in view.symbols}

    def strategies():
        return {"hold": RuleBasedStrategy(always_in, warmup=1)}

    PaperSession.create(
        path,
        name="t",
        strategies=strategies(),
        symbols=panel.symbols,
        market=US,
        config=_config(),
    )

    first = PaperSession.open(path, strategies=strategies(), config=_config())
    for i in range(10):
        first.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))
    placed_first = len(first.store.orders_frame("hold"))
    assert placed_first > 0

    _fresh_interpreter()
    second = PaperSession.open(path, strategies=strategies(), config=_config())
    placed_second = 0
    for i in range(10, len(panel)):
        report = second.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=_feed(panel, i))
        placed_second += report.orders_placed["hold"]

    blotter = second.store.orders_frame("hold")
    assert len(blotter) == placed_first + placed_second
    assert blotter["id"].is_unique

    backtest = EventDrivenEngine(_config()).run(RuleBasedStrategy(always_in, warmup=1), panel)
    np.testing.assert_allclose(
        second.result("hold").equity.to_numpy(), backtest.equity.to_numpy(), rtol=1e-12
    )
