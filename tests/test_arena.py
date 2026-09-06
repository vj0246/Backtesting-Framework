"""Arena: same gauntlet for every entry, selection-aware statistics."""

import numpy as np
import pandas as pd
import pytest

from fullbacktester.arena import Arena
from fullbacktester.execution import CostModel, EngineConfig
from fullbacktester.strategy import BatchRuleStrategy, RuleBasedStrategy, Strategy


def momentum(view):
    close = view.close
    signal = (close.iloc[-1] / close.iloc[-21] - 1.0 > 0).astype(float)
    return signal / max(signal.sum(), 1.0)


def reversal(view):
    close = view.close
    signal = (close.iloc[-1] / close.iloc[-6] - 1.0 < 0).astype(float)
    return signal / max(signal.sum(), 1.0)


def oracle(view):
    close = view.close
    signal = (close.shift(-1) / close - 1.0 > 0).astype(float)
    return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)


class BuyAndHold(Strategy):
    def on_bar(self, ctx):
        if ctx.bar == 0:
            ctx.order_target_weights({s: 1.0 / len(ctx.symbols) for s in ctx.symbols})


def test_arena_ranks_and_flags(panel):
    arena = Arena(panel, EngineConfig(cost_model=CostModel.bps(5, 5)), perturbation_samples=3)
    arena.add(RuleBasedStrategy(momentum, warmup=21), "momentum")
    arena.add(RuleBasedStrategy(reversal, warmup=6), "reversal")
    arena.add(BatchRuleStrategy(oracle, warmup=1), "oracle")
    arena.add(BuyAndHold())
    result = arena.run()

    table = result.table
    assert set(table.index) == {"momentum", "reversal", "oracle", "BuyAndHold"}
    assert result.n_trials == 4
    assert table.loc["oracle", "high_flags"] > 0
    assert table.loc["momentum", "high_flags"] == 0
    # The oracle leaks only where leaking is possible: the vectorized run sees the future,
    # the event-driven run (the one the table reports) is structurally blind to it.
    oracle_entry = result.entries["oracle"]
    assert oracle_entry.vectorized is not None
    assert oracle_entry.vectorized.metrics().sharpe > 5.0
    assert oracle_entry.event_driven.fills.empty
    assert table.loc["oracle", "engine_equity_gap"] > 0.5
    assert (
        table["deflated_sharpe"].dropna() <= table["probabilistic_sharpe"].dropna() + 1e-12
    ).all()
    assert np.isnan(table.loc["BuyAndHold", "engine_sharpe_gap"])  # no vectorized twin
    assert np.isfinite(table.loc["momentum", "engine_sharpe_gap"])
    assert result.best("sharpe") in {"momentum", "reversal", "BuyAndHold"}
    assert "HIGH-severity" in result.summary()
    cscv = result.pbo(n_blocks=8)
    assert 0.0 <= cscv.pbo <= 1.0


def test_arena_rejects_duplicates_and_empty(panel):
    arena = Arena(panel)
    arena.add(BuyAndHold())
    with pytest.raises(ValueError, match="duplicate"):
        arena.add(BuyAndHold())
    with pytest.raises(ValueError, match="no strategies"):
        Arena(panel).run()


def test_arena_without_validation_is_faster_and_still_ranks(panel):
    result = Arena(panel, validate=False).add(RuleBasedStrategy(momentum, warmup=21)).run()
    assert list(result.table.index) == ["momentum"]
    assert result.entries["momentum"].validation.flags == [] or all(
        f.source != "perturbation" for f in result.entries["momentum"].validation.flags
    )
    assert isinstance(result.returns_matrix(), pd.DataFrame)
