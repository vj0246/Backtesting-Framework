"""End-to-end demo on synthetic data (runs offline).

Builds a random-walk panel on the NYSE clock, defines three strategies including a
deliberately leaky one, and runs them through the arena. Swap the panel for
``fbt.load_panel([...], "2018-01-01", "2024-12-31", market="US")`` for real data.

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fullbacktester as fbt


def synthetic_panel(
    n_bars: int = 750, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")
) -> fbt.Panel:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=n_bars)
    stamps = pd.DatetimeIndex([fbt.US.session_close_utc(d) for d in dates])
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, (n_bars, len(symbols))), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = np.full(close.shape, 2_000_000.0)
    frame = lambda a: pd.DataFrame(a, index=stamps, columns=list(symbols))  # noqa: E731
    return fbt.Panel.from_wide(
        frame(close),
        open=frame(open_),
        high=frame(high),
        low=frame(low),
        volume=frame(volume),
        market="US",
    )


def momentum(view: fbt.PanelView) -> pd.Series:
    """Long the names that rose over the last six months, equal weight."""
    close = view.close
    up = (close.iloc[-1] / close.iloc[-126] - 1.0 > 0).astype(float)
    return up / max(up.sum(), 1.0)


def mean_reversion(view: fbt.PanelView) -> pd.DataFrame:
    """Batch form: fade last week's move. Row t uses only bars <= t."""
    close = view.close
    weekly = close / close.shift(5) - 1.0
    signal = -np.sign(weekly).fillna(0.0)
    return signal.div(signal.abs().sum(axis=1).clip(lower=1.0), axis=0)


def oracle(view: fbt.PanelView) -> pd.DataFrame:
    """Deliberately leaky: buys whatever goes up tomorrow. The gauntlet should catch this."""
    close = view.close
    tomorrow = close.shift(-1) / close - 1.0
    signal = (tomorrow > 0).astype(float)
    return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)


def main() -> None:
    panel = synthetic_panel()
    config = fbt.EngineConfig(
        initial_cash=1_000_000,
        cost_model=fbt.CostModel(
            commission=fbt.BpsCommission(1.0),
            slippage=fbt.VolumeShareSlippage(impact_coefficient=0.1),
        ),
        max_participation=0.05,
        max_gross_leverage=1.0,
    )
    arena = fbt.Arena(panel, config, perturbation_samples=4)
    arena.add(fbt.RuleBasedStrategy(momentum, warmup=126), "momentum_6m")
    arena.add(fbt.BatchRuleStrategy(mean_reversion, warmup=5), "mean_reversion_1w")
    arena.add(fbt.BatchRuleStrategy(oracle, warmup=1), "oracle_LEAKY")
    result = arena.run()

    print(result.summary())
    print()
    cscv = result.pbo(n_blocks=8)
    print(f"probability of backtest overfitting across {result.n_trials} entries: {cscv.pbo:.2f}")
    best = result.best("deflated_sharpe")
    print(f"best by deflated Sharpe: {best}")
    print()
    print(result.entries[best].validation.summary())


if __name__ == "__main__":
    main()
