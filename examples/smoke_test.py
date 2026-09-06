"""Smoke test / usage demo for the current scaffold.

This does NOT run a backtest — VectorizedEngine and EventDrivenEngine don't
exist as concrete classes yet, and StaticScanner/PerturbationTester are still
stubs. This exercises everything that *does* have real logic: the strategy
adapters, BacktestResult, and ValidationReport. Run with:

    python examples/smoke_test.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ubt.core.engines import VectorizedEngine, compare_engines
from ubt.core.interfaces import AssetClass, RuleBasedStrategy
from ubt.core.results import BacktestResult, ValidationReport
from ubt.validation.static_scanner import StaticScanner


class ToyFeed:
    """Minimal DataFeed: wraps a DataFrame indexed by timestamp."""

    asset_class = AssetClass.EQUITY

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def as_of(self, timestamp: datetime) -> pd.DataFrame:
        return self._df.loc[:timestamp]

    def symbols(self) -> list[str]:
        return list(self._df.columns)

    def timestamps(self) -> list[datetime]:
        return list(self._df.index)


def momentum_signal(data: pd.DataFrame) -> dict[str, float]:
    """Toy rule: long anything whose last return was positive."""
    last_return = data.pct_change().iloc[-1]
    return {sym: (1.0 if r > 0 else 0.0) for sym, r in last_return.items()}


def main() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"AAA": [100, 101, 99, 102, 103], "BBB": [50, 49, 49.5, 48, 47]},
        index=dates,
    )
    feed = ToyFeed(prices)
    strategy = RuleBasedStrategy(momentum_signal)

    as_of = dates[-1]
    signals = strategy.on_bar(as_of, feed.as_of(as_of))
    print("1. Strategy adapter")
    print(f"   signals at {as_of.date()}: {signals}")

    equity = pd.Series([1.0, 1.01, 1.02, 1.015, 1.03], index=dates)
    result = BacktestResult(equity_curve=equity, trades=pd.DataFrame(), engine_name="manual", cost_bps=0.0)
    print("2. BacktestResult")
    print(f"   sharpe (toy 5-day series, annualized): {result.sharpe():.3f}")

    result = VectorizedEngine().run(strategy, feed, cost_bps=10.0)
    spread = compare_engines(strategy, feed, cost_bps=10.0)
    print("3. Full backtest (VectorizedEngine, 10bps cost)")
    print(f"   final equity: {result.equity_curve.iloc[-1]:.4f}")
    print(f"   trades: {len(result.trades)}")
    print(f"   vectorized vs event-driven Sharpe spread: {spread:.6f} (~0 by construction, see engines.py)")

    def leaky_signal(data: pd.DataFrame) -> dict[str, float]:
        future = data["AAA"].shift(-1)  # actually leaky, on purpose
        return {"AAA": 1.0 if future.iloc[-1] > future.iloc[-2] else 0.0}

    scan_flags = StaticScanner().run(RuleBasedStrategy(leaky_signal), feed=None)
    print("4. StaticScanner, run against a deliberately leaky signal_fn")
    for f in scan_flags:
        print(f"   [{f.severity}] {f.message}")

    report = ValidationReport(leakage_flags=scan_flags, engine_spread_pct=spread)
    print("5. ValidationReport")
    print(f"   leakage score: {report.leakage_score}")
    print(f"   engine spread: {report.engine_spread_pct:.6f}")


if __name__ == "__main__":
    main()
