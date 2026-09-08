"""Paper trading, start to finish, offline.

Simulates a scheduler: a session is created once, then stepped one bar at a
time as if a daily task had woken it up. The last part is the point of the
exercise, and shows the two questions worth asking of a live record.

    python examples/paper_trading.py

For real use, replace the synthetic panel with a real universe and let Task
Scheduler or cron run a script that calls ``PaperSession.open(...).step()``
once per day after the close.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import fullbacktester as fbt


def synthetic_panel(n_bars: int = 260) -> fbt.Panel:
    rng = np.random.default_rng(11)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    sessions = fbt.US.expected_sessions(
        pd.Timestamp("2023-01-03").date(), pd.Timestamp("2024-12-31").date()
    )[:n_bars]
    stamps = pd.DatetimeIndex([fbt.US.session_close_utc(d) for d in sessions])
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (len(stamps), len(symbols))), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, close.shape))
    frame = lambda a: pd.DataFrame(a, index=stamps, columns=symbols)  # noqa: E731
    return fbt.Panel.from_wide(
        frame(close),
        open=frame(open_),
        high=frame(np.maximum(open_, close) * 1.004),
        low=frame(np.minimum(open_, close) * 0.996),
        volume=frame(np.full(close.shape, 5e5)),
        market="US",
    )


def momentum(view: fbt.PanelView) -> pd.Series:
    """Hold the names up over the last quarter, equally weighted."""
    close = view.close
    up = (close.iloc[-1] / close.iloc[-64] - 1.0 > 0).astype(float)
    return up / max(up.sum(), 1.0)


def buy_and_hold(view: fbt.PanelView) -> pd.Series:
    return pd.Series(1.0 / len(view.symbols), index=list(view.symbols))


def main() -> None:
    panel = synthetic_panel()
    config = fbt.EngineConfig(
        initial_cash=500_000,
        cost_model=fbt.CostModel(
            commission=fbt.BpsCommission(3.0),
            slippage=fbt.VolumeShareSlippage(impact_coefficient=0.1),
        ),
        max_participation=0.05,
        allow_short=False,
    )

    def strategies() -> dict[str, fbt.Strategy]:
        # Rebuilt on every "run" the way a scheduled script would import them.
        return {
            "momentum_3m": fbt.RuleBasedStrategy(momentum, warmup=64),
            "buy_and_hold": fbt.RuleBasedStrategy(buy_and_hold, warmup=1),
        }

    workspace = Path(tempfile.mkdtemp(prefix="fbt-paper-"))
    db = workspace / "demo.db"

    print("1. Register the session. Strategies and costs are fingerprinted here.")
    fbt.PaperSession.create(
        db,
        name="demo",
        strategies=strategies(),
        symbols=panel.symbols,
        market="US",
        config=config,
    )
    print(f"   session at {db}")

    print("\n2. Step it, one bar per run, as a daily scheduler would.")
    bars = panel.to_bars()
    for i in range(len(panel)):
        session = fbt.PaperSession.open(db, strategies=strategies(), config=config)
        visible = bars[bars["timestamp"] <= panel.timestamps[i]]
        report = session.step(now=panel.timestamps[i] + pd.Timedelta(hours=1), bars=visible)
        if i in (0, 63, 64, len(panel) - 1):
            print(f"   bar {i:3d}  {report}".replace("\n", "\n        "))

    print("\n3. The live record, read with the same metrics a backtest uses.")
    session = fbt.PaperSession.open(db, strategies=strategies(), config=config)
    results = {name: session.result(name) for name in strategies()}
    table = fbt.live.summary_table(results)
    print(
        table[["cagr", "sharpe", "max_drawdown", "avg_turnover", "cost_drag", "n_fills"]].to_string(
            float_format=lambda v: f"{v:.3f}"
        )
    )

    print("\n4. Replay check: does the live loop reproduce the backtester exactly?")
    for name, strategy in strategies().items():
        check = fbt.replay_check(results[name], strategy, session.panel(), config)
        for flag in check.flags():
            print(f"   {flag}")

    print("\n5. Expectation gap against the backtest that justified trading it.")
    # Here the 'expectation' is a backtest over the same synthetic history. On real
    # data this would be your earlier out-of-sample study.
    for name, strategy in strategies().items():
        expected = fbt.EventDrivenEngine(config).run(strategy, panel)
        gap = fbt.expectation_gap(results[name], expected)
        print(
            f"   {name}: live Sharpe {gap.live.sharpe:.2f} vs backtest "
            f"{gap.expected.sharpe:.2f} (gap {gap.sharpe_gap:+.2f})"
        )
        for flag in gap.flags():
            print(f"      {flag}")

    print("\n6. The lock. Editing a strategy mid-session is refused.")

    def momentum_retuned(view: fbt.PanelView) -> pd.Series:
        close = view.close
        up = (close.iloc[-1] / close.iloc[-20] - 1.0 > 0).astype(float)
        return up / max(up.sum(), 1.0)

    tampered = strategies()
    tampered["momentum_3m"] = fbt.RuleBasedStrategy(momentum_retuned, warmup=64)
    try:
        fbt.PaperSession.open(db, strategies=tampered, config=config)
    except fbt.SessionLockError as exc:
        print(f"   refused: {exc}")


if __name__ == "__main__":
    main()
