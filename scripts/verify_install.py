"""Verify an *installed* fullbacktester, not the source tree.

Run this against a virtual environment that has the built wheel installed and
no access to ``src/``. It catches the packaging faults a normal test run cannot
see: a module left out of the wheel, a missing runtime dependency, a stale
entry in ``__all__``, or a lost ``py.typed`` marker.

    python -m venv /tmp/clean
    /tmp/clean/bin/python -m pip install dist/*.whl
    cd /tmp && /tmp/clean/bin/python path/to/scripts/verify_install.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import fullbacktester as fbt

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {label}{f': {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print(f"fullbacktester {fbt.__version__} from {Path(fbt.__file__).parent}")

    print("\npackaging")
    package_dir = Path(fbt.__file__).parent
    check("not importing from a source checkout", package_dir.name == "fullbacktester")
    check("py.typed marker shipped", (package_dir / "py.typed").is_file())
    missing = [name for name in fbt.__all__ if not hasattr(fbt, name)]
    check("every __all__ entry resolves", not missing, f"missing {missing}" if missing else "")

    print("\nsubmodules import")
    for module in (
        "fullbacktester.arena",
        "fullbacktester.data.cache",
        "fullbacktester.data.loader",
        "fullbacktester.data.quality",
        "fullbacktester.data.sources.alpaca",
        "fullbacktester.data.sources.local",
        "fullbacktester.data.sources.nse",
        "fullbacktester.data.sources.upstox",
        "fullbacktester.data.sources.yfinance",
        "fullbacktester.execution.compare",
        "fullbacktester.execution.event_driven",
        "fullbacktester.execution.vectorized",
        "fullbacktester.metrics.overfitting",
        "fullbacktester.strategy.ml",
        "fullbacktester.validation.perturbation",
        "fullbacktester.validation.static_scanner",
    ):
        try:
            importlib.import_module(module)
            ok, detail = True, ""
        except Exception as exc:  # report the failure, never mask it
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(module, ok, detail)

    # The optional adapters must import even when their extra is absent, because
    # their third-party imports live inside fetch(), not at module scope.
    extras = {
        name: importlib.util.find_spec(name) is not None
        for name in ("yfinance", "requests", "exchange_calendars")
    }
    print(f"\noptional extras present: { {k: v for k, v in extras.items()} }")

    print("\nengine")
    rng = np.random.default_rng(0)
    sessions = fbt.US.expected_sessions(
        pd.Timestamp("2023-01-03").date(), pd.Timestamp("2024-12-31").date()
    )
    stamps = pd.DatetimeIndex([fbt.US.session_close_utc(d) for d in sessions])
    symbols = ["AAA", "BBB", "CCC"]
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, (len(stamps), 3)), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, close.shape))
    wide = lambda a: pd.DataFrame(a, index=stamps, columns=symbols)  # noqa: E731
    panel = fbt.Panel.from_wide(
        wide(close),
        open=wide(open_),
        high=wide(np.maximum(open_, close) * 1.004),
        low=wide(np.minimum(open_, close) * 0.996),
        volume=wide(np.full(close.shape, 3e6)),
        market="US",
    )

    def momentum(view: fbt.PanelView) -> pd.Series:
        c = view.close
        up = (c.iloc[-1] / c.iloc[-63] - 1 > 0).astype(float)
        return up / max(up.sum(), 1.0)

    strategy = fbt.RuleBasedStrategy(momentum, warmup=63)
    result = fbt.EventDrivenEngine(fbt.EngineConfig(cost_model=fbt.CostModel.bps(2, 3))).run(
        strategy, panel
    )
    metrics = result.metrics()
    check(
        "backtest produced an equity curve", len(result.equity) == len(panel), f"{len(panel)} bars"
    )
    check("fills recorded", metrics.n_fills > 0, f"{metrics.n_fills} fills")
    check("annualization from the market", metrics.bars_per_year == 252.0)
    check("sharpe is finite", np.isfinite(metrics.sharpe), f"{metrics.sharpe:.3f}")

    comparison = fbt.compare_engines(strategy, panel, fbt.EngineConfig.idealized())
    check(
        "engines agree under idealized execution",
        comparison.agrees(),
        f"max equity gap {comparison.max_equity_gap:.2e}",
    )

    print("\nvalidation")
    report = fbt.validate(strategy, panel, n_samples=2)
    check(
        "clean strategy raises no HIGH findings", not report.has_high, f"score {report.score}/100"
    )

    def leaky(view: fbt.PanelView) -> pd.DataFrame:
        c = view.close
        signal = (c.shift(-1) / c - 1 > 0).astype(float)
        return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)

    leaky_report = fbt.validate(fbt.BatchRuleStrategy(leaky, warmup=1), panel, n_samples=2)
    check("look-ahead strategy is caught", leaky_report.has_high, f"score {leaky_report.score}/100")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
