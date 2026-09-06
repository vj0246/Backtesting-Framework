"""Behavioural look-ahead detection.

Two tests, both built on one fact: a strategy's decision at bar ``t`` must be
a function of bars ``<= t`` only.

**Truncation test** (batch strategies). ``target_weights_batch`` on the full
history must give the same row ``t`` as ``target_weights_batch`` on history
truncated at ``t``. A difference means row ``t`` was computed from later rows:
a centred window, a negative shift, a full-sample normalisation, a model fit
on everything.

**Future-noise test** (every strategy). Run the event-driven engine on the real
panel and on a copy whose bars after ``t`` are replaced by random noise. The
orders placed at bar ``t`` must be identical. Bars up to ``t`` are byte-for-byte
the same in both runs, so the only way to differ is to have read past ``t``.

What neither test can see: a strategy that reads the future from somewhere
other than the panel it was handed (a global DataFrame, a file, a network
call). The static scanner and code review cover that gap; nothing else can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fullbacktester.data.panel import Panel
from fullbacktester.execution.config import EngineConfig
from fullbacktester.execution.event_driven import EventDrivenEngine
from fullbacktester.flags import Flag, Severity
from fullbacktester.strategy.base import BatchWeightStrategy, Strategy

_ORDER_KEYS = ["symbol", "quantity", "target_weight", "order_type", "limit_price", "stop_price"]


@dataclass(frozen=True)
class FutureLeakTester:
    """See module docstring.

    Attributes:
        n_samples: Number of bars ``t`` to test. Each future-noise sample costs
            one full event-driven run.
        seed: For choosing sample bars and generating noise.
        config: Execution settings for the future-noise runs. Idealized by
            default so lot rounding cannot mask a small leak.
        tolerance: Absolute weight difference treated as identical.
    """

    n_samples: int = 6
    seed: int = 0
    config: EngineConfig | None = None
    tolerance: float = 1e-9
    source: str = "perturbation"

    def run(self, strategy: Strategy, panel: Panel) -> list[Flag]:
        ends = self.sample_ends(panel, strategy.warmup)
        if not ends:
            return [
                Flag(
                    source=self.source,
                    severity=Severity.INFO,
                    message="panel too short after warmup to run perturbation tests",
                )
            ]
        flags: list[Flag] = []
        if isinstance(strategy, BatchWeightStrategy):
            flags += self.truncation_test(strategy, panel, ends)
        flags += self.future_noise_test(strategy, panel, ends)
        return flags

    def sample_ends(self, panel: Panel, warmup: int) -> list[int]:
        """View lengths ``e`` (decision bar ``e - 1``) spread across the testable range."""
        lo, hi = warmup + 1, len(panel) - 1
        if hi < lo:
            return []
        rng = np.random.default_rng(self.seed)
        count = min(self.n_samples, hi - lo + 1)
        if count == hi - lo + 1:
            return list(range(lo, hi + 1))
        picks = np.linspace(lo, hi, count + 2)[1:-1]
        jitter = rng.integers(-1, 2, size=count)
        return sorted(
            {int(np.clip(round(p) + j, lo, hi)) for p, j in zip(picks, jitter, strict=True)}
        )

    def truncation_test(
        self, strategy: BatchWeightStrategy, panel: Panel, ends: list[int]
    ) -> list[Flag]:
        strategy.reset()
        full = strategy.target_weights_batch(panel.view(len(panel)))
        full = full.reindex(columns=list(panel.symbols))
        flags: list[Flag] = []
        for end in ends:
            timestamp = panel.timestamps[end - 1]
            strategy.reset()
            truncated = strategy.target_weights_batch(panel.view(end))
            row_trunc = (
                truncated.reindex(columns=list(panel.symbols)).iloc[-1].to_numpy(dtype=float)
            )
            row_full = full.loc[timestamp].to_numpy(dtype=float)
            both_nan = np.isnan(row_full) & np.isnan(row_trunc)
            diff = np.where(both_nan, 0.0, np.abs(row_full - row_trunc))
            diff = np.where(np.isnan(diff), np.inf, diff)
            bad = np.flatnonzero(diff > self.tolerance)
            if bad.size:
                worst = panel.symbols[int(bad[np.argmax(diff[bad])])]
                flags.append(
                    Flag(
                        source=self.source,
                        severity=Severity.HIGH,
                        message=(
                            f"batch weights at {timestamp.date()} change when later bars "
                            f"are removed ({bad.size} symbol(s), largest gap "
                            f"{np.max(diff[bad]):.4g} in {worst}): "
                            "target_weights_batch reads the future"
                        ),
                    )
                )
        if not flags:
            flags.append(
                Flag(
                    source=self.source,
                    severity=Severity.INFO,
                    message=f"truncation test passed at {len(ends)} bars",
                )
            )
        return flags

    def future_noise_test(self, strategy: Strategy, panel: Panel, ends: list[int]) -> list[Flag]:
        config = self.config if self.config is not None else EngineConfig.idealized()
        engine = EventDrivenEngine(config)
        baseline = engine.run(strategy, panel).orders
        rng = np.random.default_rng(self.seed)
        flags: list[Flag] = []
        for end in ends:
            bar = end - 1
            noisy = panel.replace_future(end, rng)
            alternative = engine.run(strategy, noisy).orders
            expected = _orders_at(baseline, bar)
            observed = _orders_at(alternative, bar)
            if not _same_orders(expected, observed, self.tolerance):
                flags.append(
                    Flag(
                        source=self.source,
                        severity=Severity.HIGH,
                        message=(
                            f"orders at {panel.timestamps[bar].date()} change when bars after "
                            "it are replaced with noise: the strategy reads data beyond its view"
                        ),
                    )
                )
        if not flags:
            flags.append(
                Flag(
                    source=self.source,
                    severity=Severity.INFO,
                    message=f"future-noise test passed at {len(ends)} bars",
                )
            )
        return flags


def _orders_at(orders: pd.DataFrame, bar: int) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=_ORDER_KEYS)
    subset = orders.loc[orders["created_bar"] == bar, _ORDER_KEYS].copy()
    # A target-weight order's quantity is resolved at fill time from the (perturbed)
    # execution price; the decision the strategy made is the weight, so compare that.
    subset.loc[subset["target_weight"].notna(), "quantity"] = np.nan
    return subset.sort_values(["symbol", "order_type"]).reset_index(drop=True)


def _same_orders(a: pd.DataFrame, b: pd.DataFrame, tolerance: float) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    if not (a["symbol"].to_numpy() == b["symbol"].to_numpy()).all():
        return False
    if not (a["order_type"].to_numpy() == b["order_type"].to_numpy()).all():
        return False
    for column in ("quantity", "target_weight", "limit_price", "stop_price"):
        x = a[column].to_numpy(dtype=float)
        y = b[column].to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=0.0, atol=tolerance)
        if not np.all(both_nan | close):
            return False
    return True
