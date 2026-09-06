"""Strategy contracts.

Three shapes, one engine-facing interface:

* ``Strategy``: imperative. Sees a ``BarContext``, places orders. Runs on the
  event-driven engine only, because its behaviour can depend on fills.
* ``WeightStrategy``: declarative. Returns target weights from a ``PanelView``.
  Runs on both engines; the vectorized engine calls it once per bar.
* ``BatchWeightStrategy``: declarative and batched. Returns the whole weight
  history from one ``PanelView`` in a single call, VectorBT-style. Fastest, and
  the shape most likely to leak the future, which is why the perturbation
  tester exists.

Weights are fractions of equity, signed. ``sum(abs(w))`` is gross leverage.
NaN means "no opinion" and leaves the position untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import pandas as pd

from fullbacktester.data.panel import PanelView
from fullbacktester.strategy.context import BarContext

WeightMap = Mapping[str, float] | pd.Series


class Strategy(ABC):
    """Base class for every strategy shape.

    Attributes:
        warmup: Number of bars to observe before ``on_bar`` is first called.
            Use it for rolling windows so the strategy never sees a truncated one.
    """

    warmup: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> None:
        """Called once per bar with everything known at that bar's close."""

    def reset(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Clear any state accumulated during a run.

        Engines call this before the first bar. Validation re-runs strategies on
        perturbed data, so a strategy that carries state across runs (fitted
        models, caches) must drop it here to be re-runnable.
        """

    def scan_targets(self) -> list[Callable[..., object]]:
        """Callables containing user logic, for the static leakage scanner."""
        return [self.on_bar]


class WeightStrategy(Strategy):
    """Target-weight strategy. Implement ``target_weights``."""

    @abstractmethod
    def target_weights(self, view: PanelView) -> WeightMap:
        """Desired weight per symbol given everything known at ``view.now``."""

    def on_bar(self, ctx: BarContext) -> None:
        ctx.order_target_weights(_as_series(self.target_weights(ctx.view), ctx.symbols))

    def scan_targets(self) -> list[Callable[..., object]]:
        return [self.target_weights]


class BatchWeightStrategy(WeightStrategy):
    """Target weights for every bar of a view in one call. Implement ``target_weights_batch``."""

    @abstractmethod
    def target_weights_batch(self, view: PanelView) -> pd.DataFrame:
        """Rows indexed like ``view.timestamps``, columns are symbols, values are weights.

        Row ``t`` must depend only on bars ``<= t``. Nothing enforces that here;
        ``FutureLeakTester`` checks it behaviourally and the engines' divergence
        exposes it numerically.
        """

    def target_weights(self, view: PanelView) -> WeightMap:
        frame = self.target_weights_batch(view)
        return frame.iloc[-1]

    def scan_targets(self) -> list[Callable[..., object]]:
        return [self.target_weights_batch]


class RuleBasedStrategy(WeightStrategy):
    """Adapter: a plain function ``view -> weights`` becomes a strategy."""

    def __init__(
        self,
        signal_fn: Callable[[PanelView], WeightMap],
        *,
        warmup: int = 0,
        name: str | None = None,
    ) -> None:
        self._signal_fn = signal_fn
        self.warmup = warmup
        self._name = name or str(getattr(signal_fn, "__name__", "RuleBasedStrategy"))

    @property
    def name(self) -> str:
        return self._name

    def target_weights(self, view: PanelView) -> WeightMap:
        return self._signal_fn(view)

    def scan_targets(self) -> list[Callable[..., object]]:
        return [self._signal_fn]


class BatchRuleStrategy(BatchWeightStrategy):
    """Adapter: a plain function ``view -> weight frame`` becomes a batch strategy."""

    def __init__(
        self,
        batch_fn: Callable[[PanelView], pd.DataFrame],
        *,
        warmup: int = 0,
        name: str | None = None,
    ) -> None:
        self._batch_fn = batch_fn
        self.warmup = warmup
        self._name = name or str(getattr(batch_fn, "__name__", "BatchRuleStrategy"))

    @property
    def name(self) -> str:
        return self._name

    def target_weights_batch(self, view: PanelView) -> pd.DataFrame:
        return self._batch_fn(view)

    def scan_targets(self) -> list[Callable[..., object]]:
        return [self._batch_fn]


def _as_series(weights: WeightMap, symbols: tuple[str, ...]) -> pd.Series:
    series = (
        pd.Series(weights, dtype="float64")
        if not isinstance(weights, pd.Series)
        else weights.astype("float64")
    )
    unknown = [s for s in series.index if s not in symbols]
    if unknown:
        raise KeyError(f"weights for symbols not in panel: {unknown}")
    return series
