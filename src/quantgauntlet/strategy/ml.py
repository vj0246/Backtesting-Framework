"""Walk-forward machine-learning strategy with purge and embargo.

The one place ML backtests leak most is training: a label at time ``s`` is a
forward return over ``[s, s + horizon]``, so a model fit at time ``t`` may only
use samples whose label window has fully closed, ``s + horizon <= t``. The
``embargo`` widens that gap to break serial correlation between the last
training labels and the first prediction. Both are enforced here, and the
training log records the cutoff used at every refit so a test can prove it.

Conventions:
    * ``feature_fn(view)`` returns a frame indexed by ``(timestamp, symbol)``
      with one column per feature, computed from the view only.
    * ``label_fn(view, horizon)`` returns a series with the same index; the
      default is the forward close-to-close return over ``horizon`` bars.
    * ``model_factory()`` returns a fresh object with ``fit(X, y)`` and
      ``predict(X)`` (scikit-learn estimators qualify).
    * ``score_to_weights(scores)`` maps the latest per-symbol predictions to
      target weights; ``signed_equal_weight`` and ``long_short_top_k`` ship.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quantgauntlet.data.panel import PanelView
from quantgauntlet.strategy.base import WeightMap, WeightStrategy


class SupervisedModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


FeatureFn = Callable[[PanelView], pd.DataFrame]
LabelFn = Callable[[PanelView, int], pd.Series]
ScoreToWeights = Callable[[pd.Series], pd.Series]


def forward_returns(view: PanelView, horizon: int) -> pd.Series:
    """close[t + horizon] / close[t] - 1 per symbol, NaN for the last ``horizon`` bars."""
    close = view.close
    future = close.shift(-horizon) / close - 1.0
    stacked = future.stack(future_stack=True)
    assert isinstance(stacked, pd.Series)
    return stacked.rename("label")


def signed_equal_weight(scores: pd.Series) -> pd.Series:
    """Long every positive score, short every negative, equal gross weight of 1."""
    signs = pd.Series(np.sign(scores.fillna(0.0).to_numpy(dtype=float)), index=scores.index)
    n = int((signs != 0).sum())
    return signs / n if n else signs * 0.0


def long_short_top_k(k: int) -> ScoreToWeights:
    """Long the ``k`` highest scores, short the ``k`` lowest, dollar-neutral, gross weight 1."""

    def convert(scores: pd.Series) -> pd.Series:
        clean = scores.dropna()
        weights = pd.Series(0.0, index=scores.index)
        if len(clean) < 2 * k:
            return weights
        ranked = clean.sort_values()
        weights[ranked.index[-k:]] = 0.5 / k
        weights[ranked.index[:k]] = -0.5 / k
        return weights

    return convert


@dataclass(frozen=True)
class TrainingRecord:
    fitted_at: pd.Timestamp
    label_cutoff: pd.Timestamp
    n_samples: int


class WalkForwardMLStrategy(WeightStrategy):
    def __init__(
        self,
        model_factory: Callable[[], SupervisedModel],
        feature_fn: FeatureFn,
        *,
        horizon: int,
        embargo: int = 0,
        retrain_every: int = 20,
        train_window: int | None = None,
        min_train_samples: int = 100,
        label_fn: LabelFn = forward_returns,
        score_to_weights: ScoreToWeights = signed_equal_weight,
        name: str | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1 bar")
        if embargo < 0 or retrain_every < 1 or min_train_samples < 1:
            raise ValueError("embargo >= 0, retrain_every >= 1, min_train_samples >= 1")
        self.model_factory = model_factory
        self.feature_fn = feature_fn
        self.horizon = horizon
        self.embargo = embargo
        self.retrain_every = retrain_every
        self.train_window = train_window
        self.min_train_samples = min_train_samples
        self.label_fn = label_fn
        self.score_to_weights = score_to_weights
        self._name = name or "WalkForwardML"
        self.warmup = horizon + embargo + 1
        self.training_log: list[TrainingRecord] = []
        self._model: SupervisedModel | None = None
        self._last_fit_bar = -1
        self._feature_columns: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self.training_log = []
        self._model = None
        self._last_fit_bar = -1
        self._feature_columns = []

    def scan_targets(self) -> list[Callable[..., object]]:
        return [self.feature_fn, self.label_fn]

    def target_weights(self, view: PanelView) -> WeightMap:
        bar = view.index
        if self._model is None or bar - self._last_fit_bar >= self.retrain_every:
            self._fit(view)
        if self._model is None:
            return pd.Series(np.nan, index=list(view.symbols))
        features = self.feature_fn(view)
        if view.now not in features.index.get_level_values(0):
            return pd.Series(np.nan, index=list(view.symbols))
        current = features.xs(view.now, level=0)
        assert isinstance(current, pd.DataFrame)
        latest = current.reindex(columns=self._feature_columns).dropna()
        if latest.empty:
            return pd.Series(np.nan, index=list(view.symbols))
        predictions = np.asarray(
            self._model.predict(latest.to_numpy(dtype=np.float64)), dtype=float
        )
        scores = pd.Series(predictions.ravel(), index=latest.index.astype(str))
        weights = self.score_to_weights(scores)
        return weights.reindex(list(view.symbols))

    def _fit(self, view: PanelView) -> None:
        cutoff_position = view.index - self.horizon - self.embargo
        if cutoff_position < 0:
            return
        cutoff = view.timestamps[cutoff_position]
        features = self.feature_fn(view)
        labels = self.label_fn(view, self.horizon).rename("label")
        stamps = features.index.get_level_values(0)
        eligible = features[stamps <= cutoff]
        if self.train_window is not None:
            keep_from = view.timestamps[max(0, cutoff_position - self.train_window + 1)]
            eligible = eligible[eligible.index.get_level_values(0) >= keep_from]
        training = eligible.join(labels, how="inner").dropna()
        if len(training) < self.min_train_samples:
            return
        feature_columns = [c for c in training.columns if c != "label"]
        model = self.model_factory()
        model.fit(
            training[feature_columns].to_numpy(dtype=np.float64),
            training["label"].to_numpy(dtype=np.float64),
        )
        self._model = model
        self._feature_columns = feature_columns
        self._last_fit_bar = view.index
        self.training_log.append(
            TrainingRecord(fitted_at=view.now, label_cutoff=cutoff, n_samples=len(training))
        )
