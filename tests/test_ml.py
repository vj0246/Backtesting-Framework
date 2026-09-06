"""Walk-forward ML: training may only ever use fully realized labels."""

import numpy as np
import pandas as pd
import pytest

from quantgauntlet.execution import (
    EngineConfig,
    EventDrivenEngine,
    VectorizedEngine,
    compare_engines,
)
from quantgauntlet.strategy.ml import (
    WalkForwardMLStrategy,
    forward_returns,
    long_short_top_k,
    signed_equal_weight,
)
from quantgauntlet.validation import FutureLeakTester


class Ridge:
    """Tiny closed-form ridge regression so the tests need no scikit-learn."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        Xb = np.column_stack([X, np.ones(len(X))])
        penalty = self.alpha * np.eye(Xb.shape[1])
        penalty[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        return np.column_stack([X, np.ones(len(X))]) @ self.coef_


def lagged_return_features(view):
    close = view.close
    frames = {f"ret_{lag}": close.pct_change(lag).shift(0) for lag in (1, 5, 10)}
    stacked = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    return stacked


def _strategy(**overrides):
    params = dict(horizon=5, embargo=2, retrain_every=15, min_train_samples=60, train_window=120)
    params.update(overrides)
    return WalkForwardMLStrategy(lambda: Ridge(alpha=10.0), lagged_return_features, **params)


def test_training_never_uses_labels_that_end_after_the_decision_bar(panel):
    strategy = _strategy()
    result = EventDrivenEngine(EngineConfig.idealized()).run(strategy, panel)
    assert strategy.training_log, "model never trained"
    positions = {ts: i for i, ts in enumerate(panel.timestamps)}
    for record in strategy.training_log:
        fit_bar = positions[record.fitted_at]
        cutoff_bar = positions[record.label_cutoff]
        assert cutoff_bar + strategy.horizon + strategy.embargo <= fit_bar
    assert (result.orders["created_bar"] >= strategy.warmup).all()


def test_label_fn_is_only_asked_for_data_inside_the_view(panel):
    seen = []

    def spy_labels(view, horizon):
        labels = forward_returns(view, horizon)
        seen.append((view.now, labels.dropna().index.get_level_values(0).max()))
        return labels

    strategy = _strategy(label_fn=spy_labels)
    EventDrivenEngine(EngineConfig.idealized()).run(strategy, panel)
    assert seen
    positions = {ts: i for i, ts in enumerate(panel.timestamps)}
    for now, last_label in seen:
        assert positions[last_label] + strategy.horizon <= positions[now]


def test_forward_returns_are_nan_for_the_last_horizon_bars(panel):
    view = panel.view(50)
    labels = forward_returns(view, 5)
    last_defined = labels.dropna().index.get_level_values(0).max()
    assert last_defined == view.timestamps[-6]
    expected = panel.close[10, 0] / panel.close[5, 0] - 1.0
    assert labels.loc[(view.timestamps[5], "S0")] == pytest.approx(expected)


def test_score_to_weight_helpers():
    scores = pd.Series({"A": 0.3, "B": -0.1, "C": 0.05, "D": np.nan})
    equal = signed_equal_weight(scores)
    assert equal.abs().sum() == pytest.approx(1.0) and equal["D"] == 0.0
    top = long_short_top_k(1)(scores)
    assert top["A"] == 0.5 and top["B"] == -0.5 and top["C"] == 0.0
    assert long_short_top_k(3)(scores).abs().sum() == 0.0


def test_walk_forward_runs_on_both_engines_and_passes_perturbation(panel):
    strategy = _strategy()
    comparison = compare_engines(strategy, panel, EngineConfig.idealized())
    assert comparison.agrees()
    assert len(comparison.vectorized.fills) > 0
    flags = FutureLeakTester(n_samples=3).run(strategy, panel)
    assert all(f.severity.value == "info" for f in flags)


def test_reset_clears_state_between_runs(panel):
    strategy = _strategy()
    VectorizedEngine(EngineConfig.idealized()).run(strategy, panel)
    first = list(strategy.training_log)
    VectorizedEngine(EngineConfig.idealized()).run(strategy, panel)
    assert strategy.training_log == first


def test_parameter_validation():
    with pytest.raises(ValueError):
        _strategy(horizon=0)
    with pytest.raises(ValueError):
        _strategy(embargo=-1)
