"""Performance and overfitting statistics against independent calculations."""

import math

import numpy as np
import pandas as pd
import pytest

from fullbacktester.metrics import (
    compute_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_matches_numpy():
    r = pd.Series([0.01, -0.02, 0.015, 0.003, -0.004])
    expected = r.to_numpy().mean() / r.to_numpy().std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r, 252) == pytest.approx(expected)


def test_sharpe_uses_periodic_risk_free_rate():
    r = pd.Series(np.full(300, (1.05) ** (1 / 252) - 1))  # exactly the risk-free rate every bar
    assert math.isnan(sharpe_ratio(r, 252))  # zero variance
    r2 = r + np.tile([0.001, -0.001], 150)
    assert sharpe_ratio(r2, 252, rf_annual=0.05) == pytest.approx(0.0, abs=1e-9)


def test_undefined_statistics_are_nan_not_zero():
    flat = pd.Series([0.0] * 10)
    assert math.isnan(sharpe_ratio(flat, 252))
    assert math.isnan(sortino_ratio(pd.Series([0.01] * 10), 252))
    metrics = compute_metrics(pd.Series([0.01] * 10), 252)
    assert math.isnan(metrics.calmar) and metrics.max_drawdown == 0.0


def test_max_drawdown_depth_and_duration():
    equity = pd.Series([100, 110, 99, 105, 121, 100, 90, 130])
    depth, bars = max_drawdown(equity)
    assert depth == pytest.approx(90 / 121 - 1)
    assert bars == 2  # 100, 90 below the 121 peak; 130 is a new peak


def test_compute_metrics_totals_and_annualization():
    r = pd.Series([0.01] * 252)
    m = compute_metrics(r, 252, total_costs=100.0, initial_equity=10_000.0)
    assert m.total_return == pytest.approx(1.01**252 - 1)
    assert m.cagr == pytest.approx(m.total_return)
    assert m.cost_drag == pytest.approx(0.01)
    assert m.hit_rate == 1.0


def test_psr_and_dsr_behave():
    assert probabilistic_sharpe_ratio(0.1, 0.0, 500) > 0.95
    assert probabilistic_sharpe_ratio(0.0, 0.0, 500) == pytest.approx(0.5)
    assert deflated_sharpe_ratio(0.1, 500, 1, 0.01) == pytest.approx(
        probabilistic_sharpe_ratio(0.1, 0.0, 500)
    )
    few = deflated_sharpe_ratio(0.1, 500, 2, 0.01)
    many = deflated_sharpe_ratio(0.1, 500, 100, 0.01)
    assert few > many
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(1000, 1.0)


def test_fat_tails_reduce_confidence():
    normal = probabilistic_sharpe_ratio(0.1, 0.0, 300, skew=0.0, excess_kurtosis=0.0)
    fat = probabilistic_sharpe_ratio(0.1, 0.0, 300, skew=-1.0, excess_kurtosis=6.0)
    assert fat < normal


def test_minimum_backtest_length_grows_with_trials_and_shrinks_with_sharpe():
    assert minimum_backtest_length(10, 1.0) < minimum_backtest_length(1000, 1.0)
    assert minimum_backtest_length(100, 2.0) < minimum_backtest_length(100, 1.0)
    assert minimum_backtest_length(1, 1.0) == 0.0
    assert math.isinf(minimum_backtest_length(10, 0.0))


def test_pbo_high_for_noise_and_low_for_real_edge():
    # One dataset's PBO swings widely (its 70 splits share blocks); the estimator's property
    # is the mean across independent datasets: ~0.5 for pure noise, ~0 with a real edge.
    noise_pbo, skill_pbo = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.01, size=(800, 12))
        noise_pbo.append(probability_of_backtest_overfitting(noise, n_blocks=8).pbo)
        skilled = noise.copy()
        skilled[:, 0] += 0.004  # one strategy with genuine drift
        skill_pbo.append(probability_of_backtest_overfitting(skilled, n_blocks=8).pbo)
    assert 0.35 <= np.mean(noise_pbo) <= 0.65
    assert max(skill_pbo) < 0.1
    result = probability_of_backtest_overfitting(noise, n_blocks=8)
    assert result.n_splits == 70
    assert len(result.logits) == 70
    assert np.isfinite(result.performance_degradation)


def test_pbo_input_validation():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 3)), n_blocks=7)
