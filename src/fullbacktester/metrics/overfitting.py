"""Statistics that account for how many things you tried.

* ``probabilistic_sharpe_ratio``: probability that the true Sharpe exceeds a
  benchmark given the sample length and the return distribution's shape
  (Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier", 2012).
* ``deflated_sharpe_ratio``: the same test with the benchmark raised to the
  Sharpe you would expect the *best* of N unskilled trials to show
  (Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014).
* ``minimum_backtest_length``: years of data needed before a Sharpe of a given
  size is distinguishable from the best of N noise trials
  (Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and Financial
  Charlatanism", 2014).
* ``probability_of_backtest_overfitting``: combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of
  Backtest Overfitting", 2017).

All Sharpe inputs here are **per-bar**, not annualized, unless the parameter
name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-bar Sharpe among ``n_trials`` unskilled strategies.

    ``sharpe_variance`` is the variance of the per-bar Sharpe estimates across
    the trials. With one trial there is no selection and the answer is zero.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    z_hi = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z_lo = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - _EULER_GAMMA) * z_hi + _EULER_GAMMA * z_lo)


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_bars: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """P[true per-bar Sharpe > benchmark], adjusted for sample length, skew, and fat tails."""
    if n_bars < 2 or not math.isfinite(sharpe):
        return math.nan
    kurtosis = excess_kurtosis + 3.0
    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return math.nan
    z = (sharpe - benchmark_sharpe) * math.sqrt(n_bars - 1) / math.sqrt(denominator)
    return _NORMAL.cdf(z)


def deflated_sharpe_ratio(
    sharpe: float,
    n_bars: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """PSR against the expected best-of-``n_trials`` Sharpe; above ~0.95 survives selection."""
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, benchmark, n_bars, skew, excess_kurtosis)


def minimum_backtest_length(n_trials: int, annual_sharpe: float) -> float:
    """Years of history needed before ``annual_sharpe`` beats the best of ``n_trials`` noise trials.

    MinBTL = (E[max of N standard-normal annual Sharpes] / annual_sharpe)^2, with the
    expectation approximated by ``(1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N e))``.
    """
    if annual_sharpe <= 0:
        return math.inf
    expected_max = expected_max_sharpe(n_trials, 1.0)
    if expected_max <= 0:
        return 0.0
    return (expected_max / annual_sharpe) ** 2


@dataclass(frozen=True)
class CSCVResult:
    """Output of ``probability_of_backtest_overfitting``.

    Attributes:
        pbo: Fraction of train/test splits where the in-sample winner ranked
            below the median out of sample.
        logits: One logit per split; negative values are out-of-sample losers.
        in_sample_sharpe: Per-bar Sharpe of the in-sample winner, per split.
        out_of_sample_sharpe: The same strategy's out-of-sample Sharpe, per split.
        n_splits: Number of train/test combinations evaluated.
    """

    pbo: float
    logits: np.ndarray
    in_sample_sharpe: np.ndarray
    out_of_sample_sharpe: np.ndarray
    n_splits: int

    @property
    def performance_degradation(self) -> float:
        """Slope of out-of-sample on in-sample Sharpe across splits; below 1 means decay."""
        if len(self.in_sample_sharpe) < 2 or np.std(self.in_sample_sharpe) == 0:
            return math.nan
        slope, _ = np.polyfit(self.in_sample_sharpe, self.out_of_sample_sharpe, 1)
        return float(slope)


def probability_of_backtest_overfitting(
    returns: pd.DataFrame | np.ndarray, n_blocks: int = 16
) -> CSCVResult:
    """CSCV estimate of the probability that in-sample selection picks an out-of-sample loser.

    ``returns`` has one column per trial (strategy or parameter set) and one row
    per bar. Rows are cut into ``n_blocks`` contiguous blocks; every way of
    choosing half the blocks as training data forms one split. The strategy with
    the best in-sample Sharpe is scored by its out-of-sample rank.

    Per-block sums are precomputed so each split is O(blocks x trials).
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be 2-D: bars x trials")
    n_bars, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError("need at least two trials")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    if n_bars < n_blocks:
        raise ValueError(f"need at least {n_blocks} bars for {n_blocks} blocks")

    edges = np.linspace(0, n_bars, n_blocks + 1, dtype=int)
    block_sum = np.empty((n_blocks, n_trials))
    block_sq = np.empty((n_blocks, n_trials))
    block_len = np.empty(n_blocks)
    for b in range(n_blocks):
        chunk = matrix[edges[b] : edges[b + 1]]
        block_sum[b] = chunk.sum(axis=0)
        block_sq[b] = (chunk**2).sum(axis=0)
        block_len[b] = len(chunk)

    def sharpe_of(blocks: np.ndarray) -> np.ndarray:
        count = block_len[blocks].sum()
        total = block_sum[blocks].sum(axis=0)
        total_sq = block_sq[blocks].sum(axis=0)
        mean = total / count
        var = (total_sq - count * mean**2) / (count - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), 0.0)

    all_blocks = np.arange(n_blocks)
    logits: list[float] = []
    is_best: list[float] = []
    oos_best: list[float] = []
    for train in combinations(all_blocks, n_blocks // 2):
        train_idx = np.array(train)
        test_idx = np.setdiff1d(all_blocks, train_idx)
        sr_train = sharpe_of(train_idx)
        sr_test = sharpe_of(test_idx)
        winner = int(np.argmax(sr_train))
        rank = 1 + int(np.sum(sr_test < sr_test[winner]))
        relative = rank / (n_trials + 1)
        logits.append(math.log(relative / (1.0 - relative)))
        is_best.append(float(sr_train[winner]))
        oos_best.append(float(sr_test[winner]))

    logit_arr = np.array(logits)
    return CSCVResult(
        pbo=float(np.mean(logit_arr < 0)),
        logits=logit_arr,
        in_sample_sharpe=np.array(is_best),
        out_of_sample_sharpe=np.array(oos_best),
        n_splits=len(logits),
    )
