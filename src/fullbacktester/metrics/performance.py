"""Performance statistics from a bar-return series.

Every statistic that involves time uses ``bars_per_year`` from the panel, so
daily US, daily Indian, hourly crypto, and 5-minute equity strategies are all
annualized correctly and comparably. Undefined quantities (Sharpe of a constant
series, Calmar with no drawdown) are NaN, never zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    n_bars: int
    bars_per_year: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_bars: int
    calmar: float
    hit_rate: float
    best_bar: float
    worst_bar: float
    skew: float
    excess_kurtosis: float
    avg_turnover: float
    avg_gross_exposure: float
    n_fills: int
    total_costs: float
    cost_drag: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.as_dict())


def _as_float(value: Any) -> float:
    """pandas reductions are typed as a wide scalar union; we only ever feed them numbers."""
    return float(value)


def periodic_rate(annual_rate: float, bars_per_year: float) -> float:
    """Per-bar rate equivalent to a compounded annual rate."""
    return float((1.0 + annual_rate) ** (1.0 / bars_per_year) - 1.0)


def sharpe_ratio(returns: pd.Series, bars_per_year: float, *, rf_annual: float = 0.0) -> float:
    excess = returns.to_numpy(dtype=np.float64) - periodic_rate(rf_annual, bars_per_year)
    if excess.size < 2:
        return math.nan
    std = excess.std(ddof=1)
    if not std > 0:
        return math.nan
    return float(excess.mean() / std * math.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series, bars_per_year: float, *, rf_annual: float = 0.0) -> float:
    excess = returns.to_numpy(dtype=np.float64) - periodic_rate(rf_annual, bars_per_year)
    if excess.size < 2:
        return math.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    if not downside > 0:
        return math.nan
    return float(excess.mean() / downside * math.sqrt(bars_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Deepest peak-to-trough decline and the longest stretch of bars spent below a prior peak."""
    dd = drawdown_series(equity)
    underwater = dd < 0
    longest = 0
    run = 0
    for flag in underwater.to_numpy():
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(dd.min()) if len(dd) else math.nan, longest


def cagr(total_return: float, n_bars: int, bars_per_year: float) -> float:
    if n_bars <= 0 or total_return <= -1.0:
        return math.nan if n_bars <= 0 else -1.0
    years = n_bars / bars_per_year
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def compute_metrics(
    returns: pd.Series,
    bars_per_year: float,
    *,
    rf_annual: float = 0.0,
    turnover: pd.Series | None = None,
    gross_exposure: pd.Series | None = None,
    n_fills: int = 0,
    total_costs: float = 0.0,
    initial_equity: float | None = None,
) -> PerformanceMetrics:
    """All statistics from bar returns (equity[t] / equity[t-1] - 1).

    ``turnover`` is traded notional per bar as a fraction of equity;
    ``gross_exposure`` is sum(|weights|) per bar. ``cost_drag`` is total costs
    divided by initial equity: the fraction of starting capital paid away.
    """
    r = returns.astype("float64")
    n = len(r)
    growth = float(np.prod(1.0 + r.to_numpy())) if n else 1.0
    total_return = growth - 1.0
    equity = (1.0 + r).cumprod()
    mdd, mdd_bars = max_drawdown(equity) if n else (math.nan, 0)
    growth_rate = cagr(total_return, n, bars_per_year)
    vol = float(r.std(ddof=1) * math.sqrt(bars_per_year)) if n > 1 else math.nan
    nonzero = r[r != 0]
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) else math.nan
    calmar = growth_rate / abs(mdd) if (mdd < 0 and not math.isnan(growth_rate)) else math.nan

    return PerformanceMetrics(
        n_bars=n,
        bars_per_year=float(bars_per_year),
        total_return=total_return,
        cagr=growth_rate,
        annual_volatility=vol,
        sharpe=sharpe_ratio(r, bars_per_year, rf_annual=rf_annual),
        sortino=sortino_ratio(r, bars_per_year, rf_annual=rf_annual),
        max_drawdown=mdd,
        max_drawdown_bars=mdd_bars,
        calmar=calmar,
        hit_rate=hit_rate,
        best_bar=float(r.to_numpy().max()) if n else math.nan,
        worst_bar=float(r.to_numpy().min()) if n else math.nan,
        skew=_as_float(r.skew()) if n > 2 else math.nan,
        excess_kurtosis=_as_float(r.kurt()) if n > 3 else math.nan,
        avg_turnover=float(turnover.mean()) if turnover is not None and len(turnover) else math.nan,
        avg_gross_exposure=(
            float(gross_exposure.mean())
            if gross_exposure is not None and len(gross_exposure)
            else math.nan
        ),
        n_fills=int(n_fills),
        total_costs=float(total_costs),
        cost_drag=float(total_costs / initial_equity) if initial_equity else math.nan,
    )
