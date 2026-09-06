import numpy as np
import pandas as pd
import pytest

from ubt.core.results import BacktestResult, LeakageFlag, ValidationReport


def _result(equity: list[float]) -> BacktestResult:
    return BacktestResult(
        equity_curve=pd.Series(equity),
        trades=pd.DataFrame(),
        engine_name="reference",
        cost_bps=0.0,
    )


def test_sharpe_matches_independent_numpy_calculation():
    equity = [100.0, 110.0, 108.0, 115.0]  # non-constant returns on purpose
    returns = pd.Series(equity).pct_change().dropna().to_numpy()
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)

    assert _result(equity).sharpe() == pytest.approx(expected)


def test_sharpe_is_zero_when_returns_are_constant():
    assert _result([100.0, 110.0, 121.0]).sharpe() == 0.0  # flat 10% steps -> std is exactly 0


def test_leakage_score_starts_at_100_with_no_flags():
    assert ValidationReport().leakage_score == 100.0


def test_leakage_score_penalizes_by_severity():
    report = ValidationReport(
        leakage_flags=[
            LeakageFlag(source="static", severity="info", message="x"),
            LeakageFlag(source="perturbation", severity="warn", message="y"),
            LeakageFlag(source="static", severity="high", message="z"),
        ]
    )
    assert report.leakage_score == pytest.approx(100.0 - (1 + 5 + 20))


def test_leakage_score_clamps_at_zero_not_negative():
    flags = [LeakageFlag(source="static", severity="high", message="x") for _ in range(10)]
    assert ValidationReport(leakage_flags=flags).leakage_score == 0.0
