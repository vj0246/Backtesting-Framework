"""Static scanner, perturbation tester, purged CV, and the combined report."""

import pandas as pd
import pytest

from fullbacktester.execution import EngineConfig
from fullbacktester.flags import Severity
from fullbacktester.strategy import BatchRuleStrategy, RuleBasedStrategy, Strategy, WeightStrategy
from fullbacktester.validation import FutureLeakTester, PurgedKFold, StaticScanner, validate

# ------------------------------------------------------------------ scanner


def test_scanner_flags_negative_shift():
    def signal(view):
        future = view.close.shift(-1)
        return (future.iloc[-1] > 0).astype(float)

    flags = StaticScanner().scan(RuleBasedStrategy(signal))
    hits = [f for f in flags if "shift(-1)" in f.message]
    assert len(hits) == 1 and hits[0].severity is Severity.HIGH
    assert hits[0].file.endswith("test_validation.py") and hits[0].line is not None


def test_scanner_flags_centered_rolling_and_backfill():
    def signal(view):
        smooth = view.close.rolling(5, center=True).mean().bfill()
        return (smooth.iloc[-1] > 0).astype(float)

    messages = [f.message for f in StaticScanner().scan(RuleBasedStrategy(signal))]
    assert any("center=True" in m for m in messages)
    assert any("backward fill" in m for m in messages)


def test_scanner_flags_fit_before_split():
    def features(view):
        from sklearn.model_selection import train_test_split  # parsed, never executed
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(view.close)
        train, _test = train_test_split(view.close)
        return scaler.transform(train)

    class Wrapper(Strategy):
        def on_bar(self, ctx):
            pass

        def scan_targets(self):
            return [features]

    flags = StaticScanner().scan(Wrapper())
    assert any(".fit()" in f.message and f.severity is Severity.WARN for f in flags)


def test_scanner_clean_code_and_unreadable_source():
    def signal(view):
        past = view.close.shift(1)
        return (past.iloc[-1] > 0).astype(float)

    assert StaticScanner().scan(RuleBasedStrategy(signal)) == []
    unreadable = eval(compile("lambda v: {}", "<string>", "eval"))
    flags = StaticScanner().scan(RuleBasedStrategy(unreadable))
    assert len(flags) == 1 and flags[0].severity is Severity.INFO


# ------------------------------------------------------------- perturbation


def leaky_batch(view):
    close = view.close
    signal = (close.shift(-1) / close - 1.0 > 0).astype(float)  # tomorrow's return
    return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)


def clean_batch(view):
    close = view.close
    signal = (close / close.shift(5) - 1.0 > 0).astype(float)
    return signal.div(signal.sum(axis=1).clip(lower=1.0), axis=0)


def full_sample_normalized(view):
    close = view.close
    z = (close - close.mean()) / close.std()  # subtle: normalizes with the whole sample
    return (z > 0).astype(float).div(3.0)


class Cheater(WeightStrategy):
    """Reads the next bar through the panel behind the view."""

    def target_weights(self, view):
        panel = view._panel
        nxt = panel.close[min(view.index + 1, len(panel) - 1)]
        return pd.Series(
            (nxt > panel.close[view.index]).astype(float) / 3.0, index=list(view.symbols)
        )


def test_truncation_test_catches_batch_lookahead(panel):
    flags = FutureLeakTester(n_samples=4).run(BatchRuleStrategy(leaky_batch, warmup=5), panel)
    assert any(f.severity is Severity.HIGH and "reads the future" in f.message for f in flags)


def test_truncation_test_catches_full_sample_normalization(panel):
    flags = FutureLeakTester(n_samples=4).run(BatchRuleStrategy(full_sample_normalized), panel)
    assert any(f.severity is Severity.HIGH for f in flags)


def test_clean_batch_strategy_passes_both_tests(panel):
    flags = FutureLeakTester(n_samples=4).run(BatchRuleStrategy(clean_batch, warmup=5), panel)
    assert all(f.severity is Severity.INFO for f in flags)
    assert any("truncation test passed" in f.message for f in flags)
    assert any("future-noise test passed" in f.message for f in flags)


def test_future_noise_test_catches_strategy_bypassing_the_view(panel):
    flags = FutureLeakTester(n_samples=4).run(Cheater(), panel)
    assert any(f.severity is Severity.HIGH and "beyond its view" in f.message for f in flags)


def test_future_noise_test_passes_honest_imperative_strategy(panel):
    class Honest(Strategy):
        def on_bar(self, ctx):
            if ctx.bar % 10 == 0:
                ctx.order_target_weight(
                    "S0",
                    0.5 if ctx.view.close["S0"].iloc[-1] > ctx.view.close["S0"].iloc[0] else 0.0,
                )

    flags = FutureLeakTester(n_samples=4).run(Honest(), panel)
    assert all(f.severity is Severity.INFO for f in flags)


def test_perturbation_needs_enough_bars(panel_factory):
    tiny = panel_factory(n_bars=3)
    flags = FutureLeakTester().run(RuleBasedStrategy(lambda v: {}, warmup=5), tiny)
    assert flags[0].severity is Severity.INFO and "too short" in flags[0].message


# ---------------------------------------------------------------------- cv


def test_purged_kfold_removes_overlapping_and_embargoed_samples():
    cv = PurgedKFold(n_splits=4, purge=3, embargo=2)
    folds = list(cv.split(40))
    assert len(folds) == 4
    for train, test in folds:
        assert not set(train) & set(test)
        lo, hi = test.min(), test.max()
        forbidden = set(range(lo - 3, hi + 3 + 2 + 1))
        assert not set(train) & forbidden
    first_train, first_test = folds[0]
    assert first_test.min() == 0 and first_train.min() == 10 + 5
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)


# ------------------------------------------------------------------ report


def test_validate_combines_checks_and_scores(panel):
    report = validate(
        BatchRuleStrategy(leaky_batch, warmup=5),
        panel,
        config=EngineConfig.idealized(),
        n_samples=3,
    )
    sources = report.by_source()
    assert "static" in sources and "perturbation" in sources
    assert report.has_high
    assert report.score < 60
    # The vectorized engine runs the leaky batch once over the full history and profits from
    # the future; the event-driven engine evaluates it on truncated views and cannot. Their
    # disagreement is itself a look-ahead detector.
    assert report.comparison is not None and not report.comparison.agrees()
    assert report.comparison.max_equity_gap > 0.5
    assert "engine_comparison" in report.by_source()
    text = report.summary()
    assert "validation score" in text and "engines:" in text

    clean = validate(BatchRuleStrategy(clean_batch, warmup=5), panel, n_samples=3)
    assert not clean.has_high
    assert clean.score > 90
