"""Tests for StaticScanner against synthetic examples of each pattern it
knows about, plus clean code and a source-unavailable edge case."""

from ubt.core.interfaces import MLStrategy, RuleBasedStrategy
from ubt.validation.static_scanner import StaticScanner


def test_flags_negative_shift():
    def signal_fn(data):
        future = data["price"].shift(-1)
        return {"AAA": 1.0 if future.iloc[-1] > 0 else 0.0}

    flags = StaticScanner().run(RuleBasedStrategy(signal_fn), feed=None)
    shift_flags = [f for f in flags if "shift(-1)" in f.message]
    assert len(shift_flags) == 1
    assert shift_flags[0].severity == "high"


def test_flags_centered_rolling():
    def signal_fn(data):
        smoothed = data["price"].rolling(5, center=True).mean()
        return {"AAA": 1.0 if smoothed.iloc[-1] > 0 else 0.0}

    flags = StaticScanner().run(RuleBasedStrategy(signal_fn), feed=None)
    assert any("center=True" in f.message for f in flags)


def test_flags_fit_before_split():
    def feature_fn(data):
        from sklearn.model_selection import train_test_split  # not executed, only parsed
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(data)  # fit on everything first...
        train, test = train_test_split(data)  # ...then split. too late.
        return scaler.transform(data)

    class DummyModel:
        def predict(self, features):
            return {"AAA": 0.0}

    strategy = MLStrategy(model=DummyModel(), feature_fn=feature_fn)
    flags = StaticScanner().run(strategy, feed=None)
    assert any(".fit()" in f.message for f in flags)


def test_clean_code_produces_no_flags():
    def signal_fn(data):
        past = data["price"].shift(1)  # backward-looking, fine
        return {"AAA": 1.0 if past.iloc[-1] > 0 else 0.0}

    flags = StaticScanner().run(RuleBasedStrategy(signal_fn), feed=None)
    assert flags == []


def test_unreadable_source_produces_info_flag_not_a_crash():
    unreadable = eval(compile("lambda d: {}", "<string>", "eval"))
    flags = StaticScanner().run(RuleBasedStrategy(unreadable), feed=None)
    assert len(flags) == 1
    assert flags[0].severity == "info"
