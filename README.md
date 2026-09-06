# ubt (placeholder name)

Interface layer only — no working engine yet. `src/ubt/__init__.py` states
plainly what's implemented vs. stubbed; that note is the source of truth,
not this README.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the smoke test

```
python examples/smoke_test.py
```

Exercises `RuleBasedStrategy`, `BacktestResult.sharpe()`, and
`ValidationReport.leakage_score` — the only pieces with real logic right
now. It does not run a backtest; there's no engine to run one with yet.

## Run the tests

```
pytest -v
```

`test_validation_stubs.py` asserts that `StaticScanner` and
`PerturbationTester` currently raise `NotImplementedError`. That's
intentional: those two tests are meant to start failing once real detection
logic replaces the stubs. Update them at that point — don't delete them
before then.

## What's here now vs. not yet

- `VectorizedEngine` and `EventDrivenEngine` both work and are tested —
  but they currently share the exact same computation (no per-bar state
  in the event-driven one yet), so `compare_engines()` will read ~0 until
  that changes. See `src/ubt/core/engines.py` docstring.
- `StaticScanner` is real: catches negative `.shift()`, `rolling(center=
  True)`, and `.fit()`-before-`split()` (statement-order heuristic, not
  data-flow analysis). See `src/ubt/validation/static_scanner.py` for
  exactly what each check does and doesn't catch.
- Not built yet: `PerturbationTester`, Deflated Sharpe Ratio /
  Probability of Backtest Overfitting, real equity or crypto data
  adapters (the smoke test uses a toy in-memory feed).
