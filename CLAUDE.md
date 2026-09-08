# CLAUDE.md — FullBacktester

Backtesting engine with built-in defenses against look-ahead, survivorship bias, unrealistic
execution, and overfitting. Publishable Python package (PyPI distribution `FullBacktester`, import
`fullbacktester`, alias `fbt`).
Owner: beginner-to-intermediate quant evaluating several strategies against each other.
Focus markets: US and India first, others as data quality allows.

## Stack

Python >= 3.11, numpy, pandas 2.2+/3.x, pyarrow, platformdirs. Optional: yfinance, requests
(NSE), exchange_calendars. Dev: pytest, hypothesis, ruff, mypy. Build: hatchling.
Venv: `.venv` (Python 3.12).

## Commands

```
.venv\Scripts\python.exe -m pytest -q                 # full suite
.venv\Scripts\ruff.exe check src tests examples scripts
.venv\Scripts\ruff.exe format src tests examples scripts
.venv\Scripts\python.exe -m mypy                      # strict, src only
.venv\Scripts\python.exe examples\quickstart.py       # offline synthetic demo
uv pip install --python .venv\Scripts\python.exe -e ".[dev,all]"
uv build                                              # sdist + wheel into dist/
```

## CI and release

`.github/workflows/ci.yml` on every push and PR to main:

| Job | What it guards |
|---|---|
| `quality` | ruff lint, ruff format, mypy strict (3.12) |
| `test` | pytest on 3.11/3.12/3.13 Linux, plus Windows and macOS on 3.12 |
| `test-core-only` | the suite with **no** optional extras, proving lazy imports and the weekday-calendar fallback |
| `package` | build, `twine check`, install the wheel into an empty venv and run `scripts/verify_install.py` |

`scripts/verify_install.py` tests the *installed* package with the source tree
off `sys.path`. It is the only thing that catches a module missing from the
wheel, a lost `py.typed`, a stale `__all__` entry, or an extra that is not
actually optional. Run it after any change to packaging or `__init__.py`.

`.github/workflows/release.yml` publishes via PyPI Trusted Publishing (OIDC, no
stored token). Its header comments carry the one-time PyPI and GitHub
Environment setup. Dry run goes to TestPyPI via workflow_dispatch; a real
release is a GitHub Release tagged `v<version>`, and the workflow refuses to
publish when the tag and `pyproject.toml` version disagree. A PyPI version can
never be reused, so the `pypi` environment should require a reviewer.

## Module map (`src/fullbacktester/`)

| Module | Owns |
|---|---|
| `markets.py` | `Market` (tz, session, sessions/yr), `Frequency`, `US`/`INDIA`/`CRYPTO`, session close/open instants |
| `flags.py` | `Flag`, `Severity`, `score()` — shared diagnostics vocabulary |
| `data/schema.py` | canonical long bar frame, `validate_bars` (strict, lists all violations) |
| `data/panel.py` | `Panel` (aligned read-only OHLCV arrays, UTC close timestamps), `PanelView` (bars `[0, t]`, the only thing strategies see), `replace_future` |
| `data/sources/` | `DataSource` ABC + registry; `local`, `yfinance`, `nse` adapters; per-market defaults |
| `data/cache.py` | Parquet cache keyed by source/market/frequency/symbol with fetched-range sidecar |
| `data/loader.py` | `load_bars` / `load_panel`: registry + cache + validation |
| `data/quality.py` | `check_data_quality`: gaps, zero volume, jumps, stale prices, source caveats |
| `strategy/base.py` | `Strategy` (imperative), `WeightStrategy`, `BatchWeightStrategy`, adapters, `reset()` hook |
| `strategy/context.py` | `BarContext`: view + portfolio snapshot + order API |
| `strategy/ml.py` | `WalkForwardMLStrategy` (horizon + embargo cutoff, training log), score-to-weight helpers |
| `execution/orders.py` | `Order` (qty or target weight), `Fill`, blotter frames |
| `execution/costs.py` | commission + slippage models, all NumPy-vectorized, one `CostModel` for both engines |
| `execution/config.py` | `EngineConfig`, `FillTiming`, `idealized()` |
| `execution/fills.py` | `FillModel`: next-open/limit/stop rules, participation and leverage caps |
| `execution/ledger.py` | cash + positions, `PortfolioSnapshot` |
| `execution/event_driven.py` | bar loop: fill pending, mark, `on_bar`, queue orders |
| `execution/vectorized.py` | batched targets + T-step NumPy accounting scan (idealized twin) |
| `execution/compare.py` | `EngineComparison`, `compare_engines` |
| `result.py` | `BacktestResult` (equity, weights, fills, orders, metrics) |
| `metrics/performance.py` | annualized stats; NaN when undefined |
| `metrics/overfitting.py` | PSR, DSR, MinBTL, CSCV PBO |
| `validation/static_scanner.py` | AST checks: `shift(-n)`, `rolling(center=True)`, bfill, fit-before-split |
| `validation/perturbation.py` | `FutureLeakTester`: truncation test (batch) + future-noise test (any) |
| `validation/cv.py` | `PurgedKFold` |
| `validation/report.py` | `ValidationReport`, `validate()` |
| `arena.py` | run N strategies under identical assumptions; DSR uses N as trials; PBO |
| `live/store.py` | SQLite session state; bars written once and never rewritten, revisions logged |
| `live/session.py` | `PaperSession`: fingerprint lock, fetch, replay unprocessed bars, persist |
| `live/report.py` | `replay_check` (machinery correctness), `expectation_gap` (edge decay) |

## Invariants (tests enforce these; do not break them)

1. Bar `timestamp` = UTC instant of bar **close**. Daily bars use the market's session close.
2. A strategy at bar `t` sees `PanelView` of bars `<= t` only. Never hand a strategy the `Panel`.
3. Signals from bar `t` fill at bar `t+1` open by default. `SAME_CLOSE` is opt-in and flagged.
4. Under `EngineConfig.idealized()` both engines agree to 1e-9 (hypothesis property test).
5. Both engines use the same `CostModel` arithmetic; slippage moves price against the trade.
6. Target-weight orders are sized at fill time from equity marked at the execution price.
7. `WalkForwardMLStrategy` trains only on samples with `label_end + embargo <= now`.
8. Undefined metrics are NaN, never 0. Annualization comes from the panel, never a constant.
9. Every data source declares `adjusted` and `survivorship_free`; caveats become flags.
10. A `PaperSession` stepped one bar at a time must equal `EventDrivenEngine` on the same
    bars to 1e-12, whether stepped daily or caught up after missed runs. Tests enforce it.
11. Recorded live bars are immutable. A source that restates one gets a `bar_revisions`
    row and a WARN; the original is what the strategies traded on and stays.

## How to extend

* New strategy: subclass `WeightStrategy` (declarative) or `Strategy` (needs orders/fills).
  Set `warmup`. If it caches state across bars, implement `reset()`.
* New data source: subclass `DataSource`, implement `fetch` returning canonical bars with UTC
  close timestamps, register in `data/sources/__init__.py`. Never adjust the schema to a source.
* New cost model: implement `commission(quantity, price)` or `slippage_bps(participation)`
  with NumPy-compatible math so both engines share it.
* New leakage check: return `list[Flag]` with `source` set; wire into `validate()` and `Arena`.
* New metric: add to `PerformanceMetrics` and `compute_metrics`; keep NaN semantics.

## Gotchas

* `BatchWeightStrategy` on the event-driven engine is re-evaluated per bar on a growing view:
  O(T^2). Leaky batch code goes **flat** there and profits on the vectorized engine; the engine
  gap is the tell. Expected, not a bug.
* Perturbation tests compare the *decision* (target weight / explicit quantity), not the
  fill-resolved quantity, which legitimately changes with perturbed prices.
* `Order.quantity` for target orders is NaN until filled; check `is_target`.
* Turnover per bar can reach 2x equity (full rotation). Not a bug.
* NSE bhavcopy prices are unadjusted. yfinance India tickers get `.NS` unless a suffix exists.
* NSE runs a Diwali Muhurat session each year that no standard calendar lists, sometimes on a
  weekend (2019-10-27 Sun, 2020-11-14 Sat). `check_data_quality` reports these as INFO
  'unscheduled' bars, not as errors. It compares session *dates*, never counts: subtracting
  counts let those six sessions mask three genuinely absent ones on a live 2018-2024 pull.
* `exchange_calendars` has no `XNSE`; `INDIA.calendar_code` is `XBOM` (same holidays). Unknown
  codes degrade to weekday counting, never raise. NSE declares ad-hoc holidays the calendar
  may not know (2024-01-22 showed as "1 session missing" on a live pull); treat that WARN as
  a prompt to check, not as bad data.
* Live-verified 2026-09-06 with yfinance 1.7: US/IN daily, BTC-USD daily, AAPL 1h bars
  (stamped at bar end), cache sub-range slicing.
* Synthetic test panels use `Market.expected_sessions`, so session counts stay consistent
  whether or not `exchange_calendars` is installed.
* Panel arrays are read-only; pandas 3 copy-on-write means a strategy's write silently copies,
  pandas 2 raises. Either way the panel is safe.
* Paper sessions refuse to open when strategy source or `EngineConfig` changed since
  registration (`SessionLockError`). That is the feature, not a bug: retuning mid-run is
  how people fool themselves. Start a new session.
* `Order.id` comes from a process-local counter, and it is the orders table's primary key.
  `PaperSession` calls `reserve_order_ids(store.max_order_id())` on create and open; without
  it a resumed session in a fresh interpreter restarts at 1 and overwrites its own blotter.
  Tests that claim to check resumption must reset that counter or they check nothing.
* `LiveStore._connect` leaves sqlite3's `isolation_level` at its default. Driving BEGIN by
  hand breaks on `executescript`, which commits any pending transaction before running.
* `tests/test_readme_examples.py` runs every code block in README.md. Change the README,
  change that file. A failure there means the README is lying to somebody.
* `tests/` is a package; import helpers as `from tests.conftest import make_panel`.
* Windows: prefer `.venv\Scripts\python.exe -m pytest`; PowerShell blocks some header strings.

## Conventions

Ruff (line 100, rules E/F/I/UP/B/SIM/RUF), mypy strict on `src`. Dataclasses over dicts,
`StrEnum` over string constants. Docstrings state what a check does **and does not** catch.
Every architectural decision goes in `DECISIONS.md` with the argument, not just the outcome.
Conventional commits.
