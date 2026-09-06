# CLAUDE.md — quantgauntlet

Backtesting engine with built-in defenses against look-ahead, survivorship bias, unrealistic
execution, and overfitting. Publishable Python package (PyPI name `quantgauntlet`, alias `qg`).
Owner: beginner-to-intermediate quant evaluating several strategies against each other.
Focus markets: US and India first, others as data quality allows.

## Stack

Python >= 3.11, numpy, pandas 2.2+/3.x, pyarrow, platformdirs. Optional: yfinance, requests
(NSE), exchange_calendars. Dev: pytest, hypothesis, ruff, mypy. Build: hatchling.
Venv: `.venv` (Python 3.12). `.venv_pypi` is a stale leftover, safe to delete.

## Commands

```
.venv\Scripts\python.exe -m pytest -q                 # full suite
.venv\Scripts\ruff.exe check src tests; .venv\Scripts\ruff.exe format src tests
.venv\Scripts\python.exe examples\quickstart.py       # offline synthetic demo
uv pip install --python .venv\Scripts\python.exe -e ".[dev,all]"
```

## Module map (`src/quantgauntlet/`)

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
* `tests/` is a package; import helpers as `from tests.conftest import make_panel`.
* Windows: prefer `.venv\Scripts\python.exe -m pytest`; PowerShell blocks some header strings.

## Conventions

Ruff (line 100, rules E/F/I/UP/B/SIM/RUF), mypy strict on `src`. Dataclasses over dicts,
`StrEnum` over string constants. Docstrings state what a check does **and does not** catch.
Every architectural decision goes in `DECISIONS.md` with the argument, not just the outcome.
Conventional commits.
