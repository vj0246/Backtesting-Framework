# quantgauntlet

[![CI](https://github.com/vj0246/Backtesting-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/vj0246/Backtesting-Framework/actions/workflows/ci.yml)

Backtesting for people who have been burned by backtests.

`quantgauntlet` runs trading strategies through a gauntlet of checks that most
frameworks leave to the user's discipline: point-in-time data access that makes
look-ahead a construction error, two execution engines whose disagreement is
itself a diagnostic, static and behavioural leakage detectors, data-quality and
survivorship flags, and performance statistics that account for how many
strategies you tried before picking this one.

> Status: alpha. The core engine, validation, and metrics are tested; the data
> sources are thin adapters over free APIs. Read *Limitations* before trusting a
> number.

## What it defends against

| Pitfall | Defense |
|---|---|
| Look-ahead through the data | Strategies only ever receive a `PanelView` that ends at the current bar. Bars are stamped at their **close** instant in UTC, so a daily bar is not "known" until the exchange actually closed. |
| Look-ahead in strategy code | `StaticScanner` (AST patterns: negative `shift`, centred `rolling`, backward fill, fit-before-split) plus `FutureLeakTester`, which reruns the strategy with bars after `t` replaced by noise and with the history truncated at `t`; a decision at `t` that changes was reading the future. |
| Look-ahead in ML training | `WalkForwardMLStrategy` only trains on samples whose label window closed at least `embargo` bars before the decision, and logs the cutoff at every refit so a test can prove it. `PurgedKFold` for offline model selection. |
| Unrealistic execution | Event-driven engine with a cash ledger, next-open fills, limit/stop orders, commission and volume-dependent slippage, lot rounding, participation caps, leverage caps, short restrictions. |
| Vectorized wishful thinking | The vectorized engine exists for speed, uses the *same* cost model, and is checked against the event-driven engine. Under idealized settings they agree to 1e-9; under realistic settings the gap is reported as implementation risk. |
| Survivorship bias | Every source declares whether its universe is survivorship-free; the report says so. NSE's official archive gives the full listed universe per day. |
| Bad data | Schema validation on entry (no NaN closes, no negative volume, OHLC consistent), plus quality flags for missing sessions, zero volume, suspected unadjusted splits, stale prices. |
| Overfitting / selection | Deflated Sharpe Ratio with the number of trials, Probabilistic Sharpe Ratio, minimum backtest length, and Probability of Backtest Overfitting via CSCV across the strategies in an `Arena`. |

## Install

```bash
pip install quantgauntlet                 # core: numpy, pandas, pyarrow
pip install "quantgauntlet[yfinance]"     # Yahoo Finance adapter (US, India .NS/.BO, crypto)
pip install "quantgauntlet[nse]"          # NSE India official bhavcopy adapter
pip install "quantgauntlet[calendars]"    # holiday-aware session counts via exchange_calendars
pip install "quantgauntlet[all]"
```

Python 3.11 or newer.

## Quickstart

```python
import quantgauntlet as qg

# 1. Data. Daily bars for two US names, cached locally as Parquet after the first pull.
panel = qg.load_panel(["AAPL", "MSFT", "GOOGL"], "2018-01-01", "2024-12-31", market="US")

# 2. A strategy: target weights from everything known at the current bar.
def momentum(view: qg.PanelView) -> dict[str, float]:
    close = view.close
    up = (close.iloc[-1] / close.iloc[-126] - 1.0 > 0).astype(float)
    return (up / max(up.sum(), 1.0)).to_dict()

# 3. Execution assumptions, shared by every strategy you compare.
config = qg.EngineConfig(
    initial_cash=1_000_000,
    cost_model=qg.CostModel(
        commission=qg.BpsCommission(1.0),
        slippage=qg.VolumeShareSlippage(impact_coefficient=0.1),
    ),
    max_participation=0.05,
    max_gross_leverage=1.0,
)

# 4. Run the gauntlet.
arena = qg.Arena(panel, config)
arena.add(qg.RuleBasedStrategy(momentum, warmup=126), "momentum_6m")
result = arena.run()
print(result.summary())
```

The summary table reports, per strategy: CAGR, Sharpe, **deflated** Sharpe (given
the number of entries in the arena), max drawdown, turnover, cost drag, the
Sharpe gap between the idealized and realistic engines, and a validation score
with every HIGH-severity finding printed underneath.

### Indian equities

```python
panel = qg.load_panel(["RELIANCE", "TCS", "INFY"], "2015-01-01", "2024-12-31", market="IN")
# adjusted prices from Yahoo (.NS suffix added for you)

nse = qg.NSEBhavcopySource()
universe = nse.list_symbols(date(2020, 1, 1))     # every EQ/BE symbol listed that day
raw = qg.load_panel(universe[:50], "2020-01-01", "2020-12-31", market="IN", source=nse)
# official, survivorship-free, UNADJUSTED: use for universe membership and volume checks
```

### Local files

```python
src = qg.LocalSource("data/")                       # data/AAPL.csv, data/MSFT.parquet, ...
panel = qg.load_panel(["AAPL", "MSFT"], "2020-01-01", "2023-12-31", market="US", source=src)
```

Naive dates are treated as session dates and stamped at the market's close.

## Concepts

**Panel / PanelView.** `Panel` holds aligned OHLCV arrays for all symbols on one
UTC timeline (union of every symbol's bar closes; NaN where a symbol has no bar).
`PanelView` is bars `[0, t]` and is all a strategy ever sees. The arrays are
read-only.

**Strategy shapes.**

| Class | You implement | Runs on |
|---|---|---|
| `Strategy` | `on_bar(ctx)`: read `ctx.view`, `ctx.portfolio`; call `ctx.order(...)`, `ctx.order_target_weight(...)` | event-driven |
| `WeightStrategy` | `target_weights(view) -> {symbol: weight}` | both |
| `BatchWeightStrategy` | `target_weights_batch(view) -> DataFrame` (all bars at once) | both |
| `RuleBasedStrategy(fn)` / `BatchRuleStrategy(fn)` | nothing; wraps a function | both |
| `WalkForwardMLStrategy` | `feature_fn`, a model factory, horizon, embargo | both |

Weights are fractions of equity, signed; `sum(abs(w))` is gross leverage. NaN
means "no opinion", which leaves the position alone.

**Engines.** `EventDrivenEngine` fills orders at the next bar's open (by
default), keeps a ledger, applies every constraint in `EngineConfig`.
`VectorizedEngine` runs weight strategies with batched signals and a ledger-free
NumPy scan. `compare_engines` returns an `EngineComparison`.

**Fill timing.** A signal computed from bar `t`'s close executes at bar `t+1`'s
open. `FillTiming.SAME_CLOSE` is available for market-on-close strategies and is
flagged as optimistic in every report that sees it. Limit and stop orders are
evaluated against the next bar's range with conservative fill prices.

**Validation.** `validate(strategy, panel)` returns a `ValidationReport` with
flags from data quality, the static scanner, the perturbation tester, and the
engine comparison, plus a 0-100 score. The score is a summary, not a verdict;
read the flags.

**Arena.** Same panel, same config, same checks for every entry, and the
overfitting statistics use the number of entries as the number of trials.

## Metrics

`BacktestResult.metrics()` gives total return, CAGR, annualized volatility,
Sharpe, Sortino, max drawdown and its duration, Calmar, hit rate, skew, excess
kurtosis, average turnover, average gross exposure, fill count, total costs, and
cost drag. Annualization comes from the panel's market and bar frequency (252
US sessions, 250 Indian sessions, 365 crypto days, intraday multiples of each).
Undefined statistics are NaN, never zero.

`quantgauntlet.metrics` also exposes `probabilistic_sharpe_ratio`,
`deflated_sharpe_ratio`, `minimum_backtest_length`, and
`probability_of_backtest_overfitting` (CSCV) for use outside an arena.

## Data sources

| Source | Markets | Frequencies | Adjusted | Survivorship-free |
|---|---|---|---|---|
| `local` | any | any | unknown (declare it) | no |
| `yfinance` | US, IN, CRYPTO | 1m to 1w | yes (`auto_adjust`) | no |
| `nse` | IN | daily | **no** | **yes** |

Defaults: `US`, `IN`, and `CRYPTO` resolve to `yfinance` because unadjusted prices
produce wrong returns. Pass `source="nse"` or a `DataSource` instance to override.
Everything fetched lands in a Parquet cache (`platformdirs` user cache directory)
keyed by source, market, frequency, and symbol.

Adding a source is one class: subclass `DataSource`, declare `name`, `markets`,
`frequencies`, `adjusted`, `survivorship_free`, implement `fetch`, and call
`quantgauntlet.data.sources.register`.

## Limitations

* **Corporate actions** are not modelled point-in-time. You trade on the adjusted
  series your source provides (D-008 in `DECISIONS.md`). Dividends are captured
  only when the source folds them into prices.
* **Intraday fills** assume the bar's open is attainable and that limit/stop
  fills are decided by the bar's range, not by tick data.
* **Batch strategies on the event-driven engine** are re-evaluated on a growing
  window each bar (quadratic in bars). Fine for years of daily data; for long
  intraday histories write a per-bar `WeightStrategy`.
* **The leakage tests are incomplete by nature.** They cannot see a strategy that
  reads the future from a global variable or a file. The report says exactly which
  tests ran.
* **Free data is free.** Yahoo's universe is whatever exists today; NSE bhavcopy is
  unadjusted. The flags will remind you.
* No options, futures margining, borrow costs, or FX conversion in v0.1.

## Development

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev,all]"
pytest                                            # ~80 tests incl. hypothesis property tests
ruff check src tests examples scripts && ruff format --check src tests examples scripts
mypy
```

CI runs the suite on Python 3.11, 3.12, and 3.13 on Linux plus one Windows and
one macOS leg, and separately with **no optional extras installed**, because the
extras are advertised as optional and the adapters must import without them.

A passing test suite does not prove the *package* is sound. To check what users
will actually install, build it and exercise the installed copy:

```powershell
.\scripts\clean_room_test.ps1     # Windows: build, check, install, exercise
```

```bash
# any platform, by hand
python -m build && python -m twine check dist/*
python -m venv /tmp/clean && /tmp/clean/bin/python -m pip install dist/*.whl
cd /tmp && /tmp/clean/bin/python "$OLDPWD/scripts/verify_install.py"
```

`scripts/verify_install.py` runs against the installed package with the source
tree off the path. It catches what a test run cannot: a module missing from the
wheel, a lost `py.typed` marker, a stale `__all__` entry, an extra that is not
really optional.

## Releasing

Publishing uses PyPI Trusted Publishing, so no API token is stored anywhere.
The one-time PyPI and GitHub Environment setup is documented at the top of
`.github/workflows/release.yml`. Once that is done:

1. Dry run: Actions -> Release -> Run workflow -> target `testpypi`.
2. Real release: bump `version` in `pyproject.toml`, commit, then publish a
   GitHub Release tagged `v<version>`. The workflow refuses to publish if the
   tag and the declared version disagree.

A PyPI version can never be re-uploaded, so the `pypi` environment should have a
required reviewer. That approval is the last chance to stop a bad release.

`DECISIONS.md` records every architectural decision with the argument that was
had and why the resolution won. `CLAUDE.md` is the map of the codebase.

## License

MIT.
