# FullBacktester

[![CI](https://github.com/vj0246/Backtesting-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/vj0246/Backtesting-Framework/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/FullBacktester)](https://pypi.org/project/FullBacktester/)
[![Python](https://img.shields.io/pypi/pyversions/FullBacktester)](https://pypi.org/project/FullBacktester/)
[![License](https://img.shields.io/pypi/l/FullBacktester)](LICENSE)

**A backtesting engine that tries to prove your strategy wrong.**

Most backtesters help you produce a number. This one helps you find out whether
the number means anything, then keeps checking after you go live.

```bash
pip install "FullBacktester[all]"
```

---

## Why this exists

Here is a real result from this library, fifteen NSE large caps, 2018 to 2024,
with Indian retail costs applied:

| Strategy | CAGR | Sharpe | Cost drag |
|---|---|---|---|
| equal weight, rebalanced | **17.7%** | **1.02** | 1.4% |
| momentum, 6-month, top 5 | 8.7% | 0.55 | 18.0% |
| reversal, 1-week, bottom 5 | 8.2% | 0.50 | 93.1% |

Both "strategies" lost to doing nothing clever. Momentum's Sharpe was 0.68
before costs and 0.55 after; at 25bps it turns **negative**. The reversal
strategy paid away 93% of the starting capital in fees.

A backtester that reports gross returns on a vectorized engine would have shown
you an edge in both. This one shows you the bill.

---

## Contents

1. [Install](#install)
2. [Sixty-second start](#sixty-second-start)
3. [The mental model](#the-mental-model)
4. [Writing a strategy](#writing-a-strategy)
5. [Getting data](#getting-data)
6. [Making execution realistic](#making-execution-realistic)
7. [Catching look-ahead](#catching-look-ahead)
8. [Comparing strategies honestly](#comparing-strategies-honestly)
9. [Paper trading](#paper-trading)
10. [Metrics reference](#metrics-reference)
11. [Extending it](#extending-it)
12. [Limitations](#limitations)

---

## Install

```bash
pip install FullBacktester                 # core: numpy, pandas, pyarrow
pip install "FullBacktester[yfinance]"     # Yahoo: US, India (.NS/.BO), crypto
pip install "FullBacktester[nse]"          # NSE India official bhavcopy
pip install "FullBacktester[calendars]"    # holiday-aware sessions
pip install "FullBacktester[all]"          # all of the above
```

Python 3.11+. The distribution is `FullBacktester`; the import is
`fullbacktester`, lowercase, because module names must be. Everyone aliases it:

```python
import fullbacktester as fbt
```

## Sixty-second start

```python
import fullbacktester as fbt

# 1. Data. Cached as Parquet after the first pull, so the second run is instant.
panel = fbt.load_panel(
    ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"],
    "2019-01-01", "2024-12-31",
    market="IN",
)

# 2. A strategy is a function: what you know now -> what you want to hold.
def momentum(view: fbt.PanelView) -> dict[str, float]:
    returns = view.close.iloc[-1] / view.close.iloc[-126] - 1.0   # 6-month return
    winners = returns.nlargest(3).index
    return {symbol: 1 / 3 for symbol in winners}

# 3. Say what trading actually costs you.
config = fbt.EngineConfig(
    initial_cash=1_000_000,
    cost_model=fbt.CostModel(
        commission=fbt.BpsCommission(5.0),
        slippage=fbt.VolumeShareSlippage(impact_coefficient=0.1),
    ),
    max_participation=0.02,     # never take more than 2% of a bar's volume
    allow_short=False,
)

# 4. Run it, and let the library try to break it.
arena = fbt.Arena(panel, config)
arena.add(fbt.RuleBasedStrategy(momentum, warmup=126), "momentum_6m")
print(arena.run().summary())
```

`warmup=126` means "do not trade until 126 bars exist", so `iloc[-126]` is
never a truncated window. Get it wrong and you get an obvious error, not a
silent wrong answer.

Prefer to see it work with no network and no API keys?

```bash
python examples/quickstart.py       # backtesting, including a deliberately leaky strategy
python examples/paper_trading.py    # a full paper-trading cycle
```

## The mental model

Four objects. Learn these and the rest follows.

**`Panel`** is every symbol's OHLCV on one timeline. Each bar is stamped with
the UTC instant it *closed*, because that is the first moment its values are
knowable. An NSE daily bar closes at 15:30 IST, an NYSE one at 16:00 New York,
so a portfolio spanning both markets gets a correctly interleaved clock instead
of both pretending to close at midnight.

**`PanelView`** is bars `[0, t]` and is **the only thing a strategy ever sees**.
It physically cannot index past the current bar. Look-ahead through the data is
therefore a construction error, not something you check for afterwards.

**`Strategy`** turns a view into desired holdings. Weights are fractions of
equity: `0.25` means a quarter of the portfolio, `-0.25` means short that much,
`NaN` means "no opinion, leave it alone".

**`EngineConfig`** holds every execution assumption: costs, lot sizes, leverage
and participation caps, whether shorting is allowed. Two strategies compared
under the same config are genuinely comparable.

```
load_panel ──► Panel ──► PanelView(t) ──► Strategy ──► Orders ──► Fills ──► BacktestResult
                                 ▲                                  │
                                 └────── bar t+1 ───────────────────┘
```

A signal computed from bar `t`'s close fills at bar `t+1`'s open. That is the
default and it is not negotiable by accident: `FillTiming.SAME_CLOSE` exists for
market-on-close strategies and adds a WARN flag to every report that sees it.

## Writing a strategy

Three shapes. Pick the simplest one that expresses your idea.

### 1. Target weights, one bar at a time

The common case. Return what you want to hold.

```python
def equal_weight_winners(view: fbt.PanelView) -> dict[str, float]:
    momentum = view.close.iloc[-1] / view.close.iloc[-21] - 1.0
    winners = momentum[momentum > 0].index
    if len(winners) == 0:
        return {}                       # hold nothing
    return {s: 1 / len(winners) for s in winners}

strategy = fbt.RuleBasedStrategy(equal_weight_winners, warmup=21)
```

### 2. Target weights, all bars at once

Faster, and the shape most likely to leak the future. Row `t` must depend only
on rows `<= t`.

```python
def batch_momentum(view: fbt.PanelView) -> pd.DataFrame:
    signal = (view.close / view.close.shift(20) - 1.0 > 0).astype(float)
    return signal.div(signal.sum(axis=1).clip(lower=1), axis=0)

strategy = fbt.BatchRuleStrategy(batch_momentum, warmup=21)
```

Nothing stops you writing `.shift(-1)` here. Something *does* catch it: see
[Catching look-ahead](#catching-look-ahead).

### 3. Orders, when you need real control

Subclass `Strategy` when the decision depends on your fills, or you need limit
and stop orders.

```python
class BreakoutWithStop(fbt.Strategy):
    warmup = 20

    def on_bar(self, ctx: fbt.BarContext) -> None:
        for symbol in ctx.symbols:
            bars = ctx.view.bars(symbol)
            if ctx.position(symbol) == 0 and bars["close"].iloc[-1] > bars["high"].iloc[-20:-1].max():
                ctx.order_target_weight(symbol, 0.2)
                ctx.order(                                  # protective stop
                    symbol, -1,
                    order_type=fbt.OrderType.STOP,
                    stop_price=bars["close"].iloc[-1] * 0.92,
                    time_in_force=fbt.TimeInForce.GTC,
                )
```

Inside `on_bar` you get `ctx.view` (history up to now), `ctx.portfolio` (cash,
positions, weights), and the order API: `ctx.order`, `ctx.order_target_weight`,
`ctx.order_target_weights`, `ctx.cancel_all`.

### Machine learning

`WalkForwardMLStrategy` enforces the rule ML backtests break most often: a model
predicting at time `t` may only be trained on samples whose label window already
closed, plus an embargo.

```python
from sklearn.ensemble import GradientBoostingRegressor

strategy = fbt.WalkForwardMLStrategy(
    model_factory=lambda: GradientBoostingRegressor(),
    feature_fn=my_features,     # view -> frame indexed by (timestamp, symbol)
    horizon=5,                  # labels are 5-bar forward returns
    embargo=2,                  # plus 2 bars of separation
    retrain_every=20,
)
```

Every refit records `(fitted_at, label_cutoff, n_samples)` in
`strategy.training_log`, so you can *prove* the rule held rather than trusting it.

### If your strategy holds state

Implement `reset()`. Engines call it before the first bar, and validation re-runs
your strategy several times on perturbed data.

## Getting data

```python
panel = fbt.load_panel(symbols, start, end, market="IN")           # Yahoo, adjusted
panel = fbt.load_panel(symbols, start, end, market="US")
panel = fbt.load_panel(["BTC-USD"], start, end, market="CRYPTO")
panel = fbt.load_panel(symbols, start, end, market="US",
                       frequency=fbt.Frequency.HOUR_1)             # intraday
```

| Source | Markets | Adjusted | Survivorship-free |
|---|---|---|---|
| `yfinance` (default) | US, IN, CRYPTO | yes | **no** |
| `nse` | IN, daily | **no** | **yes** |
| `local` | any | you declare it | no |

Everything is cached to Parquet keyed by source, market, frequency and symbol, so
research is reproducible and the second run does not hit the network.

**Local files**, one per symbol or one long file:

```python
panel = fbt.load_panel(["AAPL"], start, end, market="US",
                       source=fbt.LocalSource("data/"))   # data/AAPL.csv
```

Column names are matched case-insensitively; naive dates become the market's
session close.

**Survivorship-free Indian universes** come from the exchange itself:

```python
nse = fbt.NSEBhavcopySource()
universe = nse.list_symbols(date(2020, 1, 1))   # every EQ/BE symbol listed that day
```

Building your universe from *today's* index is hindsight. This is the fix, at the
cost of unadjusted prices; the report tells you so.

**Check what you got.** It takes one line and it is not optional:

```python
for flag in fbt.check_data_quality(panel):
    print(flag)
```

On the real NSE pull above, this found three sessions with no data and six bars
on days no calendar lists as sessions. The six turned out to be Diwali Muhurat
trading, which is real. The three were genuine holes. "Financial data is clean"
is a belief that survives only until you check.

## Making execution realistic

Every assumption lives in one object, so you can hold it constant across
strategies or vary it deliberately.

```python
config = fbt.EngineConfig(
    initial_cash=1_000_000,
    cost_model=fbt.CostModel(
        commission=fbt.BpsCommission(5.0, minimum=20.0),
        slippage=fbt.VolumeShareSlippage(impact_coefficient=0.1),
    ),
    fill_timing=fbt.FillTiming.NEXT_OPEN,   # default
    fractional_shares=False,
    lot_size=1,
    allow_short=False,
    max_gross_leverage=1.0,   # fully funded
    max_participation=0.02,   # at most 2% of a bar's volume
)
```

Commission: `BpsCommission`, `PerUnitCommission`, `ZeroCommission`.
Slippage: `FixedSlippage`, `VolumeShareSlippage` (quadratic in participation),
`ZeroSlippage`.

**Always test cost sensitivity.** It is two lines and it is the fastest way to
find out your edge is imaginary:

```python
for bps in (0, 5, 25):
    result = fbt.EventDrivenEngine(
        fbt.EngineConfig(cost_model=fbt.CostModel.bps(bps, bps // 2))
    ).run(strategy, panel)
    print(bps, round(result.metrics().sharpe, 2))
```

### Two engines, on purpose

`VectorizedEngine` is fast and idealized. `EventDrivenEngine` keeps a real cash
ledger and applies every constraint. Under `EngineConfig.idealized()` they agree
to **1e-9**, enforced by a property test. Under realistic settings, the gap
between them is a measurement:

```python
comparison = fbt.compare_engines(strategy, panel, config)
print(comparison.max_equity_gap, comparison.sharpe_gap)
for flag in comparison.flags():
    print(flag)
```

A large gap means your strategy is sensitive to execution assumptions, which is
worth knowing before it is worth trading.

## Catching look-ahead

Three checks, each blind to what the others catch. Run all of them:

```python
report = fbt.validate(strategy, panel)
print(report.summary())
```

**Static scan** reads your source for `.shift(-n)`, `rolling(center=True)`,
backward fill, and `.fit()` before a split. Fast, exact line numbers, blind to
anything not shaped like those patterns.

**Truncation test** (batch strategies) recomputes weights on history cut at `t`.
If row `t` changes when later rows are removed, it was reading them. Catches
centred windows, full-sample normalisation, and models fit on everything.

**Future-noise test** (any strategy) re-runs the engine with all bars after `t`
replaced by random noise. The orders at `t` must be identical. Bars up to `t` are
byte-identical between runs, so any difference means the strategy looked ahead.

Here is what it looks like when it catches something real:

```
validation score: 14/100 (4 flags)
  [high] static: .shift(-1) moves data backward in time: row t sees row t+1 (strategy.py:27)
  [high] perturbation: batch weights at 2019-10-10 change when later bars are removed
  [high] perturbation: batch weights at 2021-07-02 change when later bars are removed
```

The score is a summary, not a verdict. **Read the flags.** A clean report means
nothing these checks know how to look for was found, which is not the same as
"no leakage" — none of them can see a strategy reading a global variable or a
file it was never handed.

## Comparing strategies honestly

`Arena` runs every entry on the same data, the same costs, the same checks, and
then applies the statistic almost nobody applies: it counts how many things you
tried.

```python
arena = fbt.Arena(panel, config)
arena.add(fbt.RuleBasedStrategy(momentum, warmup=126), "momentum")
arena.add(fbt.RuleBasedStrategy(reversal, warmup=6), "reversal")
arena.add(fbt.RuleBasedStrategy(equal_weight, warmup=1), "benchmark")

result = arena.run()
print(result.summary())
print(f"PBO: {result.pbo().pbo:.2f}")
print(f"best: {result.best('deflated_sharpe')}")
```

Beyond the usual metrics you get:

- **Deflated Sharpe Ratio** — the probability your Sharpe is real *given that you
  tried this many strategies*. A Sharpe of 1.5 picked from fifty attempts is
  worth less than a Sharpe of 1.0 picked from two, and this says by how much.
- **Probability of Backtest Overfitting** — combinatorial cross-validation. How
  often does the in-sample winner underperform out of sample?
- **Engine gap** — how much of the result depends on optimistic execution.
- **Validation score** with every HIGH finding printed underneath.

Always add a dumb benchmark. Equal weight beat both real strategies in the table
at the top of this README, and you only learn that if you include it.

## Paper trading

The same `on_bar`, driven by a clock instead of a file. Because it is the same
computation, live and backtest results are comparable by construction.

```python
# Register once. Strategies and costs are fingerprinted here.
fbt.PaperSession.create(
    "sessions/momentum.db",
    name="momentum",
    strategies={"momentum_6m": fbt.RuleBasedStrategy(momentum, warmup=126)},
    symbols=["RELIANCE", "TCS", "INFY"],
    market="IN",
    config=config,
)
```

Then once a day after the close, from Task Scheduler or cron:

```python
session = fbt.PaperSession.open("sessions/momentum.db", strategies=..., config=config)
print(session.step())
```

State lives in SQLite, so a reboot costs nothing and each run is idempotent. Off
for three days? The next run replays those three bars in order, exactly as the
backtester would.

Three properties worth understanding:

**Bars are written once and never rewritten.** When a provider later restates a
close you already traded on, the original stays and the change is logged with a
WARN. Otherwise your record silently becomes a log of decisions nobody made.

**The session locks to the code it was registered with.** Edit the strategy or
soften the costs and it raises `SessionLockError`. Retuning mid-run while
claiming a continuous live record is the most common way people fool themselves
with paper trading. Start a new session instead.

**Two comparisons, answering different questions:**

```python
result = session.result("momentum_6m")

fbt.replay_check(result, strategy, session.panel(), config)  # must agree
fbt.expectation_gap(result, earlier_backtest)                # expected to differ
```

`replay_check` re-runs the strategy over the session's own bars. Disagreement is
a bug in the machinery. `expectation_gap` compares live against the backtest that
convinced you to trade. A large negative gap is the signal you paper traded to
get: the edge you measured is not the edge you are getting.

Full cycle offline: `python examples/paper_trading.py`.

## Metrics reference

```python
metrics = result.metrics()
```

| | |
|---|---|
| Returns | `total_return`, `cagr`, `annual_volatility` |
| Risk-adjusted | `sharpe`, `sortino`, `calmar` |
| Drawdown | `max_drawdown`, `max_drawdown_bars` |
| Distribution | `hit_rate`, `best_bar`, `worst_bar`, `skew`, `excess_kurtosis` |
| Trading | `avg_turnover`, `avg_gross_exposure`, `n_fills`, `total_costs`, `cost_drag` |

Two conventions that differ from most libraries:

**Annualization comes from the data**, not a constant. 252 sessions for the US,
250 for India, 365 for crypto, and the correct multiple for any intraday
frequency. A `252` hard-coded into a Sharpe is wrong for most of the world.

**Undefined metrics are `NaN`, never `0`.** The Sharpe of a constant series is
undefined, not zero, and a flat strategy should not be able to masquerade as a
neutral one.

Selection-aware statistics live in `fbt.metrics`:
`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_backtest_length`,
`probability_of_backtest_overfitting`.

## Extending it

**A data source**: subclass `DataSource`, declare `markets`, `frequencies`,
`adjusted`, `survivorship_free`, implement `fetch` returning canonical bars with
UTC close timestamps, then `register` it. The declarations become flags in every
report that uses it, so a source cannot hide what it is.

**A cost model**: implement `commission(quantity, price)` or
`slippage_bps(participation)` with NumPy-compatible maths, so both engines share
the identical arithmetic.

**A leakage check**: return `list[Flag]` and wire it into `validate()`.

Docstrings throughout state what each check does **and does not** catch. That
convention is deliberate; please keep it.

## Limitations

Read these before trusting a number.

- **Corporate actions are not point-in-time.** You trade on whatever adjusted
  series your source provides. NSE bhavcopy is unadjusted and says so.
- **Free data is survivorship-biased.** Yahoo's universe is what exists today.
  Use `NSEBhavcopySource.list_symbols` for a real point-in-time universe.
- **The leakage tests are incomplete by construction.** They cannot see a
  strategy reading a global, a file, or the network. Nothing can.
- **Intraday fills** assume the bar's open is attainable and use bar ranges, not
  tick data, for limit and stop decisions.
- **Batch strategies on the event-driven engine** are re-evaluated per bar on a
  growing window, which is quadratic in bars. Fine for years of daily data; write
  a per-bar `WeightStrategy` for long intraday histories.
- **No options, futures margining, borrow costs, or FX conversion.**

## Development

```bash
pip install -e ".[dev,all]"
pytest                                    # 100 tests, incl. hypothesis property tests
ruff check src tests examples scripts
mypy
./scripts/clean_room_test.ps1             # build, then test the installed wheel
```

`scripts/verify_install.py` exercises the *installed* package with the source
tree off `sys.path`, which catches what a test run cannot: a module missing from
the wheel, a lost `py.typed`, a stale `__all__` entry.

Releasing is a GitHub Release tagged `v<version>`; the workflow publishes via PyPI
Trusted Publishing and refuses if the tag and `pyproject.toml` disagree.

`DECISIONS.md` records every architectural decision with the argument that was
had, not just the outcome. Start there if you want to know why something is the
way it is.

## License

MIT.
