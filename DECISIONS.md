# Decision Log

Running record of every architectural decision, the competing positions that were argued,
and why the resolution won. Entries are appended, never rewritten; a reversed decision gets
a new entry that references the old one.

Format per entry: **Context**, **Positions** (labelled A/B/C so the argument is traceable),
**Resolution**, **Consequences**.

---

## D-001: Audit of the inherited scaffold and what survives

**Date:** 2026-09-06

**Context.** The repository arrived with a ~450-line scaffold under `src/ubt/`: interface
classes, two "engines" sharing one computation, a Sharpe-only result type, an AST leakage
scanner, and a perturbation-tester stub. Docstrings contradicted the code (the package
docstring called the engines stubs; the README called that docstring the source of truth).
The editable install pointed at a folder that no longer existed, so the test suite could
not import the package. No `.gitignore`.

**Positions.**
- *A (preserve):* the interfaces are reasonable, keep them and grow the engines in place.
  Continuity is cheaper than a rewrite and the tests already pass.
- *B (rewrite):* the engine has no cash, positions, fills, slippage, or OHLC awareness, and
  re-slices the full history on every bar. Growing that into a real engine means replacing
  every line anyway while carrying stale docstrings and a placeholder name. The parts worth
  keeping are *ideas*, not code.

**Resolution.** B. Keep four ideas and re-implement them: (1) a single point-in-time
chokepoint for data access; (2) two execution engines whose divergence is itself a
diagnostic; (3) static AST scanning plus behavioral perturbation as complementary leakage
checks; (4) a validation report made of typed flags. Port the scanner's AST logic and its
tests nearly verbatim because they were correct. Delete everything else, including the
`ubt` name. Overrides, each with the reason it was wrong:

| Overridden | Why |
|---|---|
| `_equity_from_weights` accounting | No cash, no share quantities, no fill prices, costs on weight deltas instead of traded notional. Cannot represent a partial fill, a stop, or an unaffordable order. |
| `MLStrategy(model).predict()` per bar returning a symbol->weight dict | Bakes the model's output contract into the engine, has no notion of training window, purge, or embargo, so the one place ML leaks most (labels overlapping the prediction time) was unguarded. |
| Sharpe with `252` hard-coded | Wrong for India (~250 sessions), crypto (365), and every intraday frequency. Annualization now comes from the panel's market and frequency. |
| String-typed severities and engine names | Typos become silent data corruption. Enums. |
| `engine_spread_pct` | Not a percentage. Replaced by an `EngineComparison` with named statistics. |
| Package docstring as "source of truth" | Documentation that must be hand-synchronized with code drifts within a week. The module docstrings now describe contracts, and the tests assert behavior. |

**Consequences.** Old scaffold removed in one commit. Git history keeps it.

---

## D-002: Execution model, hybrid with cross-check

**Date:** 2026-09-06 (confirmed with the project owner)

**Context.** A backtester needs to be fast enough to sweep parameters and realistic enough
to price stops, partial fills, and capital constraints. Those pull in opposite directions.

**Positions.**
- *A (event-driven only):* one loop, one ledger, one source of truth. Every fill is an
  explicit object with a price and a cost. Path dependence is natural. Cost: Python-loop
  speed, so a 2,000-run parameter sweep on daily data takes minutes, on intraday data hours.
- *B (vectorized only):* signals and P&L as array operations, hundreds of times faster.
  Cost: cannot honestly represent anything that depends on the path, which includes stop
  losses, limit orders, volume-capped fills, and running out of cash. Vectorized engines are
  where most published "too good to be true" results come from.
- *C (hybrid):* one `Strategy` contract that both engines can run. Use the vectorized engine
  for research sweeps and the event-driven engine for the final number. Report the gap
  between them as an *implementation-risk* statistic: when a strategy's idealized and
  realistic P&L diverge, the strategy is sensitive to execution assumptions.

**Resolution.** C, with two hard requirements that make it more than a slogan:
1. Under idealized settings (fractional shares, no leverage cap, no participation cap,
   next-open fills) the two engines must agree to floating-point precision. That equivalence
   is a property test, not a hope. Any future change that breaks it is a bug in one engine.
2. Strategies that need order-level control (limit/stop orders, path-dependent sizing)
   implement `Strategy.on_bar` and run only on the event-driven engine. The vectorized engine
   refuses them with an explicit error rather than approximating.

**Consequences.** The vectorized engine is not "fully vectorized" over time (see D-007);
its speed comes from batching signal computation and skipping order objects. Two engines
means two code paths to keep honest; the equivalence test is the cost of the design.

---

## D-003: Data layer, pluggable sources behind one canonical schema

**Date:** 2026-09-06 (confirmed with the project owner: US + India first, as many markets
as data quality allows, end user picks the source at runtime)

**Context.** Free data (yfinance) is convenient and survivorship-biased; official exchange
archives (NSE bhavcopy) are complete but unadjusted for corporate actions and slow to
fetch; paid feeds are point-in-time but cost money. No single source is right for every
market or every question.

**Positions.**
- *A (one blessed source):* fewer adapters, fewer bugs, one schema. Breaks the moment the
  user wants Indian equities from the official archive or crypto from an exchange.
- *B (adapter per source, engine consumes any DataFrame):* maximum flexibility, but every
  adapter invents its own column names and timestamp conventions, and the engine ends up
  with `if "Adj Close" in df.columns` branches. That is how look-ahead sneaks in: a bar
  stamped with its open time is a bar the engine will consider known 6.5 hours early.
- *C (adapters behind one validated schema and one point-in-time container):* every source
  emits the same long frame (`timestamp, symbol, open, high, low, close, volume`) with
  `timestamp` = UTC instant of bar *close*. `validate_bars` rejects anything else. A
  `Panel` aligns symbols on one timeline; a `PanelView` is the only thing a strategy sees
  and physically ends at the current bar.

**Resolution.** C. Additional rulings inside it:
- **Bar timestamp = close instant, in UTC.** A bar's values become knowable at its close.
  Daily bars from an exchange are stamped at that exchange's session close (NYSE 16:00
  America/New_York, NSE 15:30 Asia/Kolkata), so a portfolio spanning both markets has a
  correct interleaved clock instead of pretending both closed at midnight.
- **Per-market source registry with defaults.** `US -> yfinance`, `IN -> yfinance` (adjusted
  prices, needed for correct returns), with `nse` (official bhavcopy, unadjusted,
  survivorship-free) registered as an explicit alternative. The end user overrides per call.
- **Parquet cache is canonical local storage** regardless of source, keyed by
  `source/market/frequency/symbol`. Reproducible research reads the cache, never the network.
- **Data quality report** runs on every load: missing sessions, zero volume, suspected
  unadjusted splits, survivorship flag inherited from the source.
- **Read-only arrays.** Panel arrays are flagged non-writable so a strategy that mutates the
  frame it was handed fails loudly instead of corrupting the data for every other strategy.

**Consequences.** Adding a source is one class implementing `DataSource.fetch`. The schema
validator is deliberately strict (it raises on a single NaN close) because a source that
emits garbage should be fixed at the source, not tolerated downstream. Adjustment policy is
recorded in D-008.

---

## D-004: Portfolio-first accounting, daily and intraday bars from day one

**Date:** 2026-09-06 (confirmed with the project owner)

**Positions.**
- *A (single asset first):* simplest possible ledger, fastest to a demo. Retrofitting a
  multi-asset ledger later means rewriting position, cash, and weight logic.
- *B (portfolio-first):* a single asset is a portfolio of one. Cross-sectional strategies
  (rank, long/short) are the ones most exposed to survivorship bias and need the multi-asset
  design to test at all.
- On frequency: daily-only calendars are simple, but intraday retrofits touch the meaning of
  every timestamp. Designing the close-instant convention up front (D-003) makes intraday a
  data question, not an engine question.

**Resolution.** B, with bars of any frequency the `Frequency` enum names. Symbols with
different calendars coexist on one panel: a symbol with no bar at a tick has NaN prices there
and is marked at its last known close.

---

## D-005: Runtime, Python 3.11+, synchronous core

**Date:** 2026-09-06 (confirmed with the project owner)

**Positions.**
- *A (async core):* a live-trading adapter could share the engine loop.
- *B (sync core):* a backtest is CPU-bound; `async` adds no throughput, complicates every
  strategy author's life, and makes the vectorized engine awkward. Live trading, if ever,
  wraps a sync engine in its own event loop.

**Resolution.** B. Python floor 3.11 for `Self`, `tomllib`, `ExceptionGroup`, and the
interpreter speedup. The inherited venv was 3.10; a 3.12 venv replaces it.

---

## D-006: Package name

**Date:** 2026-09-06

**Context.** `ubt` was a placeholder. PyPI check: `gauntlet` and `tribunal` taken;
`fullbacktester`, `strictbt`, `veribt` available.

**Resolution.** `fullbacktester`, import alias `fbt`. It says what the package does
(strategies run the gauntlet) and is brandable. Provisional until the owner objects; a
rename is a mechanical find-and-replace over `src/`, `tests/`, and `pyproject.toml`.

---

## D-007: Vectorized engine as a time-scan, not a pure array formula

**Date:** 2026-09-06

**Context.** The textbook vectorized backtest is `equity = cumprod(1 + (w.shift(1) * r).sum(1) - turnover * cost)`.

**Positions.**
- *A (pure formula):* one line, maximally fast. Wrong in two ways that matter: (1) the
  weight that drifts between rebalances depends on the equity path, which depends on costs
  already paid, so turnover is a recursion, not a shift; (2) with next-open fills, the return
  a weight earns is split into an overnight leg (close to next open, earned by the *old*
  weights) and an intraday leg (open to close, earned by the *new* weights). The formula
  attributes both legs to one weight vector.
- *B (T-step NumPy scan):* loop over bars (T iterations), vectorize over symbols (N). Each
  step does exactly the arithmetic the ledger does: drift positions to the open, rebalance to
  target notional, pay costs from cash, mark to close. For T = 2,500 daily bars this is
  milliseconds; the signal computation, which is what actually costs time, is still batched.

**Resolution.** B. The engines are then *exactly* comparable (D-002 requirement 1). If a
JIT ever earns its dependency, this scan is the function to compile, which is the only place
`numba` would have belonged in the original scaffold.

---

## D-008: Price adjustment policy

**Date:** 2026-09-06

**Context.** Splits and dividends change quoted prices without changing wealth. Backtesting
on unadjusted prices manufactures fake -50% days; backtesting on dividend-adjusted prices
trades at prices that never printed.

**Positions.**
- *A (Zipline-style point-in-time adjustments):* store raw prices plus an adjustments table,
  apply as-of each bar. The gold standard, and a large build with its own data requirements
  (a corporate-actions feed) that the free sources do not provide.
- *B (trade on adjusted series, state it):* use total-return-adjusted OHLC where the source
  offers it (yfinance `auto_adjust=True`), accept that fill prices are the adjusted series,
  and flag sources that are unadjusted (NSE bhavcopy) so their P&L is read with that caveat.

**Resolution.** B for v1, with the data quality report flagging suspected unadjusted
corporate actions (single-bar moves beyond a threshold). A is a documented upgrade path and
the schema leaves room for it (an `adjustments` table keyed by symbol and effective date).

---

## D-009: Fill timing defaults to next bar open

**Date:** 2026-09-06

**Positions.**
- *A (same-bar close):* a signal computed from bar t's close fills at bar t's close. Common
  in tutorials, and impossible: the close is known only after the close.
- *B (next-bar open):* the first tradeable price after the information exists.
- *C (next-bar close):* conservative, but throws away a bar of information and does not match
  how anyone trades.

**Resolution.** B as the default. A remains available as an explicit option for strategies
that genuinely trade market-on-close auctions, and choosing it adds a WARN flag to the
validation report so the optimism is on record. Orders placed on the final bar of a panel
never fill and are reported as unfilled rather than silently executed.

---

## D-010: Leakage detection, three complementary checks with stated blind spots

**Date:** 2026-09-06

**Context.** No single test catches every way a backtest can read the future. The inherited
scaffold had an AST scanner and a sketch of a "truncated lookback" perturbation test.

**Positions.**
- *A (static only):* fast, deterministic, points at a line number. Blind to anything not shaped
  like its patterns and to leakage through data content.
- *B (behavioural only):* rerun the strategy with the future altered and see if decisions
  change. Catches any shape of leak that goes through the data handed to the strategy. Blind
  to leaks from outside that data (globals, files), costs a full run per sample, cannot point
  at a line.
- *C (the scaffold's truncated-lookback test: rerun with N, 2N, 4N bars of history):* measures
  sensitivity to window length, which is a stability property, not look-ahead. Dropped.

**Resolution.** A and B together, plus a third test specific to batch strategies:
1. **Static scanner** (line-level): negative `shift`, `rolling(center=True)`, backward fill,
   fit-before-split. New pattern (`bfill`) added; forward-index access (`iloc[t+1]`) still
   excluded for its false-positive rate.
2. **Truncation test** (batch strategies): `target_weights_batch` on the full history must
   agree at row `t` with the same call on history truncated at `t`. Catches centred windows,
   full-sample normalisation, models fit on everything.
3. **Future-noise test** (any strategy): run the event-driven engine on the real panel and on
   a copy whose bars after `t` are noise; the *decisions* at `t` must be identical. Decisions
   are the target weight or the explicit quantity, never the fill-resolved quantity, which
   legitimately differs once the execution price is perturbed (a bug found by the test suite).

Every check's docstring says what it cannot see, and the report lists which checks ran.
A perturbation test needs re-runnable strategies, so `Strategy.reset()` was added and both
engines call it.

---

## D-011: Engine disagreement as a look-ahead detector

**Date:** 2026-09-06

**Context.** Found while testing D-010. A batch strategy that reads the future (e.g. tomorrow's
return) does so on the vectorized engine, which evaluates the batch function once over the
full history. On the event-driven engine the same function is re-evaluated per bar on a
`PanelView` that ends at that bar, so the "future" column is NaN and the strategy goes flat.

**Positions.**
- *A:* make the event-driven engine also evaluate the batch function once (fast, consistent).
  That would let the leak through both engines and remove the structural guarantee.
- *B:* keep per-bar evaluation (quadratic in bars) and treat the divergence as a signal.

**Resolution.** B. The cost is acceptable for years of daily data and documented for long
intraday histories (write a per-bar `WeightStrategy` instead). The arena table reports the
event-driven numbers, so a leaky batch strategy shows a flat line, a huge engine gap, HIGH
static and truncation flags, and no way to look good.

---

## D-012: Selection-aware statistics in the arena

**Date:** 2026-09-06

**Context.** A Sharpe ratio is meaningless without the number of things tried.

**Resolution.** `Arena` computes, per entry, the Probabilistic Sharpe Ratio and the Deflated
Sharpe Ratio with `n_trials` = number of entries and the trials' Sharpe variance taken across
them, and offers Probability of Backtest Overfitting via CSCV over the entries' return matrix.
Implementation notes: per-block sums make each CSCV split O(blocks x trials); the OOS rank is
converted with `rank / (N + 1)` so the logit is finite at the extremes; PBO for a single
dataset varies widely because splits share blocks, so the test suite asserts on the mean
across seeds (0.49 for noise, 0.00 with one real edge in twelve). Formulas are cited in the
module docstring; inputs are per-bar Sharpe, not annualized.

---

## D-013: NSE India source

**Date:** 2026-09-06

**Context.** The owner wants the official NSE archive alongside Yahoo for Indian equities.

**Findings (verified live).** The archive serves two layouts: UDiFF
(`BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`, columns `TradDt, TckrSymb, SctySrs,
OpnPric, ...`) for recent dates and legacy (`cmDDMONYYYYbhav.csv.zip`, columns `SYMBOL,
SERIES, OPEN, ..., TIMESTAMP`) for history. A browser User-Agent and Referer are enough.

**Resolution.** One source class tries UDiFF then legacy per session date, caches each raw
day (and each confirmed non-session) on disk, filters to equity series `EQ`/`BE`, and exposes
`list_symbols(date)` for point-in-time universes. It declares `adjusted=False` and
`survivorship_free=True`, so every report using it says "unadjusted". Default for `IN`
stays `yfinance` because unadjusted prices produce wrong returns; `nse` is for universe
membership and volume checks until a corporate-actions table exists (D-008).

---

## D-014: Metric conventions

**Date:** 2026-09-06

- Undefined statistics are NaN (Sharpe of a constant series, Calmar with no drawdown), never
  zero, so a flat strategy cannot masquerade as a neutral one.
- Annualization comes from the panel: market sessions per year times bars per session, or the
  empirical bar count per year for mixed-market panels.
- Turnover per bar is traded notional over equity and can reach twice gross leverage on a
  full rotation.
- Cost drag is total costs over initial equity.
- The first bar's return is measured against initial cash.

---

## D-015: Walk-forward ML training rule

**Date:** 2026-09-06

**Resolution.** At decision bar `i` with label horizon `h` and embargo `e`, training samples
are those with timestamp index `<= i - h - e`; their labels use closes up to `i - e`, which are
inside the view. The model is refit every `retrain_every` bars on at most `train_window`
samples. Each refit appends a `TrainingRecord(fitted_at, label_cutoff, n_samples)` so a test
can assert the rule held on every refit. `PurgedKFold` applies the same idea offline:
training indices within `purge` of either edge of the test fold are dropped, plus `embargo`
after it. Feature/label functions receive the view only; the static scanner scans both.

---

## D-016: Package name, FullBacktester (supersedes D-006)

**Date:** 2026-09-07

**Context.** D-006 chose `quantgauntlet` as a provisional name, flagged as pending the
owner's confirmation. The owner chose `FullBacktester`. Nothing had been published, so the
name was still free to change at zero cost.

**Positions.**
- *A (`FullBacktester`, the owner's choice):* plain, searchable, says what the thing is.
- *B (keep `quantgauntlet`):* names the differentiator, which is the gauntlet of leakage and
  overfitting checks rather than breadth of features.

**Concern raised and overruled.** "Full" reads as a completeness claim, and the README's
Limitations section states the opposite in several places: no options, no futures margining,
no borrow costs, no FX, corporate actions not point-in-time. A package whose entire design
principle is refusing to overclaim carries some tension in the name. The owner confirmed the
choice, so A stands.

**Resolution.** A. Distribution name `FullBacktester`, preserving the owner's capitalization
on the PyPI page. Import name `fullbacktester`, lowercase, because module names must be
(PEP 8). Documentation alias `fbt`, replacing `qg`.

**Consequences.** Mechanical rename of 213 references across 46 files plus the package
directory. Nothing had been published to any index, so no compatibility shim is needed and
no version was burned. The Limitations section stays exactly as written; it now does more
work, because the name no longer does it.

---

## D-017: Session-gap detection compares dates, not counts

**Date:** 2026-09-08

**Context.** First run against real market data: fifteen NSE large caps, 2018-2024, pulled
through yfinance. `check_data_quality` reported no session problems at all. It was wrong.

**What real data exposed.** The check computed `missing = len(expected_sessions) - n_valid`.
The panel held 1727 bars against 1724 calendar sessions, so the subtraction gave -3 and
nothing was flagged. Comparing the date *sets* showed both a surplus and a deficit:

* Six bars on days `exchange_calendars` does not list as sessions: 2018-11-07, 2019-10-27,
  2020-11-14, 2021-11-04, 2022-10-24, 2024-11-01. These are NSE's annual Diwali Muhurat
  session, a one-hour ceremonial auction that twice fell on a weekend. They are real data.
* Three calendar sessions with no bar at all: 2019-02-13, 2019-03-29, 2024-01-20.

The surplus cancelled the deficit. A panel can be missing an entire week and report clean
so long as it carries a week of unexpected bars.

**Positions.**
- *A (keep counting, subtract):* one line, no allocation. Correct only when the data never
  contains a session the calendar omits, which India violates every single year.
- *B (compare date sets):* build both sets and difference them in each direction. Costs one
  set per symbol, catches the masking case, and can name the offending dates.

**Resolution.** B, reported as two distinct findings rather than one number. Absent sessions
are a WARN, since data is genuinely missing. Unscheduled bars are an INFO naming Muhurat
explicitly, since flagging a real exchange session as an error trains the reader to ignore
the report. Both list the actual dates; "3 session(s) missing" is unactionable, while
"2019-02-13, 2019-03-29, 2024-01-20" can be checked in a minute.

**Consequences.** Ships in 0.1.1. The regression test reproduces the exact cancellation,
three dropped sessions against three added weekend bars, which the old arithmetic could not
see. Worth recording that no synthetic fixture would ever have found this: the bug needed a
market whose real calendar disagrees with the reference calendar, which is precisely why the
real-data run was worth doing before adding features.
