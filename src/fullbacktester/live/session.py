"""Paper trading: the backtester's bar loop, driven by a clock instead of a file.

A session runs one or more strategies forward on live data under exactly the
execution rules a backtest would apply. Nothing about the strategy changes; the
same ``on_bar`` that a backtest calls is what runs here.

Each invocation is a separate process:

1. Fetch recent bars from the configured source.
2. Record any bar not seen before. Bars already stored are never rewritten, and
   a provider that restates one has the disagreement logged instead.
3. Build a ``Panel`` from the bars *as first seen*, then process every bar after
   the last one processed, in order. A machine that was off for three days
   replays those three bars exactly as the backtester would have.
4. Persist cash, positions, orders, fills, and equity.

The session refuses to run if the strategy source or the execution config has
changed since it was registered. That is deliberate: a paper record is only
evidence if the thing being measured held still. To change a strategy, start a
new session.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fullbacktester.data.loader import load_bars, resolve_source
from fullbacktester.data.panel import Panel
from fullbacktester.data.sources.base import DataSource
from fullbacktester.execution.config import EngineConfig, FillTiming
from fullbacktester.execution.fills import FillModel
from fullbacktester.execution.ledger import Ledger
from fullbacktester.execution.orders import Order, OrderStatus, OrderType
from fullbacktester.flags import Flag, Severity
from fullbacktester.live.store import LiveStore, SessionSpec, StrategyRecord
from fullbacktester.markets import Frequency, Market, get_market
from fullbacktester.result import BacktestResult
from fullbacktester.strategy.base import Strategy
from fullbacktester.strategy.context import BarContext


def strategy_fingerprint(strategy: Strategy) -> str:
    """Hash of the strategy's own logic, used to detect edits mid-session.

    Hashes the source of everything ``scan_targets`` exposes, which is the same
    set of callables the leakage scanner reads, plus the class name and warmup.
    Sources that cannot be read (a lambda built with ``eval``, a C extension)
    contribute a marker instead, so such a strategy is simply not protected.
    """
    parts = [type(strategy).__name__, f"warmup={strategy.warmup}"]
    for target in strategy.scan_targets():
        try:
            parts.append(inspect.getsource(target))
        except (OSError, TypeError):
            parts.append(f"<unreadable:{getattr(target, '__name__', repr(target))}>")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def config_fingerprint(config: EngineConfig) -> str:
    """Hash of the execution assumptions, so costs cannot be softened mid-session."""
    parts = [
        f"cash={config.initial_cash!r}",
        f"timing={config.fill_timing.value}",
        f"fractional={config.fractional_shares}",
        f"lot={config.lot_size!r}",
        f"short={config.allow_short}",
        f"leverage={config.max_gross_leverage!r}",
        f"participation={config.max_participation!r}",
        f"commission={config.cost_model.commission!r}",
        f"slippage={config.cost_model.slippage!r}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class SessionLockError(RuntimeError):
    """Raised when the registered strategies or config no longer match the code supplied."""


@dataclass
class StepReport:
    """What one invocation did."""

    ran_at: pd.Timestamp
    bars_added: int
    bars_revised: int
    bars_processed: int
    processed_through: pd.Timestamp | None
    orders_placed: dict[str, int] = field(default_factory=dict)
    fills: dict[str, int] = field(default_factory=dict)
    equity: dict[str, float] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)

    def __str__(self) -> str:
        through = "nothing yet" if self.processed_through is None else str(self.processed_through)
        lines = [
            f"step at {self.ran_at:%Y-%m-%d %H:%M %Z}: +{self.bars_added} bars, "
            f"{self.bars_processed} processed, through {through}"
        ]
        for name, value in self.equity.items():
            lines.append(
                f"  {name}: equity {value:,.2f}, "
                f"{self.orders_placed.get(name, 0)} order(s), {self.fills.get(name, 0)} fill(s)"
            )
        lines.extend(f"  {flag}" for flag in self.flags)
        return "\n".join(lines)


class PaperSession:
    """A live paper-trading session over one shared bar stream.

    Create it once with :meth:`create`, then reopen it with :meth:`open` on every
    scheduled run and call :meth:`step`.
    """

    def __init__(
        self,
        store: LiveStore,
        spec: SessionSpec,
        strategies: Mapping[str, Strategy],
        config: EngineConfig,
        source: DataSource | None = None,
        max_replay_bars: int = 30,
    ) -> None:
        self.store = store
        self.spec = spec
        self.strategies = dict(strategies)
        self.config = config
        self.market: Market = get_market(spec.market)
        self._source = source
        self.max_replay_bars = max_replay_bars

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def create(
        cls,
        path: Path | str,
        *,
        name: str,
        strategies: Mapping[str, Strategy],
        symbols: Sequence[str],
        market: str | Market,
        config: EngineConfig,
        frequency: Frequency = Frequency.DAY_1,
        source: str | DataSource | None = None,
        now: pd.Timestamp | None = None,
    ) -> PaperSession:
        """Register a new session. Fails if one already exists at ``path``."""
        store = LiveStore(path)
        if store.exists:
            raise FileExistsError(
                f"{store.path} already holds a session; open it instead of creating it"
            )
        if not strategies:
            raise ValueError("a session needs at least one strategy")
        resolved_market = get_market(market)
        source_name = source.name if isinstance(source, DataSource) else source
        spec = SessionSpec(
            name=name,
            market=resolved_market.code,
            frequency=frequency,
            symbols=tuple(str(s) for s in symbols),
            source=source_name,
            config_hash=config_fingerprint(config),
            created_at=_utc_now(now),
        )
        records = [
            StrategyRecord(
                name=label,
                code_hash=strategy_fingerprint(strategy),
                warmup=strategy.warmup,
                registered_at=spec.created_at,
            )
            for label, strategy in strategies.items()
        ]
        store.initialise(spec, records)
        return cls(
            store=store,
            spec=spec,
            strategies=strategies,
            config=config,
            source=source if isinstance(source, DataSource) else None,
        )

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        strategies: Mapping[str, Strategy],
        config: EngineConfig,
        source: DataSource | None = None,
        max_replay_bars: int = 30,
    ) -> PaperSession:
        """Reopen an existing session, verifying nothing has been edited since."""
        store = LiveStore(path)
        if not store.exists:
            raise FileNotFoundError(f"no session at {store.path}")
        spec = store.load_spec()
        registered = store.load_strategies()

        if config_fingerprint(config) != spec.config_hash:
            raise SessionLockError(
                "the execution config differs from the one this session was registered with. "
                "Changing costs or constraints mid-session invalidates the record; start a "
                "new session instead."
            )
        missing = sorted(set(registered) - set(strategies))
        added = sorted(set(strategies) - set(registered))
        if missing or added:
            raise SessionLockError(
                f"strategy set differs from registration (missing {missing}, unexpected {added}). "
                "Start a new session to change which strategies are being measured."
            )
        for label, strategy in strategies.items():
            actual = strategy_fingerprint(strategy)
            if actual != registered[label].code_hash:
                raise SessionLockError(
                    f"strategy {label!r} has changed since it was registered "
                    f"({registered[label].code_hash} -> {actual}). A paper record only means "
                    "something if the strategy held still; start a new session."
                )
        return cls(
            store=store,
            spec=spec,
            strategies=strategies,
            config=config,
            source=source,
            max_replay_bars=max_replay_bars,
        )

    # ------------------------------------------------------------------ step

    @property
    def source(self) -> DataSource:
        if self._source is None:
            self._source = resolve_source(self.spec.source, self.market)
        return self._source

    def fetch(self, now: pd.Timestamp, lookback_days: int = 10) -> pd.DataFrame:
        """Recent bars from the source. Overlaps deliberately so gaps self-heal."""
        end: date = now.tz_convert(self.market.tz).date()
        start = end - timedelta(days=lookback_days)
        return load_bars(
            self.spec.symbols,
            start,
            end,
            market=self.market,
            frequency=self.spec.frequency,
            source=self.source,
            cache=False,
        )

    def step(self, now: pd.Timestamp | None = None, bars: pd.DataFrame | None = None) -> StepReport:
        """Ingest new bars and run every strategy over each unprocessed bar in order.

        ``bars`` bypasses the network and is what the tests use; leaving it None
        fetches from the session's source.
        """
        ran_at = _utc_now(now)
        flags: list[Flag] = []

        incoming = self.fetch(ran_at) if bars is None else bars
        added, revised = self.store.record_bars(incoming, ran_at)
        if revised:
            flags.append(
                Flag(
                    source="live",
                    severity=Severity.WARN,
                    message=(
                        f"{revised} field(s) in already-recorded bars now differ at the source. "
                        "The originals were kept, since they are what the strategies traded on. "
                        "See LiveStore.revisions()"
                    ),
                )
            )

        observed = self.store.observed_bars()
        if observed.empty:
            self.store.record_step(ran_at, added, revised, 0, None, "no bars yet")
            return StepReport(ran_at, added, revised, 0, None, flags=flags)

        panel = Panel.from_bars(
            observed, frequency=self.spec.frequency, market=self.market, symbols=self.spec.symbols
        )
        processed_through = self.store.processed_through()
        start_index = 0 if processed_through is None else panel.index_of(processed_through)
        pending_indices = list(range(start_index, len(panel)))

        if len(pending_indices) > self.max_replay_bars:
            skipped = len(pending_indices) - self.max_replay_bars
            pending_indices = pending_indices[-self.max_replay_bars :]
            flags.append(
                Flag(
                    source="live",
                    severity=Severity.WARN,
                    message=(
                        f"{skipped} unprocessed bar(s) skipped: more than max_replay_bars="
                        f"{self.max_replay_bars} accumulated. The equity curve has a hole; "
                        "raise max_replay_bars or start a new session"
                    ),
                )
            )

        report = StepReport(
            ran_at=ran_at,
            bars_added=added,
            bars_revised=revised,
            bars_processed=len(pending_indices),
            processed_through=panel.timestamps[-1] if pending_indices else processed_through,
            flags=flags,
        )
        if not pending_indices:
            self.store.record_step(ran_at, added, revised, 0, processed_through, "nothing new")
            return report

        for label, strategy in self.strategies.items():
            self._advance(label, strategy, panel, pending_indices, report)

        self.store.record_step(
            ran_at, added, revised, len(pending_indices), report.processed_through
        )
        return report

    def _advance(
        self,
        label: str,
        strategy: Strategy,
        panel: Panel,
        indices: Sequence[int],
        report: StepReport,
    ) -> None:
        cash, positions, pending_ids = self.store.load_state(label)
        ledger = Ledger(panel.symbols, self.config.initial_cash)
        if cash is not None:
            ledger.cash = cash
            for symbol, quantity in positions.items():
                ledger.positions[ledger.index_of(symbol)] = quantity

        pending = self.store.load_orders(label, pending_ids)
        fill_model = FillModel(self.config)
        touched: dict[int, Order] = {order.id: order for order in pending}
        new_fills = []
        equity_points: list[tuple[pd.Timestamp, float, float]] = []
        orders_placed = 0

        marks = _marks_before(panel, int(indices[0]))

        for i in indices:
            if pending:
                filled, pending = fill_model.fill_bar(pending, panel, i, ledger, marks)
                new_fills.extend(filled)

            close = panel.close[i]
            marks = np.where(np.isfinite(close), close, marks)

            if i >= strategy.warmup:
                snapshot = ledger.snapshot(panel.timestamps[i], i, marks)
                ctx = BarContext(panel.view(i + 1), snapshot, self.config, marks)
                strategy.on_bar(ctx)
                if ctx.cancel_requested:
                    for order in pending:
                        order.status = OrderStatus.CANCELLED
                        order.reason = "cancelled by strategy"
                        touched[order.id] = order
                    pending = []
                submitted = ctx.orders
                orders_placed += len(submitted)
                for order in submitted:
                    touched[order.id] = order
                live = [order for order in submitted if order.is_open]
                if self.config.fill_timing is FillTiming.SAME_CLOSE:
                    now_orders = [o for o in live if o.order_type is OrderType.MARKET]
                    later = [o for o in live if o.order_type is not OrderType.MARKET]
                    if now_orders:
                        filled, still = fill_model.fill_bar(
                            now_orders, panel, i, ledger, marks, at_close=True
                        )
                        new_fills.extend(filled)
                        later.extend(still)
                    pending.extend(later)
                else:
                    pending.extend(live)

            equity_points.append((panel.timestamps[i], ledger.cash, ledger.equity(marks)))

        for order in pending:
            touched[order.id] = order

        self.store.save_state(
            strategy=label,
            cash=ledger.cash,
            positions={s: float(q) for s, q in zip(panel.symbols, ledger.positions, strict=True)},
            pending=pending,
            touched_orders=touched.values(),
            new_fills=new_fills,
            equity_points=equity_points,
        )
        report.orders_placed[label] = orders_placed
        report.fills[label] = len(new_fills)
        report.equity[label] = equity_points[-1][2] if equity_points else float("nan")

    # ---------------------------------------------------------------- report

    def result(self, label: str) -> BacktestResult:
        """The live record as a ``BacktestResult``, so every backtest metric applies."""
        equity = self.store.equity_frame(label)
        if equity.empty:
            raise ValueError(f"strategy {label!r} has no recorded equity yet")
        index = pd.DatetimeIndex(equity["timestamp"])
        panel = Panel.from_bars(
            self.store.observed_bars(),
            frequency=self.spec.frequency,
            market=self.market,
            symbols=self.spec.symbols,
        )
        fills = self.store.fills_frame(label)
        positions = pd.DataFrame(0.0, index=index, columns=list(panel.symbols))
        if not fills.empty:
            traded = fills.pivot_table(
                index="timestamp", columns="symbol", values="quantity", aggfunc="sum"
            )
            positions = (
                traded.reindex(index=index, columns=list(panel.symbols)).fillna(0.0).cumsum()
            )
        closes = pd.DataFrame(panel.close, index=panel.timestamps, columns=list(panel.symbols))
        closes = closes.reindex(index).ffill()
        equity_series = pd.Series(equity["equity"].to_numpy(), index=index, name="equity")
        weights = (positions * closes).div(equity_series, axis=0).fillna(0.0)
        return BacktestResult(
            engine="paper",
            strategy_name=label,
            config=self.config,
            equity=equity_series,
            cash=pd.Series(equity["cash"].to_numpy(), index=index, name="cash"),
            positions=positions,
            weights=weights,
            fills=fills,
            orders=self.store.orders_frame(label),
            bars_per_year=panel.bars_per_year,
            flags=[],
        )

    def panel(self) -> Panel:
        """The bars this session actually traded on, as first seen."""
        return Panel.from_bars(
            self.store.observed_bars(),
            frequency=self.spec.frequency,
            market=self.market,
            symbols=self.spec.symbols,
        )


def _marks_before(panel: Panel, index: int) -> np.ndarray:
    """Last known close for each symbol strictly before ``index``."""
    marks = np.full(panel.n_symbols, np.nan)
    if index <= 0:
        return marks
    history = panel.close[:index]
    for j in range(panel.n_symbols):
        column = history[:, j]
        seen = np.flatnonzero(np.isfinite(column))
        if seen.size:
            marks[j] = column[seen[-1]]
    return marks


def _utc_now(now: pd.Timestamp | None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")
    stamp = pd.Timestamp(now)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
