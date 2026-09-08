"""Durable state for a paper-trading session.

One SQLite file per session. Every run is a separate process that opens the
file, does its work in a transaction, and exits; nothing is held in memory
between runs, so a reboot costs nothing.

The important table is ``bars``. A bar is written once, the first time it is
seen, and never updated. When a later fetch disagrees with a bar the session
has already traded on, the disagreement is recorded in ``bar_revisions`` and
the original is kept. Providers restate history, and a paper record that
silently absorbs restatements is a record of decisions nobody actually made.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fullbacktester.data.schema import BAR_COLUMNS
from fullbacktester.execution.orders import Fill, Order, OrderStatus, OrderType, TimeInForce
from fullbacktester.markets import Frequency

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategies (
    name          TEXT PRIMARY KEY,
    code_hash     TEXT NOT NULL,
    warmup        INTEGER NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bars (
    timestamp   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);
CREATE TABLE IF NOT EXISTS bar_revisions (
    timestamp  TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    field      TEXT NOT NULL,
    first_seen REAL NOT NULL,
    later_seen REAL NOT NULL,
    noticed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    strategy        TEXT NOT NULL,
    order_id        INTEGER NOT NULL,
    created_at      TEXT,
    created_bar     INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    quantity        REAL,
    target_weight   REAL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    time_in_force   TEXT NOT NULL,
    max_bars        INTEGER,
    status          TEXT NOT NULL,
    filled_quantity REAL NOT NULL,
    reason          TEXT,
    PRIMARY KEY (strategy, order_id)
);
CREATE TABLE IF NOT EXISTS fills (
    strategy        TEXT NOT NULL,
    order_id        INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    bar             INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    quantity        REAL NOT NULL,
    price           REAL NOT NULL,
    reference_price REAL NOT NULL,
    commission      REAL NOT NULL,
    slippage_cost   REAL NOT NULL,
    participation   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
    strategy  TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cash      REAL NOT NULL,
    equity    REAL NOT NULL,
    PRIMARY KEY (strategy, timestamp)
);
CREATE TABLE IF NOT EXISTS positions (
    strategy TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    quantity REAL NOT NULL,
    PRIMARY KEY (strategy, symbol)
);
CREATE TABLE IF NOT EXISTS steps (
    ran_at            TEXT NOT NULL,
    bars_added        INTEGER NOT NULL,
    bars_revised      INTEGER NOT NULL,
    bars_processed    INTEGER NOT NULL,
    processed_through TEXT,
    note              TEXT
);
"""

_BAR_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SessionSpec:
    """Immutable identity of a paper-trading session.

    ``config_hash`` and each strategy's code hash are recorded at creation and
    checked on every reopen. Neither the config nor the strategies themselves
    are stored: they are supplied in code each run, and a mismatch stops the
    session rather than quietly continuing a different experiment.
    """

    name: str
    market: str
    frequency: Frequency
    symbols: tuple[str, ...]
    source: str | None
    config_hash: str
    created_at: pd.Timestamp

    def to_rows(self) -> dict[str, str]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "market": self.market,
            "frequency": self.frequency.value,
            "symbols": json.dumps(list(self.symbols)),
            "source": "" if self.source is None else self.source,
            "config_hash": self.config_hash,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_rows(cls, rows: dict[str, str]) -> SessionSpec:
        return cls(
            name=rows["name"],
            market=rows["market"],
            frequency=Frequency(rows["frequency"]),
            symbols=tuple(json.loads(rows["symbols"])),
            source=rows["source"] or None,
            config_hash=rows["config_hash"],
            created_at=pd.Timestamp(rows["created_at"]),
        )


@dataclass(frozen=True)
class StrategyRecord:
    name: str
    code_hash: str
    warmup: int
    registered_at: pd.Timestamp


class LiveStore:
    """SQLite persistence for one session. Every mutation runs in a transaction."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Leave isolation_level at its default so sqlite3 opens a transaction on
        # the first write and commit()/rollback() stay valid even when there is
        # nothing open. Driving BEGIN by hand breaks on executescript(), which
        # commits any pending transaction before it runs.
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # ------------------------------------------------------------- lifecycle

    def initialise(self, spec: SessionSpec, strategies: Sequence[StrategyRecord]) -> None:
        with self._connect() as db:
            db.executescript(_SCHEMA)
            db.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                list(spec.to_rows().items()),
            )
            db.executemany(
                "INSERT OR REPLACE INTO strategies (name, code_hash, warmup, registered_at)"
                " VALUES (?, ?, ?, ?)",
                [(s.name, s.code_hash, s.warmup, s.registered_at.isoformat()) for s in strategies],
            )

    def load_spec(self) -> SessionSpec:
        with self._connect() as db:
            rows = {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM meta")}
        if not rows:
            raise ValueError(f"{self.path} holds no session")
        stored = rows.get("schema_version")
        if stored != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} was written by schema version {stored!r}; this build expects "
                f"{SCHEMA_VERSION!r}. Start a new session rather than mixing formats."
            )
        return SessionSpec.from_rows(rows)

    def load_strategies(self) -> dict[str, StrategyRecord]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT name, code_hash, warmup, registered_at FROM strategies"
            ).fetchall()
        return {
            row["name"]: StrategyRecord(
                name=row["name"],
                code_hash=row["code_hash"],
                warmup=int(row["warmup"]),
                registered_at=pd.Timestamp(row["registered_at"]),
            )
            for row in rows
        }

    # ------------------------------------------------------------------ bars

    def record_bars(self, bars: pd.DataFrame, observed_at: pd.Timestamp) -> tuple[int, int]:
        """Insert bars not seen before. Returns ``(inserted, revisions_noted)``.

        A bar already present is never modified. If any field differs from what
        was first stored, the difference is appended to ``bar_revisions``.
        """
        if bars.empty:
            return 0, 0
        stamp = observed_at.isoformat()
        stamps = pd.DatetimeIndex(bars["timestamp"])
        symbols = [str(s) for s in bars["symbol"]]
        columns = {name: bars[name].to_numpy(dtype=np.float64) for name in _BAR_FIELDS}
        inserted = 0
        revisions = 0
        with self._connect() as db:
            for i, (timestamp, symbol) in enumerate(zip(stamps, symbols, strict=True)):
                key = (timestamp.isoformat(), symbol)
                values = {name: float(columns[name][i]) for name in _BAR_FIELDS}
                existing = db.execute(
                    "SELECT open, high, low, close, volume FROM bars"
                    " WHERE timestamp = ? AND symbol = ?",
                    key,
                ).fetchone()
                if existing is None:
                    db.execute(
                        "INSERT INTO bars (timestamp, symbol, open, high, low, close, volume,"
                        " observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            key[0],
                            key[1],
                            values["open"],
                            values["high"],
                            values["low"],
                            values["close"],
                            values["volume"],
                            stamp,
                        ),
                    )
                    inserted += 1
                    continue
                for name in _BAR_FIELDS:
                    was = float(existing[name])
                    now = values[name]
                    if was != now:
                        db.execute(
                            "INSERT INTO bar_revisions (timestamp, symbol, field, first_seen,"
                            " later_seen, noticed_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (key[0], key[1], name, was, now, stamp),
                        )
                        revisions += 1
        return inserted, revisions

    def observed_bars(self) -> pd.DataFrame:
        """Every bar as first seen, in canonical long form."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT timestamp, symbol, open, high, low, close, volume FROM bars"
                " ORDER BY timestamp, symbol"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        frame = pd.DataFrame([dict(r) for r in rows])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.loc[:, list(BAR_COLUMNS)]

    def revisions(self) -> pd.DataFrame:
        columns = ["timestamp", "symbol", "field", "first_seen", "later_seen", "noticed_at"]
        with self._connect() as db:
            rows = db.execute(
                "SELECT timestamp, symbol, field, first_seen, later_seen, noticed_at"
                " FROM bar_revisions ORDER BY rowid"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=columns)

    # ------------------------------------------------------------ strategies

    def save_state(
        self,
        strategy: str,
        cash: float,
        positions: dict[str, float],
        pending: Iterable[Order],
        touched_orders: Iterable[Order],
        new_fills: Iterable[Fill],
        equity_points: Iterable[tuple[pd.Timestamp, float, float]],
    ) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM positions WHERE strategy = ?", (strategy,))
            db.executemany(
                "INSERT INTO positions (strategy, symbol, quantity) VALUES (?, ?, ?)",
                [(strategy, symbol, qty) for symbol, qty in positions.items() if qty != 0],
            )
            db.executemany(
                "INSERT OR REPLACE INTO orders (strategy, order_id, created_at, created_bar,"
                " symbol, quantity, target_weight, order_type, limit_price, stop_price,"
                " time_in_force, max_bars, status, filled_quantity, reason)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [_order_row(strategy, order) for order in touched_orders],
            )
            db.executemany(
                "INSERT INTO fills (strategy, order_id, timestamp, bar, symbol, quantity, price,"
                " reference_price, commission, slippage_cost, participation)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        strategy,
                        fill.order_id,
                        fill.timestamp.isoformat(),
                        fill.bar,
                        fill.symbol,
                        fill.quantity,
                        fill.price,
                        fill.reference_price,
                        fill.commission,
                        fill.slippage_cost,
                        fill.participation,
                    )
                    for fill in new_fills
                ],
            )
            db.executemany(
                "INSERT OR REPLACE INTO equity (strategy, timestamp, cash, equity)"
                " VALUES (?, ?, ?, ?)",
                [(strategy, ts.isoformat(), c, e) for ts, c, e in equity_points],
            )
            db.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [
                    (f"cash:{strategy}", repr(float(cash))),
                    (f"pending:{strategy}", json.dumps([order.id for order in pending])),
                ],
            )

    def load_state(self, strategy: str) -> tuple[float | None, dict[str, float], list[int]]:
        with self._connect() as db:
            cash_row = db.execute(
                "SELECT value FROM meta WHERE key = ?", (f"cash:{strategy}",)
            ).fetchone()
            pending_row = db.execute(
                "SELECT value FROM meta WHERE key = ?", (f"pending:{strategy}",)
            ).fetchone()
            positions = {
                row["symbol"]: float(row["quantity"])
                for row in db.execute(
                    "SELECT symbol, quantity FROM positions WHERE strategy = ?", (strategy,)
                )
            }
        cash = float(cash_row["value"]) if cash_row is not None else None
        pending: list[int] = json.loads(pending_row["value"]) if pending_row is not None else []
        return cash, positions, pending

    def load_orders(self, strategy: str, ids: Sequence[int] | None = None) -> list[Order]:
        query = (
            "SELECT order_id, created_at, created_bar, symbol, quantity, target_weight,"
            " order_type, limit_price, stop_price, time_in_force, max_bars, status,"
            " filled_quantity, reason FROM orders WHERE strategy = ?"
        )
        params: list[Any] = [strategy]
        if ids is not None:
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            query += f" AND order_id IN ({placeholders})"
            params.extend(ids)
        with self._connect() as db:
            rows = db.execute(query + " ORDER BY order_id", params).fetchall()
        return [_row_to_order(row) for row in rows]

    def equity_frame(self, strategy: str) -> pd.DataFrame:
        with self._connect() as db:
            rows = db.execute(
                "SELECT timestamp, cash, equity FROM equity WHERE strategy = ? ORDER BY timestamp",
                (strategy,),
            ).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows], columns=["timestamp", "cash", "equity"])
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def fills_frame(self, strategy: str) -> pd.DataFrame:
        columns = [
            "timestamp",
            "bar",
            "order_id",
            "symbol",
            "quantity",
            "price",
            "reference_price",
            "commission",
            "slippage_cost",
            "participation",
        ]
        with self._connect() as db:
            rows = db.execute(
                "SELECT timestamp, bar, order_id, symbol, quantity, price, reference_price,"
                " commission, slippage_cost, participation FROM fills WHERE strategy = ?"
                " ORDER BY bar, order_id",
                (strategy,),
            ).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows], columns=columns)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def orders_frame(self, strategy: str) -> pd.DataFrame:
        from fullbacktester.execution.orders import orders_to_frame

        return orders_to_frame(self.load_orders(strategy))

    # ----------------------------------------------------------------- steps

    def record_step(
        self,
        ran_at: pd.Timestamp,
        bars_added: int,
        bars_revised: int,
        bars_processed: int,
        processed_through: pd.Timestamp | None,
        note: str = "",
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO steps (ran_at, bars_added, bars_revised, bars_processed,"
                " processed_through, note) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ran_at.isoformat(),
                    bars_added,
                    bars_revised,
                    bars_processed,
                    None if processed_through is None else processed_through.isoformat(),
                    note,
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (
                    "processed_through",
                    "" if processed_through is None else processed_through.isoformat(),
                ),
            )

    def processed_through(self) -> pd.Timestamp | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'processed_through'").fetchone()
        if row is None or not row["value"]:
            return None
        return pd.Timestamp(row["value"])

    def steps_frame(self) -> pd.DataFrame:
        columns = [
            "ran_at",
            "bars_added",
            "bars_revised",
            "bars_processed",
            "processed_through",
            "note",
        ]
        with self._connect() as db:
            rows = db.execute(
                "SELECT ran_at, bars_added, bars_revised, bars_processed, processed_through, note"
                " FROM steps ORDER BY rowid"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=columns)


def _order_row(strategy: str, order: Order) -> tuple[Any, ...]:
    quantity = None if pd.isna(order.quantity) else float(order.quantity)
    return (
        strategy,
        order.id,
        None if order.created_at is None else order.created_at.isoformat(),
        order.created_bar,
        order.symbol,
        quantity,
        order.target_weight,
        order.order_type.value,
        order.limit_price,
        order.stop_price,
        order.time_in_force.value,
        order.max_bars,
        order.status.value,
        order.filled_quantity,
        order.reason,
    )


def _row_to_order(row: sqlite3.Row) -> Order:
    order = Order(
        symbol=row["symbol"],
        quantity=float("nan") if row["quantity"] is None else float(row["quantity"]),
        order_type=OrderType(row["order_type"]),
        limit_price=row["limit_price"],
        stop_price=row["stop_price"],
        time_in_force=TimeInForce(row["time_in_force"]),
        target_weight=row["target_weight"],
        created_at=None if row["created_at"] is None else pd.Timestamp(row["created_at"]),
        created_bar=int(row["created_bar"]),
        max_bars=row["max_bars"],
        id=int(row["order_id"]),
    )
    order.status = OrderStatus(row["status"])
    order.filled_quantity = float(row["filled_quantity"])
    order.reason = row["reason"]
    return order
