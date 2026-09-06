"""Canonical bar schema and validation.

Every data source produces a long-format ``DataFrame`` with exactly these
columns, and ``validate_bars`` is the only way bars enter a ``Panel``. Sources
never get to decide what "close enough" means.

Columns:
    timestamp: tz-aware UTC instant at which the bar *closed*.
    symbol:    instrument identifier as a string.
    open, high, low, close: float64 prices.
    volume:    float64 traded quantity in the bar (0.0 when unknown).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_COLUMNS: tuple[str, ...] = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class SchemaError(ValueError):
    """Raised when a bar frame violates the canonical schema. Lists every violation found."""


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a long-format bar frame.

    Returns a new frame sorted by (timestamp, symbol) with a fresh RangeIndex
    and canonical dtypes. Raises ``SchemaError`` listing all problems rather
    than the first one, so a bad source can be fixed in one pass.
    """
    problems: list[str] = []

    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}; required: {list(BAR_COLUMNS)}")

    out = bars.loc[:, list(BAR_COLUMNS)].copy()

    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if ts.isna().any():
        problems.append(f"{int(ts.isna().sum())} unparseable timestamps")
    if ts.dt.tz is None:
        problems.append(
            "timestamps are naive; sources must localize bar close times "
            "(use Market.session_close_utc for daily bars)"
        )
    else:
        ts = ts.dt.tz_convert("UTC")
    out["timestamp"] = ts

    out["symbol"] = out["symbol"].astype(str)

    for col in (*PRICE_COLUMNS, "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if out["close"].isna().any():
        problems.append(f"{int(out['close'].isna().sum())} rows with NaN close")
    prices = out[list(PRICE_COLUMNS)]
    if (prices <= 0).any().any():
        problems.append(f"{int((prices <= 0).any(axis=1).sum())} rows with non-positive prices")
    if out["volume"].isna().any():
        problems.append(f"{int(out['volume'].isna().sum())} rows with NaN volume")
    if (out["volume"] < 0).any():
        problems.append(f"{int((out['volume'] < 0).sum())} rows with negative volume")

    hi_lo = out["high"] < out["low"]
    if hi_lo.any():
        problems.append(f"{int(hi_lo.sum())} rows with high < low")
    hi_oc = out["high"] < np.fmax(out["open"], out["close"])
    lo_oc = out["low"] > np.fmin(out["open"], out["close"])
    if hi_oc.any() or lo_oc.any():
        problems.append(
            f"{int((hi_oc | lo_oc).sum())} rows where open/close fall outside [low, high]"
        )

    dupes = out.duplicated(subset=["timestamp", "symbol"], keep=False)
    if dupes.any():
        problems.append(f"{int(dupes.sum())} rows share a (timestamp, symbol) key")

    if problems:
        raise SchemaError("; ".join(problems))

    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    return out
