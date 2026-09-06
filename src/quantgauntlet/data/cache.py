"""Parquet cache: canonical local storage for every source.

Layout: ``<root>/<source>/<market>/<frequency>/<symbol>.parquet`` plus a
JSON sidecar recording the date range that has been fetched, so a request
inside that range is served without touching the network even when the
symbol simply had no bars on some days.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from platformdirs import user_cache_dir

from quantgauntlet.data.schema import BAR_COLUMNS, validate_bars
from quantgauntlet.markets import Frequency

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_cache_root() -> Path:
    return Path(user_cache_dir("quantgauntlet"))


@dataclass(frozen=True)
class CacheKey:
    source: str
    market: str
    frequency: Frequency
    symbol: str

    def relative_path(self) -> Path:
        safe_symbol = _SAFE.sub("_", self.symbol)
        return Path(self.source) / self.market / self.frequency.value / f"{safe_symbol}.parquet"


class ParquetCache:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_root()

    def path(self, key: CacheKey) -> Path:
        return self.root / key.relative_path()

    def _meta_path(self, key: CacheKey) -> Path:
        return self.path(key).with_suffix(".json")

    def fetched_range(self, key: CacheKey) -> tuple[date, date] | None:
        meta = self._meta_path(key)
        if not meta.exists():
            return None
        payload = json.loads(meta.read_text(encoding="utf-8"))
        return date.fromisoformat(payload["start"]), date.fromisoformat(payload["end"])

    def covers(self, key: CacheKey, start: date, end: date) -> bool:
        fetched = self.fetched_range(key)
        return fetched is not None and fetched[0] <= start and fetched[1] >= end

    def load(self, key: CacheKey) -> pd.DataFrame | None:
        path = self.path(key)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if frame.empty:
            return frame.reindex(columns=list(BAR_COLUMNS))
        return validate_bars(frame)

    def store(self, key: CacheKey, bars: pd.DataFrame, start: date, end: date) -> Path:
        """Merge ``bars`` into the cache and extend the fetched range to ``[start, end]``."""
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load(key)
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, bars], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        else:
            merged = bars
        merged = (
            validate_bars(merged) if not merged.empty else merged.reindex(columns=list(BAR_COLUMNS))
        )
        merged.to_parquet(path, index=False)
        fetched = self.fetched_range(key)
        if fetched is not None:
            start, end = min(start, fetched[0]), max(end, fetched[1])
        self._meta_path(key).write_text(
            json.dumps({"start": start.isoformat(), "end": end.isoformat()}), encoding="utf-8"
        )
        return path

    def clear(self, key: CacheKey) -> None:
        for path in (self.path(key), self._meta_path(key)):
            if path.exists():
                path.unlink()
