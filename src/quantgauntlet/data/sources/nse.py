"""NSE India official end-of-day bhavcopy. Optional dependency: ``pip install quantgauntlet[nse]``.

One file per session from the exchange's public archive, in either the UDiFF
format (``BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip``, 2024 onward) or the
legacy format (``cmDDMONYYYYbhav.csv.zip``); both are tried per date.

Good for: a survivorship-free universe (every listed symbol, every day, from
the exchange itself), official volumes, and ``list_symbols`` for building
point-in-time universes.

Not good for: adjusted prices. Bhavcopy prices are raw; a 1:10 split shows up
as a -90% day. Use it for universe membership and cross-check volumes, and
take adjusted prices from a source that adjusts.

Raw daily files are cached under the cache root so a range is fetched once.
The exchange serves at most a few requests per second politely; a small delay
is inserted between downloads.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from quantgauntlet.data.cache import default_cache_root
from quantgauntlet.data.schema import BAR_COLUMNS
from quantgauntlet.data.sources.base import DataSource
from quantgauntlet.markets import INDIA, Frequency, Market

_UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)
_LEGACY_URL = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
_UDIFF_COLUMNS = {
    "TradDt": "date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "TtlTradgVol": "volume",
}
_LEGACY_COLUMNS = {
    "TIMESTAMP": "date",
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "TOTTRDQTY": "volume",
}


class NSEBhavcopySource(DataSource):
    name = "nse"
    markets = frozenset({"IN"})
    frequencies = frozenset({Frequency.DAY_1})
    adjusted = False
    survivorship_free = True

    def __init__(
        self,
        *,
        series: Sequence[str] = ("EQ", "BE"),
        raw_dir: Path | str | None = None,
        delay_seconds: float = 0.4,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.series = tuple(series)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else default_cache_root() / "nse_raw"
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries

    # ---------------------------------------------------------------- public

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        frequency: Frequency,
        market: Market,
    ) -> pd.DataFrame:
        if frequency is not Frequency.DAY_1:
            raise ValueError("NSE bhavcopy is daily only")
        wanted = set(symbols)
        frames = []
        for session in INDIA.expected_sessions(start, end):
            day = self._session_frame(session.date())
            if day is None:
                continue
            day = day[day["symbol"].isin(wanted)]
            if not day.empty:
                frames.append(day)
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        return pd.concat(frames, ignore_index=True).loc[:, list(BAR_COLUMNS)]

    def list_symbols(self, session: date) -> list[str]:
        """Every symbol in the chosen series on ``session`` (empty if not a trading day)."""
        frame = self._session_frame(session)
        return [] if frame is None else sorted(frame["symbol"].unique())

    # -------------------------------------------------------------- internals

    def _session_frame(self, session: date) -> pd.DataFrame | None:
        raw = self._raw_csv(session)
        if raw is None:
            return None
        table = pd.read_csv(io.StringIO(raw))
        columns = _UDIFF_COLUMNS if "TckrSymb" in table.columns else _LEGACY_COLUMNS
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"bhavcopy for {session} lacks columns {missing}")
        table = table.rename(columns=columns)[list(columns.values())]
        table["series"] = table["series"].astype(str).str.strip()
        table = table[table["series"].isin(self.series)]
        out = pd.DataFrame(
            {
                "timestamp": INDIA.session_close_utc(session),
                "symbol": table["symbol"].astype(str).str.strip(),
                "open": table["open"].astype(float),
                "high": table["high"].astype(float),
                "low": table["low"].astype(float),
                "close": table["close"].astype(float),
                "volume": table["volume"].astype(float),
            }
        )
        return out.reset_index(drop=True)

    def _raw_csv(self, session: date) -> str | None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.raw_dir / f"{session.isoformat()}.csv"
        missing_marker = self.raw_dir / f"{session.isoformat()}.missing"
        if csv_path.exists():
            return csv_path.read_text(encoding="utf-8")
        if missing_marker.exists():
            return None
        payload = self._download(session)
        if payload is None:
            missing_marker.touch()
            return None
        csv_path.write_text(payload, encoding="utf-8")
        return payload

    def _download(self, session: date) -> str | None:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is not installed; run: pip install 'quantgauntlet[nse]'"
            ) from exc

        urls = [
            _UDIFF_URL.format(ymd=session.strftime("%Y%m%d")),
            _LEGACY_URL.format(
                yyyy=session.strftime("%Y"),
                mon=session.strftime("%b").upper(),
                dd=session.strftime("%d"),
            ),
        ]
        with requests.Session() as http:
            http.headers.update(_HEADERS)
            for url in urls:
                for attempt in range(self.max_retries):
                    time.sleep(self.delay_seconds)
                    response = http.get(url, timeout=self.timeout)
                    if response.status_code == 404:
                        break
                    if response.ok:
                        return _unzip_single_csv(response.content)
                    if response.status_code in (403, 429) or response.status_code >= 500:
                        time.sleep(self.delay_seconds * (2**attempt) + 1.0)
                        continue
                    response.raise_for_status()
        return None


def _unzip_single_csv(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV inside bhavcopy zip, found {names}")
        return archive.read(names[0]).decode("utf-8")
