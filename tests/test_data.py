"""Cache, local source, loader, quality checks, and source registry."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fullbacktester.data.cache import CacheKey, ParquetCache
from fullbacktester.data.loader import load_bars, load_panel
from fullbacktester.data.quality import check_data_quality
from fullbacktester.data.sources import LocalSource, available_sources, get_source
from fullbacktester.data.sources.base import DataSource
from fullbacktester.data.sources.nse import NSEBhavcopySource
from fullbacktester.data.sources.yfinance import YFinanceSource
from fullbacktester.flags import Severity
from fullbacktester.markets import INDIA, US, Frequency


def _write_csv(directory, symbol, dates, price=100.0):
    frame = pd.DataFrame(
        {
            "Date": [d.date().isoformat() for d in dates],
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Adj Close": price,
            "Volume": 1000,
        }
    )
    frame.to_csv(directory / f"{symbol}.csv", index=False)


def test_local_source_stamps_daily_bars_at_session_close(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=5)
    _write_csv(tmp_path, "AAPL", dates)
    bars = load_bars(
        ["AAPL"], "2024-01-02", "2024-01-08", market="US", source=LocalSource(tmp_path), cache=False
    )
    assert list(bars["symbol"].unique()) == ["AAPL"]
    assert bars["timestamp"].iloc[0] == US.session_close_utc(dates[0])
    assert len(bars) == 5


def test_loader_uses_cache_after_first_fetch(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=5)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(data_dir, "RELIANCE", dates)
    cache = ParquetCache(tmp_path / "cache")
    source = LocalSource(data_dir)
    first = load_panel(
        ["RELIANCE"], "2024-01-02", "2024-01-08", market="IN", source=source, cache=cache
    )
    assert len(first) == 5
    key = CacheKey("local", "IN", Frequency.DAY_1, "RELIANCE")
    assert cache.covers(key, date(2024, 1, 3), date(2024, 1, 5))
    (data_dir / "RELIANCE.csv").unlink()  # the source is gone; the cache must answer
    second = load_panel(
        ["RELIANCE"], "2024-01-03", "2024-01-05", market="IN", source=source, cache=cache
    )
    assert len(second) == 3
    assert second.timestamps[0] == INDIA.session_close_utc(pd.Timestamp("2024-01-03"))


def test_cache_merges_and_extends_range(tmp_path):
    cache = ParquetCache(tmp_path)
    key = CacheKey("local", "US", Frequency.DAY_1, "X")
    stamps = [US.session_close_utc(d) for d in pd.bdate_range("2024-01-02", periods=3)]

    def rows(ts, px):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": "X",
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 1.0,
            }
        )

    cache.store(key, rows(stamps[:2], 1.0), date(2024, 1, 2), date(2024, 1, 3))
    cache.store(key, rows(stamps[1:], 2.0), date(2024, 1, 3), date(2024, 1, 4))
    merged = cache.load(key)
    assert len(merged) == 3
    assert merged.loc[merged["timestamp"] == stamps[1], "close"].item() == 2.0  # newest wins
    assert cache.fetched_range(key) == (date(2024, 1, 2), date(2024, 1, 4))
    cache.clear(key)
    assert cache.load(key) is None


def test_missing_symbol_raises(tmp_path):
    with pytest.raises(ValueError, match="no bars"):
        load_bars(
            ["NOPE"],
            "2024-01-02",
            "2024-01-08",
            market="US",
            source=LocalSource(tmp_path),
            cache=False,
        )


def test_registry_and_defaults():
    assert {"local", "yfinance", "nse"} <= set(available_sources())
    assert isinstance(get_source("nse"), DataSource)
    with pytest.raises(KeyError):
        get_source("bloomberg")
    assert YFinanceSource.ticker_for("RELIANCE", INDIA) == "RELIANCE.NS"
    assert YFinanceSource.ticker_for("TCS.BO", INDIA) == "TCS.BO"
    assert YFinanceSource.ticker_for("AAPL", US) == "AAPL"
    assert not NSEBhavcopySource().supports(US, Frequency.DAY_1)
    assert NSEBhavcopySource().supports(INDIA, Frequency.DAY_1)


def test_source_caveats_become_flags():
    yf_flags = YFinanceSource().caveats()
    assert any("survivorship" in f.message for f in yf_flags)
    nse_flags = NSEBhavcopySource().caveats()
    assert any("unadjusted" in f.message and f.severity is Severity.WARN for f in nse_flags)
    assert not any("survivorship" in f.message for f in nse_flags)


def test_nse_session_frame_parses_udiff_and_legacy_layouts(tmp_path):
    source = NSEBhavcopySource(raw_dir=tmp_path)
    udiff = (
        "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
        "2025-01-02,2025-01-02,CM,NSE,STK,11394,INE358U01012,ZOTA,EQ,,,,,ZOTA HEALTH CARE LIMITED,799.95,814.25,796.30,802.45,804.00,797.10,,802.45,,,21689,17428538.20,1007,F1,1,,,,,\n"
        "2025-01-02,2025-01-02,CM,NSE,STK,458,IN0020170091,SGBNOV25,GB,,,,,2.50% GOLDBONDS2025SR-VII,7948.00,7948.00,7948.00,7948.00,7948.00,7950.00,,7948.00,,,10,79480.00,3,F1,1,,,,,\n"
    )
    (tmp_path / "2025-01-02.csv").write_text(udiff, encoding="utf-8")
    legacy = (
        "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
        "RELIANCE,EQ,2550.0,2580.5,2540.0,2571.2,2570.0,2548.0,4500000,11500000000,02-JAN-2023,150000,INE002A01018,\n"
        "SOMEBOND,GB,100,100,100,100,100,100,10,1000,02-JAN-2023,1,INXXXX,\n"
    )
    (tmp_path / "2023-01-02.csv").write_text(legacy, encoding="utf-8")
    (tmp_path / "2023-01-03.missing").touch()

    bars = source.fetch(
        ["ZOTA", "RELIANCE"], date(2025, 1, 2), date(2025, 1, 2), Frequency.DAY_1, INDIA
    )
    assert list(bars["symbol"]) == ["ZOTA"]
    assert bars["timestamp"].iloc[0] == INDIA.session_close_utc(date(2025, 1, 2))
    assert bars["volume"].iloc[0] == 21689
    assert source.list_symbols(date(2023, 1, 2)) == ["RELIANCE"]  # GB series filtered out
    assert source.list_symbols(date(2023, 1, 3)) == []  # cached non-session, no network


def test_quality_checks_flag_jumps_gaps_zero_volume_and_stale(panel_factory):
    clean = panel_factory(n_bars=40)
    assert check_data_quality(clean) == []

    bars = clean.to_bars()
    bars.loc[(bars["symbol"] == "S0") & (bars.index % 7 == 3), "volume"] = 0.0
    jump_ts = bars.loc[bars["symbol"] == "S1", "timestamp"].iloc[20]
    bars.loc[
        (bars["symbol"] == "S1") & (bars["timestamp"] == jump_ts), ["open", "high", "low", "close"]
    ] *= 2.0
    drop_ts = bars.loc[bars["symbol"] == "S2", "timestamp"].iloc[10]
    bars = bars[~((bars["symbol"] == "S2") & (bars["timestamp"] == drop_ts))]
    from fullbacktester.data.panel import Panel

    dirty = Panel.from_bars(bars, market=US)
    flags = check_data_quality(dirty)
    by_symbol = {(f.symbol, f.severity) for f in flags}
    assert ("S0", Severity.INFO) in by_symbol
    assert any(f.symbol == "S1" and "single-bar move" in f.message for f in flags)
    assert any(f.symbol == "S2" and "no bar" in f.message for f in flags)

    stale_close = pd.DataFrame(np.full((40, 1), 50.0), index=clean.timestamps, columns=["Z"])
    stale = Panel.from_wide(stale_close, market=US)
    assert any("stale" in f.message for f in check_data_quality(stale))


def test_unscheduled_sessions_do_not_mask_absent_ones(panel_factory):
    """Regression: counting sessions let extra bars hide real gaps.

    Found on a live 2018-2024 NSE pull. Six Diwali Muhurat sessions, which no
    standard calendar lists, cancelled out three genuinely absent sessions, so
    a panel with real holes reported clean.
    """
    from fullbacktester.data.panel import Panel

    clean = panel_factory(n_bars=40)
    bars = clean.to_bars()

    # Drop three real sessions for one symbol.
    dropped = sorted(bars.loc[bars["symbol"] == "S0", "timestamp"].unique())[10:13]
    bars = bars[~((bars["symbol"] == "S0") & (bars["timestamp"].isin(dropped)))]

    # Add three bars on days the calendar does not call sessions (a weekend).
    saturdays = [pd.Timestamp("2023-01-07"), pd.Timestamp("2023-01-14"), pd.Timestamp("2023-01-21")]
    extra = pd.DataFrame(
        {
            "timestamp": [US.session_close_utc(d) for d in saturdays],
            "symbol": "S0",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    panel = Panel.from_bars(pd.concat([bars, extra], ignore_index=True), market=US)

    flags = [f for f in check_data_quality(panel) if f.symbol == "S0"]
    absent = [f for f in flags if "no bar" in f.message]
    unscheduled = [f for f in flags if "does not list as sessions" in f.message]

    # Counts cancel out exactly: 3 dropped, 3 added. Only set comparison sees both.
    assert len(absent) == 1 and "3 session(s)" in absent[0].message
    assert absent[0].severity is Severity.WARN if US.holiday_aware else Severity.INFO
    assert len(unscheduled) == 1 and "3 bar(s)" in unscheduled[0].message
    assert str(dropped[0].date()) in absent[0].message


def test_clean_panel_reports_neither_absent_nor_unscheduled(panel_factory):
    assert check_data_quality(panel_factory(n_bars=40)) == []
