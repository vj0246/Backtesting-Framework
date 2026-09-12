"""Broker feeds (Upstox, Alpaca) and the bar-timing rules they depend on.

Neither provider is called here; responses are built from the shapes their
documentation specifies. What these tests pin down is what a user would trip
over: symbols typed the way people type them, credentials that are missing or
expired, bars stamped when they close rather than when they start, rate limits,
pagination, the free-plan data delay, and bars that have not finished forming.
"""

import gzip
import json
from datetime import date, timedelta
from itertools import pairwise
from urllib.parse import unquote

import pandas as pd
import pytest

import fullbacktester as fbt
from fullbacktester.data import loader
from fullbacktester.data.cache import CacheKey
from fullbacktester.data.sources import _http
from fullbacktester.data.sources import alpaca as alpaca_module
from fullbacktester.data.sources.base import DataSource
from fullbacktester.markets import INDIA, US, Frequency

TOKEN = "secret-token-do-not-print"


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("body is not JSON")
        return self._payload


class FakeSession:
    def __init__(self, route):
        self.route = route
        self.calls = []

    def get(self, url, *, headers=None, params=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": dict(headers or {}), "params": dict(params or {})}
        )
        return self.route(url, dict(params or {}))


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    return slept


# ------------------------------------------------------------------- upstox

INSTRUMENTS = [
    {
        "segment": "NSE_FO",
        "instrument_type": "FUT",
        "trading_symbol": "RELIANCE",
        "instrument_key": "NSE_FO|35001",
    },
    {
        "segment": "NSE_EQ",
        "instrument_type": "EQ",
        "trading_symbol": "RELIANCE",
        "instrument_key": "NSE_EQ|INE002A01018",
    },
    {
        "segment": "NSE_EQ",
        "instrument_type": "EQ",
        "trading_symbol": "TCS",
        "instrument_key": "NSE_EQ|INE467B01029",
    },
]
MASTER_GZ = gzip.compress(json.dumps(INSTRUMENTS).encode())


def upstox_route(url, params):
    if url.endswith("NSE.json.gz"):
        return FakeResponse(content=MASTER_GZ)
    _key, unit, _interval, to, start = unquote(url.split("/historical-candle/")[1]).split("/")
    candles = []
    for day in pd.bdate_range(start, to):
        if unit == "days":
            candles.append([f"{day.date()}T00:00:00+05:30", 100, 101, 99, 100.5, 1_000, 0])
        else:
            candles.append([f"{day.date()}T09:15:00+05:30", 100, 101, 99, 100.5, 1_000, 0])
    return FakeResponse(payload={"status": "success", "data": {"candles": candles[::-1]}})


def make_upstox(tmp_path, route=upstox_route, token=TOKEN):
    session = FakeSession(route)
    return fbt.UpstoxSource(token=token, session=session, cache_dir=tmp_path), session


def candle_calls(session):
    return [c for c in session.calls if "historical-candle" in c["url"]]


def test_upstox_resolves_symbols_and_stamps_daily_bars_at_the_nse_close(tmp_path):
    source, session = make_upstox(tmp_path)
    bars = source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 10), Frequency.DAY_1, INDIA)

    assert list(bars["symbol"].unique()) == ["RELIANCE"]
    assert bars["timestamp"].is_monotonic_increasing  # Upstox returns newest first
    assert bars["timestamp"].iloc[0] == INDIA.session_close_utc(date(2024, 1, 1))
    call = candle_calls(session)[0]
    assert call["url"].endswith("NSE_EQ%7CINE002A01018/days/1/2024-01-10/2024-01-01")
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_upstox_splits_long_histories_into_spans_the_api_serves(tmp_path):
    source, session = make_upstox(tmp_path)
    bars = source.fetch(["TCS"], date(1995, 1, 1), date(2024, 12, 31), Frequency.DAY_1, INDIA)

    spans = []
    for call in candle_calls(session):
        to, start = unquote(call["url"]).split("/")[-2:]
        spans.append((date.fromisoformat(start), date.fromisoformat(to)))
    assert spans[0][0] == date(2000, 1, 1)  # daily history begins in 2000
    assert spans[-1][1] == date(2024, 12, 31)
    assert all((to - start).days < 3650 for start, to in spans)  # under a decade each
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in pairwise(spans))
    assert bars["timestamp"].is_unique


def test_upstox_intraday_bars_are_known_when_they_end(tmp_path):
    source, session = make_upstox(tmp_path)
    bars = source.fetch(["RELIANCE"], date(2021, 6, 1), date(2022, 1, 5), Frequency.MINUTE_5, INDIA)

    assert unquote(candle_calls(session)[0]["url"]).endswith("/2022-01-01")  # intraday from 2022
    first = bars["timestamp"].iloc[0].tz_convert("Asia/Kolkata")
    assert (first.hour, first.minute) == (9, 20)  # the 09:15 candle is final at 09:20


def test_upstox_suggests_the_symbol_you_probably_meant(tmp_path):
    source, _ = make_upstox(tmp_path)
    with pytest.raises(ValueError, match="did you mean 'RELIANCE'"):
        source.fetch(["RELIANC"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)


def test_upstox_ignores_derivatives_that_share_a_trading_symbol(tmp_path):
    source, _ = make_upstox(tmp_path)
    assert source.instrument_keys(["RELIANCE"]) == {"RELIANCE": "NSE_EQ|INE002A01018"}


def test_upstox_accepts_full_instrument_keys_without_the_master(tmp_path):
    source, session = make_upstox(tmp_path)
    bars = source.fetch(
        ["NSE_EQ|INE002A01018"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA
    )
    assert not bars.empty
    assert not any(c["url"].endswith("NSE.json.gz") for c in session.calls)


def test_upstox_downloads_the_instrument_master_once_a_day(tmp_path):
    first, first_session = make_upstox(tmp_path)
    first.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)
    second, second_session = make_upstox(tmp_path)
    second.fetch(["TCS"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)

    assert sum(c["url"].endswith("NSE.json.gz") for c in first_session.calls) == 1
    assert sum(c["url"].endswith("NSE.json.gz") for c in second_session.calls) == 0


def test_upstox_works_without_a_token_while_the_endpoint_is_public(tmp_path, monkeypatch):
    """Observed live in September 2026: historical candles answer without auth."""
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    source, session = make_upstox(tmp_path, token=None)
    bars = source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)
    assert not bars.empty
    assert "Authorization" not in candle_calls(session)[0]["headers"]


def test_if_upstox_starts_requiring_auth_the_error_says_what_to_set(tmp_path, monkeypatch):
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)

    def route(url, params):
        if url.endswith("NSE.json.gz"):
            return FakeResponse(content=MASTER_GZ)
        return FakeResponse(401, {"status": "error", "errors": [{"message": "Unauthorized"}]})

    source, _ = make_upstox(tmp_path, route=route, token=None)
    with pytest.raises(
        fbt.CredentialError, match=r"requires authentication.*UPSTOX_ANALYTICS_TOKEN"
    ):
        source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)


def test_upstox_reads_the_token_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "from-env")
    source, session = make_upstox(tmp_path, token=None)
    source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)
    assert candle_calls(session)[0]["headers"]["Authorization"] == "Bearer from-env"


def test_rejected_upstox_token_is_explained_and_never_echoed(tmp_path):
    def route(url, params):
        if url.endswith("NSE.json.gz"):
            return FakeResponse(content=MASTER_GZ)
        body = {
            "status": "error",
            "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token used to access API"}],
        }
        return FakeResponse(401, body)

    source, _ = make_upstox(tmp_path, route=route)
    with pytest.raises(fbt.CredentialError) as info:
        source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)
    message = str(info.value)
    assert "Invalid token" in message and "one year" in message
    assert TOKEN not in message
    assert TOKEN not in repr(source)


def test_rate_limits_are_retried_with_the_advised_backoff(tmp_path, no_sleeping):
    attempts = {"n": 0}

    def route(url, params):
        if url.endswith("NSE.json.gz"):
            return FakeResponse(content=MASTER_GZ)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(429, {"message": "Too many requests"}, headers={"Retry-After": "2"})
        return upstox_route(url, params)

    source, _ = make_upstox(tmp_path, route=route)
    bars = source.fetch(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 5), Frequency.DAY_1, INDIA)
    assert not bars.empty
    assert no_sleeping == [2.0]


def test_upstox_says_its_prices_exclude_dividends():
    assert any("dividends" in f.message for f in fbt.UpstoxSource().caveats())


# ------------------------------------------------------------------- alpaca


def alpaca_route(url, params):
    """Daily bars, stamped the way Alpaca stamps them: midnight New York."""
    end = pd.Timestamp(params["end"])
    bars = {}
    for ticker in params["symbols"].split(","):
        rows = []
        for day in pd.bdate_range(params["start"][:10], params["end"][:10]):
            label = pd.Timestamp(day.date()).tz_localize("America/New_York").tz_convert("UTC")
            if label <= end:
                rows.append(
                    {
                        "t": label.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1e6,
                        "n": 9,
                        "vw": 100.2,
                    }
                )
        bars[ticker] = rows
    return FakeResponse(payload={"bars": bars, "next_page_token": None, "currency": "USD"})


def make_alpaca(route=alpaca_route, **keys):
    session = FakeSession(route)
    keys = keys or {"key_id": "key-id", "secret_key": "secret-do-not-print"}
    return fbt.AlpacaSource(session=session, **keys), session


def test_alpaca_daily_bars_are_restamped_from_midnight_to_the_close():
    source, session = make_alpaca()
    bars = source.fetch(["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)

    new_york = bars["timestamp"].dt.tz_convert("America/New_York")
    assert (new_york.dt.hour == 16).all()
    assert new_york.iloc[0].date() == date(2024, 1, 2)
    params, headers = session.calls[0]["params"], session.calls[0]["headers"]
    assert (params["adjustment"], params["feed"], params["timeframe"]) == ("all", "sip", "1Day")
    assert headers["APCA-API-KEY-ID"] == "key-id"


def test_alpaca_follows_pagination_to_the_last_page():
    def route(url, params):
        rows = alpaca_route(url, {**params, "symbols": "AAPL"}).json()["bars"]["AAPL"]
        if "page_token" not in params:
            return FakeResponse(payload={"bars": {"AAPL": rows[:2]}, "next_page_token": "p2"})
        return FakeResponse(payload={"bars": {"AAPL": rows[2:]}, "next_page_token": None})

    source, session = make_alpaca(route)
    bars = source.fetch(["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)
    assert len(bars) == 4
    assert [c["params"].get("page_token") for c in session.calls] == [None, "p2"]


def test_alpaca_never_asks_for_the_last_fifteen_minutes_nor_keeps_a_forming_bar(monkeypatch):
    now = pd.Timestamp("2024-01-05 15:00", tz="America/New_York").tz_convert("UTC")
    monkeypatch.setattr(alpaca_module, "_utc_now", lambda: now)
    source, session = make_alpaca()
    bars = source.fetch(["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)

    assert pd.Timestamp(session.calls[0]["params"]["end"]) <= now - pd.Timedelta(minutes=15)
    # Friday is mid-session at 15:00; its bar must not come back as if it were final.
    assert bars["timestamp"].max() == US.session_close_utc(date(2024, 1, 4))


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("forbidden", fbt.CredentialError),
        ("subscription does not permit querying recent SIP data", RuntimeError),
    ],
)
def test_alpaca_errors_say_what_to_do_without_leaking_keys(message, error):
    source, _ = make_alpaca(lambda url, params: FakeResponse(403, {"message": message}))
    with pytest.raises(RuntimeError) as info:
        source.fetch(["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)
    assert type(info.value) is error
    assert "secret-do-not-print" not in str(info.value)
    assert "secret-do-not-print" not in repr(source)


def test_alpaca_missing_secret_names_the_official_variable(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key-id")
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    session = FakeSession(alpaca_route)
    with pytest.raises(fbt.CredentialError, match="APCA_API_SECRET_KEY"):
        fbt.AlpacaSource(session=session).fetch(
            ["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US
        )
    assert session.calls == []


def test_alpaca_accepts_yahoo_style_share_classes():
    source, session = make_alpaca()
    bars = source.fetch(["BRK-B"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)
    assert session.calls[0]["params"]["symbols"] == "BRK.B"
    assert set(bars["symbol"]) == {"BRK-B"}


# ------------------------------------------------------------ bar timing


def test_weekly_bars_are_known_at_the_last_session_of_their_week():
    assert US.bar_close_utc(pd.Timestamp("2024-01-08"), Frequency.WEEK_1) == US.session_close_utc(
        date(2024, 1, 12)
    )


def test_weekly_bar_over_a_holiday_is_never_stamped_before_its_last_trade():
    stamp = US.bar_close_utc(pd.Timestamp("2024-03-25"), Frequency.WEEK_1)  # Good Friday closed
    assert (
        US.session_close_utc(date(2024, 3, 28)) <= stamp <= US.session_close_utc(date(2024, 3, 29))
    )


def test_daily_labels_in_any_timezone_resolve_to_the_local_session_close():
    alpaca_label = pd.Timestamp("2024-01-02T05:00:00Z")  # midnight New York
    assert US.bar_close_utc(alpaca_label, Frequency.DAY_1) == US.session_close_utc(date(2024, 1, 2))
    upstox_label = pd.Timestamp("2024-01-02T00:00:00+05:30")
    assert INDIA.bar_close_utc(upstox_label, Frequency.DAY_1) == INDIA.session_close_utc(
        date(2024, 1, 2)
    )


def test_yahoo_weekly_bars_are_no_longer_stamped_on_their_monday():
    """Regression: a week of trades was treated as known on Monday afternoon."""
    from fullbacktester.data.sources.yfinance import _close_instants

    stamped = _close_instants(pd.DatetimeIndex(["2024-01-01", "2024-01-08"]), Frequency.WEEK_1, US)
    assert [s.tz_convert("America/New_York").day_name() for s in stamped] == ["Friday", "Friday"]


# ---------------------------------------------------------------- loader


class DailySource(DataSource):
    name = "daily"
    markets = frozenset({"US"})
    frequencies = frozenset({Frequency.DAY_1})
    adjusted = True
    survivorship_free = False

    def fetch(self, symbols, start, end, frequency, market):
        days = pd.bdate_range(start, end)
        return pd.DataFrame(
            {
                "timestamp": [market.session_close_utc(d) for d in days for _ in symbols],
                "symbol": [s for _ in days for s in symbols],
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        )


def test_bars_still_forming_are_neither_returned_nor_cached(tmp_path, monkeypatch):
    now = pd.Timestamp("2024-01-05 11:00", tz="America/New_York").tz_convert("UTC")
    monkeypatch.setattr(loader, "_utc_now", lambda: now)
    cache = fbt.ParquetCache(tmp_path)

    bars = loader.load_bars(
        ["AAPL"], "2024-01-02", "2024-01-05", market="US", source=DailySource(), cache=cache
    )

    assert bars["timestamp"].max() == US.session_close_utc(date(2024, 1, 4))
    key = CacheKey("daily", "US", Frequency.DAY_1, "AAPL")
    assert cache.covers(key, date(2024, 1, 2), date(2024, 1, 4))
    assert not cache.covers(key, date(2024, 1, 2), date(2024, 1, 5))  # refetched after the close


def test_load_panel_names_a_symbol_that_came_back_empty():
    class OnlyAAPL(DailySource):
        def fetch(self, symbols, start, end, frequency, market):
            return super().fetch([s for s in symbols if s == "AAPL"], start, end, frequency, market)

    with pytest.raises(ValueError, match="APPL"):
        fbt.load_panel(
            ["AAPL", "APPL"],
            "2024-01-02",
            "2024-01-05",
            market="US",
            source=OnlyAAPL(),
            cache=False,
        )


def test_panel_columns_follow_the_order_you_asked_for():
    panel = fbt.load_panel(
        ["MSFT", "AAPL"], "2024-01-02", "2024-01-05", market="US", source=DailySource(), cache=False
    )
    assert panel.symbols == ("MSFT", "AAPL")


# ----------------------------------------------------------- end to end


def test_upstox_feeds_a_panel_and_a_paper_session(tmp_path):
    source, _ = make_upstox(tmp_path / "upstox")
    panel = fbt.load_panel(
        ["RELIANCE", "TCS"], "2024-01-01", "2024-03-29", market="IN", source=source, cache=False
    )
    assert panel.symbols == ("RELIANCE", "TCS")
    assert panel.bars_per_year == 250
    assert any("dividends" in f.message for f in fbt.check_data_quality(panel, source=source))

    def half_each(view):
        return {s: 0.5 for s in view.symbols}

    config = fbt.EngineConfig(initial_cash=100_000, fractional_shares=True)
    session = fbt.PaperSession.create(
        tmp_path / "session.db",
        name="upstox",
        strategies={"hold": fbt.RuleBasedStrategy(half_each, warmup=1)},
        symbols=["RELIANCE", "TCS"],
        market="IN",
        config=config,
        source=source,
    )
    report = session.step(now=pd.Timestamp("2024-03-29 12:00", tz="UTC"))
    assert report.bars_added > 0 and report.bars_processed > 0
    assert session.spec.source == "upstox"  # the name is stored, never the token


def test_gateway_html_error_pages_are_reduced_to_their_title():
    """Seen live: Alpaca answers bad keys with an nginx HTML page, not JSON."""
    response = FakeResponse(401)
    response.text = (
        "<html>\n<head><title>401 Authorization Required</title></head>\n"
        "<body><center><h1>401 Authorization Required</h1></center></body></html>"
    )
    source, _ = make_alpaca(lambda url, params: response)
    with pytest.raises(fbt.CredentialError) as info:
        source.fetch(["AAPL"], date(2024, 1, 2), date(2024, 1, 5), Frequency.DAY_1, US)
    assert "401 Authorization Required" in str(info.value)
    assert "<" not in str(info.value)
