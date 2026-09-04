import json
import urllib.error

import pytest

from bot import universe
from bot.universe import TICKER_URLS, fetch_ticker_json, parse_symbols


def _ticker(symbols_vols):
    return json.dumps([{"symbol": s, "quoteVolume": v} for s, v in symbols_vols])


def _http_error(code):
    return urllib.error.HTTPError("https://example.invalid", code, "blocked", {}, None)


def test_orders_by_volume_and_filters_stables():
    raw = _ticker(
        [
            ("BTCUSDT", "5e9"),
            ("ETHUSDT", "3e9"),
            ("USDCUSDT", "9e9"),  # stablecoin pair — excluded
            ("BTCUPUSDT", "8e9"),  # leveraged token — excluded
            ("BNBUSDT", "1e9"),
            ("SOLUSDT", "2e9"),
            ("FOOBAR", "7e9"),  # wrong quote — excluded
        ]
    )
    assert parse_symbols(raw, n=5) == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def test_limits_to_n():
    raw = _ticker([(f"A{i}USDT", str(10 - i)) for i in range(8)])
    assert len(parse_symbols(raw, n=3)) == 3


def test_bad_volume_rows_skipped():
    raw = json.dumps(
        [
            {"symbol": "BTCUSDT", "quoteVolume": "1e9"},
            {"symbol": "XRPUSDT", "quoteVolume": "oops"},
            {"symbol": "NOVOL"},
        ]
    )
    assert parse_symbols(raw) == ["BTCUSDT"]


def test_dedupes_when_n_exceeds_rows():
    raw = _ticker([("BTCUSDT", "1"), ("ETHUSDT", "2")])
    assert parse_symbols(raw, n=10) == ["ETHUSDT", "BTCUSDT"]


class TestTickerMirrorFallback:
    """Binance answers HTTP 451 to some datacentre ranges — GitHub Actions
    runners included — so a hard-coded single host makes the scheduled paper
    run fail for reasons unrelated to the strategy."""

    def test_stops_at_the_first_host_that_answers(self, monkeypatch):
        calls = []

        def fake_get(url, timeout=15, attempts=3):
            calls.append((url, attempts))
            return _ticker([("BTCUSDT", "1e9")])

        monkeypatch.setattr(universe, "_get", fake_get)
        fetch_ticker_json(urls=("primary", "mirror"))
        # attempts=1 per host: the point of the loop is a different host, not waiting
        assert calls == [("primary", 1)]

    def test_falls_back_past_a_blocked_primary(self, monkeypatch):
        def fake_get(url, timeout=15, attempts=3):
            if url == "primary":
                raise _http_error(451)
            return _ticker([("BTCUSDT", "1e9")])

        monkeypatch.setattr(universe, "_get", fake_get)
        assert fetch_ticker_json(urls=("primary", "mirror")) == _ticker([("BTCUSDT", "1e9")])

    def test_raises_the_last_error_not_a_stale_first_one(self, monkeypatch):
        """The operator needs to know why the FINAL attempt failed."""

        def fake_get(url, timeout=15, attempts=3):
            raise _http_error(451 if url == "primary" else 503)

        monkeypatch.setattr(universe, "_get", fake_get)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            fetch_ticker_json(urls=("primary", "mirror"))
        assert excinfo.value.code == 503

    def test_top_symbols_inherits_the_fallback(self, monkeypatch):
        seen = []

        def fake_get(url, timeout=15, attempts=3):
            seen.append(url)
            if len(seen) == 1:
                raise _http_error(451)
            return _ticker([("BTCUSDT", "1e9"), ("ETHUSDT", "2e9")])

        monkeypatch.setattr(universe, "_get", fake_get)
        assert universe.top_symbols(2) == ["ETHUSDT", "BTCUSDT"]
        assert len(seen) == 2

    def test_every_configured_mirror_is_a_real_endpoint(self):
        assert len(TICKER_URLS) >= 2
        assert all(u.startswith("https://") for u in TICKER_URLS)
        assert len(set(TICKER_URLS)) == len(TICKER_URLS), "duplicate mirror hosts"

    def test_vision_mirror_is_tried_first(self):
        """Measured on a real runner, not assumed.

        Every *.binance.com host answered 451 there; only
        data-api.binance.vision answered 200. It must therefore lead, or the
        scheduled run pays four dead requests before the one that succeeds.
        """
        assert TICKER_URLS[0].startswith("https://data-api.binance.vision/")
