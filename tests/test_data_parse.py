"""Offline tests for data parsing/cleaning (no network)."""
import json
import urllib.error

import pytest

from bot import data
from bot.data import DAY_MS, NON_RETRYABLE_HTTP, _parse_klines, clean_candles


def test_parse_klines_maps_binance_columns():
    raw = json.dumps(
        [
            [1722470400000, "100.5", "110.0", "95.0", "108.0", "42.5"],
            [1722556800000, "108.0", "112.0", "99.0", "111.0", "17.25"],
        ]
    )
    candles = _parse_klines(raw)
    assert candles[0] == {
        "open_time": 1722470400000,
        "open": 100.5,
        "high": 110.0,
        "low": 95.0,
        "close": 108.0,
        "volume": 42.5,
    }
    assert len(candles) == 2


def test_clean_candles_drops_bad_and_dedupes_keeps_higher_volume():
    candles = [
        {"open_time": 0, "close": 10.0, "volume": 1.0},
        {"open_time": DAY_MS, "close": float("nan"), "volume": 1.0},  # non-finite
        {"open_time": 2 * DAY_MS, "close": -3.0, "volume": 1.0},  # non-positive
        {"open_time": 3 * DAY_MS, "close": 12.0, "volume": 5.0},
        {"open_time": 3 * DAY_MS, "close": 11.0, "volume": 9.0},  # dup, higher volume wins
        {"open_time": DAY_MS, "close": 4.0, "volume": 1.0},  # out of order
    ]
    cleaned = clean_candles(candles)
    # non-finite and non-positive closes are dropped entirely
    assert [c["open_time"] for c in cleaned] == [0, DAY_MS, 3 * DAY_MS]
    assert cleaned[-1]["close"] == 11.0  # the volume-9 print beat the volume-5 print


class TestGetRetryPolicy:
    """A 451 from a datacentre range is deterministic: retrying it three times
    just burns the scheduled run's timeout budget before failing anyway."""

    @pytest.fixture(autouse=True)
    def no_sleeping(self, monkeypatch):
        monkeypatch.setattr(data.time, "sleep", lambda *_: None)

    @staticmethod
    def _raise_http(code):
        def inner(url, timeout=None):
            raise urllib.error.HTTPError("https://example.invalid", code, "nope", {}, None)

        return inner

    @pytest.mark.parametrize("code", sorted(NON_RETRYABLE_HTTP))
    def test_client_errors_are_not_retried(self, monkeypatch, code):
        calls = []

        def boom(url, timeout=None):
            calls.append(url)
            raise urllib.error.HTTPError("https://example.invalid", code, "nope", {}, None)

        monkeypatch.setattr(data.urllib.request, "urlopen", boom)
        with pytest.raises(urllib.error.HTTPError):
            data._get("https://example.invalid", attempts=3)
        assert len(calls) == 1, f"HTTP {code} must not be retried"

    def test_451_is_in_the_non_retryable_set(self):
        """The exact code Binance returns to GitHub Actions runner IPs, which
        is what broke the scheduled paper run. Pinned so a future edit to the
        set can't quietly start retrying it again."""
        assert 451 in NON_RETRYABLE_HTTP

    def test_rate_limits_are_still_retried(self, monkeypatch):
        calls = []

        def boom(url, timeout=None):
            calls.append(url)
            raise urllib.error.HTTPError("https://example.invalid", 429, "slow down", {}, None)

        monkeypatch.setattr(data.urllib.request, "urlopen", boom)
        with pytest.raises(urllib.error.HTTPError):
            data._get("https://example.invalid", attempts=3)
        assert len(calls) == 3, "429 clears, so backing off is worth it"

    def test_server_errors_are_still_retried(self, monkeypatch):
        calls = []

        def boom(url, timeout=None):
            calls.append(url)
            raise urllib.error.HTTPError("https://example.invalid", 503, "unavailable", {}, None)

        monkeypatch.setattr(data.urllib.request, "urlopen", boom)
        with pytest.raises(urllib.error.HTTPError):
            data._get("https://example.invalid", attempts=3)
        assert len(calls) == 3

    def test_transport_errors_are_still_retried(self, monkeypatch):
        calls = []

        def boom(url, timeout=None):
            calls.append(url)
            raise OSError("connection reset")

        monkeypatch.setattr(data.urllib.request, "urlopen", boom)
        with pytest.raises(OSError):
            data._get("https://example.invalid", attempts=3)
        assert len(calls) == 3


_ONE_CANDLE = json.dumps([[1722470400000, "100.5", "110.0", "95.0", "108.0", "42.5"]])


class TestKlinesMirrorFallback:
    """The forward EVIDENCE path must survive one host refusing.

    api.binance.com answers 451 to GitHub Actions runner IPs. With a single
    hard-coded host, ten of the thirteen frozen sleeves silently record an
    outage every night — and the day is still counted as a forward trading
    day, so a broken feed manufactures the evidence it never produced.
    """

    @staticmethod
    def _stub(monkeypatch, behaviour):
        """Replace _get so each mirror's fate is controlled; record every call."""
        calls: list[dict] = []

        def fake_get(url, timeout=15, attempts=3):
            calls.append({"url": url, "attempts": attempts})
            return behaviour(url)

        monkeypatch.setattr(data, "_get", fake_get)
        return calls

    @staticmethod
    def _refuse_first(n, code=451):
        """Refuse the first `n` mirrors, serve the rest.

        Matched by origin, not equality: real calls carry a query string.
        """
        refused = [u.split("/api/v3")[0] for u in list(data.BINANCE_KLINES_URLS)[:n]]

        def behaviour(url):
            if any(url.startswith(prefix) for prefix in refused):
                raise urllib.error.HTTPError(url, code, "unavailable for legal reasons", {}, None)
            return _ONE_CANDLE

        return behaviour

    def test_stops_at_the_first_host_that_answers(self, monkeypatch):
        calls = self._stub(monkeypatch, lambda url: _ONE_CANDLE)
        assert data._get_any(data.BINANCE_KLINES_URLS) == _ONE_CANDLE
        assert len(calls) == 1
        assert calls[0]["attempts"] == 1, "one try per host: ask elsewhere, don't wait"

    def test_vision_mirror_is_tried_first(self):
        """Measured on a runner: every *.binance.com host answers 451 there.

        Only data-api.binance.vision returned 200, so it must lead. Putting it
        last made every fetch pay four dead requests before the one that works.
        """
        assert data.BINANCE_KLINES_URLS[0].startswith("https://data-api.binance.vision/")

    def test_falls_back_past_refusing_hosts(self, monkeypatch):
        calls = self._stub(monkeypatch, self._refuse_first(3))
        assert data._get_any(data.BINANCE_KLINES_URLS) == _ONE_CANDLE
        assert len(calls) == 4
        # each refused host is tried exactly once before moving on
        assert [c["attempts"] for c in calls[:3]] == [1, 1, 1]

    def test_raises_the_last_error_not_a_stale_first_one(self, monkeypatch):
        first = data.BINANCE_KLINES_URLS[0].split("/api/v3")[0]

        def behaviour(url):
            code = 451 if url.startswith(first) else 503
            raise urllib.error.HTTPError(url, code, "nope", {}, None)

        self._stub(monkeypatch, behaviour)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            data._get_any(data.BINANCE_KLINES_URLS)
        assert excinfo.value.code == 503, "the log must show why the FINAL attempt failed"

    def test_exhausting_every_mirror_still_raises(self, monkeypatch):
        def behaviour(url):
            raise OSError("all mirrors down")

        calls = self._stub(monkeypatch, behaviour)
        with pytest.raises(OSError):
            data._get_any(data.BINANCE_KLINES_URLS)
        assert len(calls) == len(data.BINANCE_KLINES_URLS)

    def test_fetch_candles_inherits_the_fallback(self, monkeypatch):
        calls = self._stub(monkeypatch, self._refuse_first(1))
        out = data.fetch_candles("BTCUSDT", interval="1d", limit=1)
        assert len(out) == 1
        assert out[0]["close"] == 108.0
        assert len(calls) == 2, "one refusal, then the next mirror"

    def test_daily_history_retries_mirrors_on_every_page(self, monkeypatch):
        """A host can start refusing mid-history, so each page re-tries."""
        calls = self._stub(monkeypatch, self._refuse_first(1))
        out = data.fetch_daily_history("BTCUSDT", max_candles=1000)
        assert out and out[0]["close"] == 108.0
        assert len(calls) == 2

    def test_every_mirror_is_a_unique_https_endpoint(self):
        urls = data.BINANCE_KLINES_URLS
        assert len(urls) == len(set(urls))
        assert all(u.startswith("https://") for u in urls)
        assert all(u.endswith("/api/v3/klines") for u in urls)

    def test_klines_urls_appends_the_query_to_every_mirror(self):
        urls = data.klines_urls("?symbol=BTCUSDT&interval=1d&limit=1000")
        assert len(urls) == len(data.BINANCE_KLINES_URLS)
        assert all(u.endswith("?symbol=BTCUSDT&interval=1d&limit=1000") for u in urls)
        assert urls[0].startswith("https://data-api.binance.vision/api/v3/klines?")
