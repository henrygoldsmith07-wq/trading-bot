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
