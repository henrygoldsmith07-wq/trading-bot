"""Offline tests for data parsing/cleaning (no network)."""
import json

from bot.data import DAY_MS, _parse_klines, clean_candles


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
