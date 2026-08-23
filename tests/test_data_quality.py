"""Tests for calendar alignment, gap handling, and delisting simulation."""
import pytest

from bot.data import (
    DAY_MS,
    extend_returns_to_timeline,
    fill_small_gaps,
    gap_report,
    simulate_delisting,
)


def _candles(closes):
    return [{"open_time": DAY_MS * i + 1000, "close": c} for i, c in enumerate(closes)]


class TestGapReport:
    def test_flags_multi_day_gaps(self):
        candles = [
            {"open_time": 0, "close": 1.0},
            {"open_time": DAY_MS, "close": 2.0},
            {"open_time": 4 * DAY_MS, "close": 3.0},  # 3-day gap
        ]
        gaps = gap_report(candles)
        assert len(gaps) == 1
        assert gaps[0]["start"] == DAY_MS
        assert gaps[0]["end"] == 4 * DAY_MS
        assert gaps[0]["days"] == pytest.approx(3.0)

    def test_contiguous_history_no_gaps(self):
        assert gap_report(_candles([1.0, 2.0, 3.0])) == []


class TestFillSmallGaps:
    def test_fills_small_gap_with_carried_close(self):
        candles = [
            {"open_time": 0, "close": 100.0},
            {"open_time": 3 * DAY_MS, "close": 110.0},  # 2 missing days
        ]
        filled = fill_small_gaps(candles, max_gap_days=3)
        assert len(filled) == 4
        assert [c["close"] for c in filled] == [100.0, 100.0, 100.0, 110.0]
        assert all(c.get("filled") for c in filled[1:3])
        assert filled[1]["volume"] == 0.0

    def test_large_gap_left_visible(self):
        candles = [
            {"open_time": 0, "close": 100.0},
            {"open_time": 10 * DAY_MS, "close": 110.0},
        ]
        filled = fill_small_gaps(candles, max_gap_days=3)
        assert len(filled) == 2
        assert not any(c.get("filled") for c in filled)

    def test_exactly_max_gap_filled(self):
        candles = [
            {"open_time": 0, "close": 5.0},
            {"open_time": 4 * DAY_MS, "close": 6.0},  # 3 missing
        ]
        filled = fill_small_gaps(candles, max_gap_days=3)
        assert len(filled) == 5


class TestExtendReturnsToTimeline:
    def test_interior_gap_becomes_invested_flat_day(self):
        t0, t1, t2, t3 = 10, 20, 30, 40
        daily = {t0: 0.01, t1: -0.02, t3: 0.05}  # t2 missing (interior)
        timeline = [t0, t1, t2, t3]
        out = extend_returns_to_timeline(daily, timeline)
        assert out[t2] == 0.0
        # compounding preserved exactly
        eq_src = 1.0
        for v in daily.values():
            eq_src *= 1 + v
        eq_out = 1.0
        for t in timeline:
            eq_out *= 1 + out[t]
        assert eq_out == pytest.approx(eq_src)

    def test_leading_and_trailing_days_stay_absent(self):
        daily = {30: 0.01, 31: 0.02}
        timeline = [10, 20, 30, 31, 32, 33]
        out = extend_returns_to_timeline(daily, timeline)
        assert set(out) == {30, 31}  # not yet listed / delisted -> cash sleeve

    def test_empty_daily_gives_empty(self):
        assert extend_returns_to_timeline({}, [1, 2, 3]) == {}


class TestSimulateDelisting:
    def _history(self, n=100):
        return [{"open_time": DAY_MS * i, "open": 100.0 + i, "close": 101.0 + i} for i in range(n)]

    def test_truncates_at_delist_date_and_marks_down(self):
        hist = self._history()
        kept = simulate_delisting(hist, delist_at_ms=DAY_MS * 49, terminal_cost_bps=200.0)
        assert len(kept) == 50
        assert kept[-1]["close"] == pytest.approx((101.0 + 49) * 0.98)
        assert kept[-1]["note"] == "simulated delisting"
        assert kept[-2]["close"] == pytest.approx(101.0 + 48)

    def test_delist_after_end_is_noop(self):
        hist = self._history()
        kept = simulate_delisting(hist, delist_at_ms=DAY_MS * 999)
        assert len(kept) == len(hist)

    def test_delist_before_start_empties(self):
        kept = simulate_delisting(self._history(), delist_at_ms=-DAY_MS)
        assert kept == []

    def test_downstream_stale_detection_fires(self):
        from bot.data import is_stale

        hist = self._history()
        cutoff = DAY_MS * 49
        kept = simulate_delisting(hist, delist_at_ms=cutoff)
        assert is_stale(kept, now_ms=cutoff + 46 * DAY_MS)
        assert not is_stale(kept, now_ms=cutoff + 10 * DAY_MS)
