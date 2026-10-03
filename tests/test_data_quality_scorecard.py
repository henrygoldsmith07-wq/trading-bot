"""Data-quality scorecard: every problem counted and visible, never repaired.

Invariants pinned here:
* missing fields fail closed (counted as malformed, never guessed);
* universe-snapshot gaps are counted but membership is never invented;
* the score formula is deterministic and clamped to [0, 100];
* an unknown benchmark is disclosed as unknown, not as absent.
"""
from __future__ import annotations

from bot.data_quality import (
    PENALTY_MALFORMED_BAR,
    PENALTY_OUTAGE_DAY,
    PENALTY_PARTIAL_DAY,
    SCORE_START,
    scorecard,
)


def _row(date: str, **extra):
    base = {"date": date, "port_ret": 0.001, "dayStatus": "traded",
            "assets": {"A": {"sleeve_ret": 0.001}, "B": {"sleeve_ret": 0.001}}}
    base.update(extra)
    return base


class TestCounts:
    def test_clean_tape_scores_full(self):
        out = scorecard([_row("2026-09-01"), _row("2026-09-02")],
                        reproducibility_status="pass", benchmark_available=True)
        assert out["score"] == SCORE_START
        assert out["full_evidence_days"] == 2
        assert out["problems"] == []

    def test_outage_days_counted(self):
        rows = [_row("2026-09-01", outages=[{"symbol": "A"}])]
        out = scorecard(rows, reproducibility_status="pass", benchmark_available=True)
        assert out["outage_days"] == 1
        assert out["missing_bars"] == 1
        assert out["score"] == SCORE_START - PENALTY_OUTAGE_DAY

    def test_duplicate_dates_counted(self):
        out = scorecard([_row("2026-09-01"), _row("2026-09-01")],
                        reproducibility_status="pass", benchmark_available=True)
        assert out["duplicate_timestamps"] == 1

    def test_malformed_rows_fail_closed(self):
        out = scorecard([{"date": "2026-09-01"}], reproducibility_status="pass", benchmark_available=True)
        assert out["malformed_bars"] == 1
        assert out["score"] == SCORE_START - PENALTY_MALFORMED_BAR

    def test_rejected_counts_are_carried_through(self):
        out = scorecard([], rejected_counts={"malformed": 3, "future_dated": 2},
                        reproducibility_status="pass", benchmark_available=True)
        assert out["malformed_bars"] == 3
        assert out["future_dated_rejected"] == 2

    def test_future_rows_after_as_of_counted(self):
        out = scorecard([_row("2026-09-10")], as_of="2026-09-05",
                        reproducibility_status="pass", benchmark_available=True)
        assert out["future_dated_rejected"] == 1

    def test_empty_tape_is_zero_not_a_crash(self):
        out = scorecard([], reproducibility_status="pass", benchmark_available=True)
        assert out["score"] == SCORE_START
        assert out["full_evidence_days"] == 0


class TestUniverseSnapshots:
    def test_gaps_counted_never_invented(self):
        out = scorecard([],
                        universe_snapshot_dates=["2026-09-01"],
                        expected_snapshot_dates=["2026-09-01", "2026-09-02", "2026-09-03"],
                        reproducibility_status="pass", benchmark_available=True)
        assert out["missing_universe_snapshots"] == 2
        assert any(p["code"] == "missing_universe_snapshots" for p in out["problems"])


class TestStatusValidation:
    def test_unknown_reproducibility_is_disclosed(self):
        out = scorecard([], reproducibility_status="sort-of")
        assert out["reproducibility_status"] == "unknown"
        assert any(p["code"] == "reproducibility_status_invalid" for p in out["problems"])

    def test_failed_reproducibility_costs_score(self):
        out = scorecard([], reproducibility_status="fail", benchmark_available=True)
        assert out["score"] < SCORE_START

    def test_unknown_benchmark_is_not_absent(self):
        out = scorecard([], reproducibility_status="pass", benchmark_available=None)
        assert out["benchmark_available"] is None
        assert any(p["code"] == "benchmark_availability_unknown" for p in out["problems"])
        assert out["score"] == SCORE_START


class TestScore:
    def test_partial_days_cost_less_than_outages(self):
        partial_rows = [_row("2026-09-01", assets={"A": {"sleeve_ret": 0.001}, "B": {"note": "session_pending", "sleeve_ret": 0.0}})]
        assert scorecard(partial_rows, reproducibility_status="pass", benchmark_available=True)["score"] == \
            SCORE_START - PENALTY_PARTIAL_DAY

    def test_score_clamps_at_zero(self):
        rows = [_row(f"2026-09-{i:02d}", outages=[{"symbol": "A"}]) for i in range(1, 20)]
        out = scorecard(rows, reproducibility_status="fail", benchmark_available=False)
        assert out["score"] == 0.0

    def test_score_never_exceeds_the_ceiling(self):
        assert scorecard([], reproducibility_status="pass", benchmark_available=True)["score"] <= SCORE_START

    def test_deterministic(self):
        rows = [_row("2026-09-01", outages=[{"symbol": "A"}]), _row("2026-09-02")]
        assert scorecard(rows) == scorecard(rows)
