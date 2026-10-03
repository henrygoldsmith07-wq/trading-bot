"""Data quality: what is actually wrong with the tape, stated rather than repaired.

WHY THIS EXISTS
    Missing bars, stale feeds, duplicate timestamps and malformed records were
    once absorbed silently by whatever read them — each surface papering over
    whatever it found. A reader could not tell a clean evidence day from one
    stitched together by forgiveness. This module scores the tape as it IS,
    counting every problem instead of repairing it, so the dashboard can show
    reliability beside the return instead of underneath it.

    Nothing here mutates a record. A repair is only honest when it is
    deterministic and explicitly recorded elsewhere (the quarantine layer
    preserves every original byte); a scorecard's job is to describe.

The score is a deterministic penalty sum, documented below, not a probability
and not a grade: two runs over the same tape always produce the same number.

Paper trading only.
"""
from __future__ import annotations

from typing import Any

from .prospective import classify_forward_days

REQUIRED_ROW_FIELDS = ("date", "port_ret", "dayStatus")

# Penalty weights for the scorecard score. Defined once so the dashboard, the
# report and the tests all read the same numbers.
SCORE_START = 100.0
PENALTY_PARTIAL_DAY = 5.0
PENALTY_OUTAGE_DAY = 10.0
PENALTY_DUPLICATE_TIMESTAMP = 5.0
PENALTY_MALFORMED_BAR = 10.0
PENALTY_MISSING_SNAPSHOT = 5.0
PENALTY_REPRODUCIBILITY_FAIL = 10.0
PENALTY_NO_BENCHMARK = 10.0

VALID_REPRODUCIBILITY = ("pass", "fail", "unknown")


def scorecard(
    forward_entries: list[dict[str, Any]],
    *,
    universe_snapshot_dates: list[str] | None = None,
    expected_snapshot_dates: list[str] | None = None,
    benchmark_available: bool | None = None,
    reproducibility_status: str = "unknown",
    rejected_counts: dict[str, int] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Score the reliability of the evidence tape without touching it.

    `expected_snapshot_dates` is the universe-snapshot schedule the caller
    says SHOULD exist; gaps are counted, never invented. `rejected_counts`
    carries what the ingestion layer refused (future-dated rows, malformed
    rows) so refusals stay visible instead of vanishing as non-events.
    """
    problems: list[dict[str, str]] = []

    def problem(code: str, severity: str, detail: str) -> None:
        problems.append({"code": code, "severity": severity, "detail": detail})

    rejected = dict(rejected_counts or {})
    dates = [str(e.get("date") or "") for e in forward_entries]

    malformed = int(rejected.get("malformed", 0))
    for entry, d in zip(forward_entries, dates, strict=True):
        if not d or any(k not in entry for k in REQUIRED_ROW_FIELDS):
            malformed += 1
    if malformed:
        problem("malformed_rows", "error", f"{malformed} row(s) missing required fields or rejected as malformed")

    duplicates = len(dates) - len(set(dates))
    if duplicates:
        problem("duplicate_timestamps", "warning", f"{duplicates} duplicate session date(s) in the forward tape")

    future_dated = int(rejected.get("future_dated", 0))
    if as_of is not None:
        future_dated += sum(1 for d in dates if d and d > as_of)
    if future_dated:
        problem("future_dated_rows", "error", f"{future_dated} row(s) dated after {as_of or 'the reporting date'}")

    stale_bars = 0
    outage_days = 0
    missing_bars = 0
    for entry in forward_entries:
        outages = entry.get("outages") or []
        if outages:
            outage_days += 1
            missing_bars += len(outages)
        stale_bars += len(entry.get("stale_data") or [])

    day_counts = classify_forward_days(forward_entries) if forward_entries else {"full": 0, "partial": 0, "dark": 0, "closed": 0}

    observed = set(universe_snapshot_dates or [])
    expected = list(expected_snapshot_dates or [])
    missing_snapshots = sum(1 for d in expected if d not in observed)
    if missing_snapshots:
        problem(
            "missing_universe_snapshots",
            "warning",
            f"{missing_snapshots} expected universe snapshot date(s) were never observed; membership on those dates is unknown",
        )

    if benchmark_available is None:
        problem("benchmark_availability_unknown", "info", "benchmark availability was not established; treated as unknown, not as absent")
    elif benchmark_available is False:
        problem("benchmark_unavailable", "warning", "the benchmark series is unavailable for this window")

    status = reproducibility_status if reproducibility_status in VALID_REPRODUCIBILITY else "unknown"
    if status != reproducibility_status:
        problem("reproducibility_status_invalid", "warning",
                f"reproducibility_status {reproducibility_status!r} is not one of {VALID_REPRODUCIBILITY}; recorded as 'unknown'")
    elif status == "fail":
        problem("reproducibility_failed", "error", "the last reproduction attempt failed; numbers may not be reproducible")

    score = SCORE_START
    score -= PENALTY_PARTIAL_DAY * day_counts.get("partial", 0)
    score -= PENALTY_OUTAGE_DAY * outage_days
    score -= PENALTY_DUPLICATE_TIMESTAMP * duplicates
    score -= PENALTY_MALFORMED_BAR * malformed
    score -= PENALTY_MISSING_SNAPSHOT * missing_snapshots
    score -= PENALTY_REPRODUCIBILITY_FAIL if status == "fail" else 0.0
    score -= PENALTY_NO_BENCHMARK if benchmark_available is False else 0.0
    score = max(0.0, min(SCORE_START, score))

    return {
        "missing_bars": missing_bars,
        "stale_bars": stale_bars,
        "future_dated_rejected": future_dated,
        "duplicate_timestamps": duplicates,
        "malformed_bars": malformed,
        "outage_days": outage_days,
        "partial_outage_days": day_counts.get("partial", 0),
        "full_evidence_days": day_counts.get("full", 0),
        "missing_universe_snapshots": missing_snapshots,
        "benchmark_available": benchmark_available,
        "reproducibility_status": status,
        "problems": problems,
        "score": score,
    }
