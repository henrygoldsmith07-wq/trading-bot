"""The observed-membership snapshot log is evidence. It must not be corruptible.

The snapshot log is the only genuinely point-in-time membership evidence this
system has. If a day's universe can be silently replaced by a later append, or a
conflicting row can win on file order, then the point-in-time claim rests on
whatever the last writer happened to produce.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bot.universe_pit import (
    UniverseEvidenceError,
    coverage_report,
    eligibility_from_snapshots,
    load_snapshots,
    record_snapshot,
)

REPO = Path(__file__).resolve().parents[1]


def _line(date: str, symbols: list[str], **extra) -> str:
    entry = {"date": date, "generated_at": f"{date}T00:00:00+00:00", "source": "test",
             "universe": [{"symbol": s, "quote_volume_usd": 1.0} for s in symbols]}
    entry.update(extra)
    return json.dumps(entry)


# ---------------------------------------------------------------------------
# conflicts are refused, not last-write-wins
# ---------------------------------------------------------------------------


def test_same_date_with_a_different_universe_is_a_conflict(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC", "AAA"]) + chr(10) + _line("2026-08-23", ["BTC"]) + chr(10),
                 encoding="utf-8")
    with pytest.raises(UniverseEvidenceError) as exc:
        load_snapshots(p)
    assert "2026-08-23" in str(exc.value)


def test_identical_duplicate_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC", "AAA"]) + chr(10) + _line("2026-08-23", ["AAA", "BTC"]) + chr(10),
                 encoding="utf-8")
    assert load_snapshots(p) == {"2026-08-23": ["AAA", "BTC"]}


def test_duplicate_symbols_within_a_day_are_collapsed(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC", "AAA", "BTC"]) + chr(10), encoding="utf-8")
    assert load_snapshots(p) == {"2026-08-23": ["BTC", "AAA"]}


def test_a_conflicting_snapshot_is_refused_even_when_a_broken_line_follows(tmp_path: Path) -> None:
    """A corrupt tail must not let an earlier conflict pass unnoticed."""
    p = tmp_path / "u.jsonl"
    p.write_text(
        _line("2026-08-23", ["BTC"]) + chr(10) + _line("2026-08-23", ["ETH"]) + chr(10) + "{broken",
        encoding="utf-8",
    )
    with pytest.raises(UniverseEvidenceError):
        load_snapshots(p)


# ---------------------------------------------------------------------------
# strict vs lenient malformed-line handling
# ---------------------------------------------------------------------------


def test_torn_trailing_line_is_skipped_in_lenient_mode(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC"]) + chr(10) + "{torn", encoding="utf-8")
    assert load_snapshots(p) == {"2026-08-23": ["BTC"]}


def test_strict_mode_raises_on_a_malformed_line(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC"]) + chr(10) + "{torn", encoding="utf-8")
    with pytest.raises(UniverseEvidenceError):
        load_snapshots(p, strict=True)


def test_row_missing_a_date_is_malformed(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(json.dumps({"universe": []}) + chr(10), encoding="utf-8")
    with pytest.raises(UniverseEvidenceError):
        load_snapshots(p, strict=True)


def test_recording_refuses_to_append_onto_a_conflicting_log(tmp_path: Path) -> None:
    """Never add a good row to a log that is already internally inconsistent."""
    p = tmp_path / "u.jsonl"
    p.write_text(_line("2026-08-23", ["BTC"]) + chr(10) + _line("2026-08-23", ["ETH"]) + chr(10),
                 encoding="utf-8")
    with pytest.raises(UniverseEvidenceError):
        record_snapshot([("SOL", 1e9)], log_path=p, now=datetime(2026, 8, 25, tzinfo=UTC))


# ---------------------------------------------------------------------------
# coverage reporting
# ---------------------------------------------------------------------------


def test_coverage_reports_the_missing_fraction(tmp_path: Path) -> None:
    log = tmp_path / "u.jsonl"
    record_snapshot([("BTC", 1e9)], log_path=log, now=datetime(2026, 8, 23, tzinfo=UTC))
    snaps = load_snapshots(log)
    timeline = [
        int(datetime(2026, 8, d, tzinfo=UTC).timestamp() * 1000) for d in (23, 24, 25, 26)
    ]
    rep = coverage_report(snaps, timeline)
    assert rep["days_requested"] == 4
    assert rep["days_covered"] == 1
    assert rep["days_missing"] == 3
    assert rep["fraction_covered"] == 0.25
    assert "2026-08-24" in rep["missing_dates"]


def test_coverage_of_nothing_is_zero_not_a_division_error() -> None:
    assert coverage_report({}, [])["fraction_covered"] == 0.0


# ---------------------------------------------------------------------------
# eligibility from snapshots still omits, never invents
# ---------------------------------------------------------------------------


def test_missing_snapshot_days_stay_missing(tmp_path: Path) -> None:
    log = tmp_path / "u.jsonl"
    record_snapshot([("BTC", 1e9)], log_path=log, now=datetime(2026, 8, 23, tzinfo=UTC))
    elig = eligibility_from_snapshots(log, timeline=[int(datetime(2026, 8, 24, tzinfo=UTC).timestamp() * 1000)])
    assert elig == {}, "a gap day must be absent from the mask, not invented"


def test_committed_snapshot_log_is_internally_consistent() -> None:
    """The shipped universe evidence must load strictly with no conflict."""
    snaps = load_snapshots(REPO / "universe_log.jsonl", strict=True)
    assert snaps, "the repository ships observed-membership snapshots"
    assert all(v for v in snaps.values()), "no day may have an empty observed universe"
