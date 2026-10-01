"""A new experiment must grade only the forward days it actually produced.

The tape is append-only and spans experiments. A v2 freeze created today must
not be able to claim the v1 days: those were produced by earlier code under
an earlier accounting model, and reusing them would be backfilling
prospective evidence from history.
"""
from __future__ import annotations

import json
from pathlib import Path

from bot.prospective import (
    experiment_stamp,
    forward_performance,
    select_prospective_rows,
)

ROOT = Path(__file__).resolve().parents[1]

V2_MANIFEST = {
    "experiment_version": "v2",
    "accounting_model": "exact-wealth-v2",
    "git_commit_at_freeze": "a" * 40,
    "code_sha256": "b" * 64,
    "config_sha256": "c" * 64,
    "frozen_at_date": "2026-10-01",
}

V1_STAMP = {
    "experiment_version": "v1",
    "accounting_model": "shares-equal-weight-v1",
    "git_commit": "d" * 40,
    "code_sha256": "e" * 64,
    "config_sha256": "f" * 64,
    "frozen_at_date": "2026-09-06",
}


def _row(day: str, ret: float, stamp: dict | None) -> dict:
    row = {"date": day, "port_ret": ret, "rule_ret": ret, "assets": {"X": {}},
           "dayStatus": "traded", "orders": []}
    if stamp is not None:
        row["experiment"] = stamp
    return row


def _v2_stamp() -> dict:
    return experiment_stamp(V2_MANIFEST)


# ---------------------------------------------------------------------------
# row selection
# ---------------------------------------------------------------------------


def test_no_expectation_grades_every_row() -> None:
    rows = [_row("2026-09-07", 0.01, None), _row("2026-09-08", 0.02, None)]
    kept, scope = select_prospective_rows(rows, None)
    assert len(kept) == 2
    assert scope["excluded"] == 0


def test_rows_on_or_before_the_freeze_are_excluded() -> None:
    # The freeze day itself and any earlier day are not prospective: a row must
    # be recorded strictly after the freeze date to count.
    rows = [_row("2026-09-30", 0.01, _v2_stamp()), _row("2026-10-01", 0.02, _v2_stamp()),
            _row("2026-10-02", 0.03, _v2_stamp())]
    kept, scope = select_prospective_rows(rows, _v2_stamp())
    assert [r["date"] for r in kept] == ["2026-10-02"]
    assert scope["excluded"] == 2
    assert "recorded on or before the freeze date" in " ".join(scope["reasons"])


def test_legacy_unstamped_rows_are_never_counted_as_v2_evidence() -> None:
    """The real v1 tape: no stamp, and older than the v2 freeze."""
    rows = [_row(f"2026-09-{d:02d}", 0.01, None) for d in range(7, 23)]
    kept, scope = select_prospective_rows(rows, _v2_stamp())
    assert kept == []
    assert scope["excluded"] == 16


def test_rows_from_another_experiment_are_excluded() -> None:
    rows = [_row("2026-10-05", 0.01, V1_STAMP), _row("2026-10-06", 0.02, _v2_stamp())]
    kept, scope = select_prospective_rows(rows, _v2_stamp())
    assert [r["date"] for r in kept] == ["2026-10-06"]
    assert scope["excluded"] == 1


def test_a_differently_stamped_code_commit_is_excluded() -> None:
    bad = dict(_v2_stamp())
    bad["code_sha256"] = "9" * 64
    rows = [_row("2026-10-05", 0.01, bad), _row("2026-10-06", 0.02, _v2_stamp())]
    kept, scope = select_prospective_rows(rows, _v2_stamp())
    assert len(kept) == 1
    assert "code_sha256" in " ".join(scope["reasons"])


def test_selection_never_mutates_or_truncates_the_tape() -> None:
    rows = [_row("2026-09-07", 0.01, None), _row("2026-10-06", 0.02, _v2_stamp())]
    before = json.dumps(rows, sort_keys=True)
    select_prospective_rows(rows, _v2_stamp())
    assert json.dumps(rows, sort_keys=True) == before
    assert len(rows) == 2


def test_unparseable_dates_are_excluded_not_guessed() -> None:
    rows = [{"date": "not-a-date", "port_ret": 0.01}, _row("2026-10-06", 0.02, _v2_stamp())]
    kept, scope = select_prospective_rows(rows, _v2_stamp())
    assert len(kept) == 1
    assert "unparseable date" in scope["reasons"]


# ---------------------------------------------------------------------------
# grading reflects the scope
# ---------------------------------------------------------------------------


def test_performance_ignores_prior_experiment_rows() -> None:
    rows = [_row("2026-09-07", 0.50, None), _row("2026-09-08", 0.50, None)]
    perf = forward_performance(rows, freeze_date="2026-10-01", experiment=_v2_stamp())
    assert perf["return"] is None, "a tape with no rows for this experiment has no return"
    assert perf["scope"]["excluded"] == 2


def test_performance_on_a_matching_tape_reports_no_exclusions() -> None:
    rows = [_row("2026-10-06", 0.01, _v2_stamp()), _row("2026-10-07", 0.02, _v2_stamp())]
    perf = forward_performance(rows, freeze_date="2026-10-01", experiment=_v2_stamp())
    assert perf["scope"]["excluded"] == 0
    assert perf["return"] is not None and perf["return"] > 0


def test_performance_without_an_experiment_grades_the_whole_tape() -> None:
    rows = [_row("2026-09-07", 0.10, None), _row("2026-09-08", 0.10, None)]
    perf = forward_performance(rows, freeze_date="2026-09-06")
    assert perf["scope"]["excluded"] == 0
    assert perf["return"] is not None


# ---------------------------------------------------------------------------
# the committed freeze and tape
# ---------------------------------------------------------------------------


def test_committed_freeze_is_the_v1_additive_accounting_experiment() -> None:
    """The shipped freeze predates experiment stamping, so it is v1 by absence."""
    from bot.experiments import ACCOUNTING_MODEL_ADDITIVE, infer_accounting_model

    manifest = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
    # absence of the field IS the evidence it was sealed by v1 code
    assert "accounting_model" not in manifest
    assert "experiment_version" not in manifest
    assert infer_accounting_model(manifest) == ACCOUNTING_MODEL_ADDITIVE


def test_committed_v1_tape_carries_no_experiment_stamp() -> None:
    """v1 evidence stays exactly as recorded. It is history, not v2 evidence."""
    stamped = 0
    rows = 0
    for line in (ROOT / "forward_log.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        if "experiment" in json.loads(line):
            stamped += 1
    assert rows > 0
    assert stamped == 0, "the committed v1 tape must not be rewritten to carry a v2 stamp"
