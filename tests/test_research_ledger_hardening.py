"""Fail-closed integrity tests for the research experiment ledger."""

import json

import pytest

from bot.research_ledger import (
    append_entry,
    load_entries,
    recommended_trial_count,
    verify_chain,
)


def _append(path, result=0.5):
    return append_entry(
        path,
        category="strategy",
        hypothesis="test hypothesis",
        configuration={"window": 20},
        primaryMetric="OOS Sharpe",
        result=result,
        accepted=False,
    )


def test_interior_json_corruption_is_not_silently_skipped(tmp_path):
    ledger = tmp_path / "research.jsonl"
    _append(ledger)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write('{"id":BROKEN}\n')
        f.write('{"id":3}\n')
    with pytest.raises(ValueError, match="corrupt JSON at line 2"):
        load_entries(ledger)


def test_nonfinite_experiment_result_is_rejected(tmp_path):
    ledger = tmp_path / "research.jsonl"
    with pytest.raises(ValueError, match="finite"):
        _append(ledger, result=float("nan"))
    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_append_refuses_to_extend_tampered_history(tmp_path):
    ledger = tmp_path / "research.jsonl"
    _append(ledger)
    _append(ledger)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[0]["result"] = 99.0
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        _append(ledger)


def test_trial_count_refuses_tampered_history(tmp_path):
    ledger = tmp_path / "research.jsonl"
    _append(ledger)
    _append(ledger)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[1]["accepted"] = True
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        recommended_trial_count(ledger)


def test_append_repairs_only_torn_final_crash_fragment(tmp_path):
    ledger = tmp_path / "research.jsonl"
    _append(ledger)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write('{"id": 2, "category": "strat')

    # The malformed unterminated tail is the one tolerated crash shape.
    assert len(load_entries(ledger)) == 1
    appended = _append(ledger)
    assert appended["id"] == 2

    entries = load_entries(ledger)
    assert [entry["id"] for entry in entries] == [1, 2]
    verify_chain(entries)
