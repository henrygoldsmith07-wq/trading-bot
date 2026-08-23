"""Research ledger: append-only experiment history + honest trial counts."""
import json

import pytest

from bot.research_ledger import (
    append_entry,
    deflated_sharpe_against_ledger,
    ledger_fingerprint,
    load_entries,
    recommended_trial_count,
    summarize,
    validate_entry,
    verify_chain,
)


@pytest.fixture()
def ledger(tmp_path):
    return tmp_path / "research_ledger.jsonl"


def _add(path, n=3, category="portfolio", metric="OOS Sharpe", accepted=True, result=None):
    out = []
    for i in range(n):
        out.append(append_entry(
            path,
            category=category,
            hypothesis=f"idea {len(load_entries(path))}",
            configuration={"k": i},
            primaryMetric=metric,
            result=result if result is not None else 0.5 + i * 0.1,
            accepted=accepted,
        ))
    return out


class TestAppendLoad:
    def test_ids_increment_and_entries_roundtrip(self, ledger):
        added = _add(ledger, n=3)
        assert [e["id"] for e in added] == [1, 2, 3]
        loaded = load_entries(ledger)
        assert len(loaded) == 3
        assert loaded[1]["hypothesis"] == "idea 1"

    def test_chain_links_and_verifies(self, ledger):
        _add(ledger, n=4)
        entries = load_entries(ledger)
        verify_chain(entries)  # no raise
        assert entries[2]["prev_hash"] == entries[1]["hash"]

    def test_edited_history_breaks_chain(self, ledger):
        _add(ledger, n=3)
        raw = [json.loads(line) for line in ledger.read_text().splitlines()]
        raw[1]["result"] = 9.99  # rewrite a failed experiment's result
        ledger.write_text("".join(json.dumps(e) + "\n" for e in raw))
        with pytest.raises(ValueError, match="content hash mismatch"):
            verify_chain(load_entries(ledger))

    def test_gap_detection(self, ledger):
        _add(ledger, n=3)
        raw = [json.loads(line) for line in ledger.read_text().splitlines()]
        del raw[1]  # delete a failed experiment — forbidden
        ledger.write_text("".join(json.dumps(e) + "\n" for e in raw))
        with pytest.raises(ValueError, match="gap"):
            verify_chain(load_entries(ledger))

    def test_torn_final_line_tolerated(self, ledger):
        _add(ledger, n=2)
        with open(ledger, "a", encoding="utf-8") as f:
            f.write('{"id": 3, "categ')
        assert len(load_entries(ledger)) == 2


class TestValidation:
    def test_unknown_category_rejected(self, ledger):
        with pytest.raises(ValueError, match="unknown category"):
            append_entry(ledger, category="vibes", hypothesis="h", configuration={},
                         primaryMetric="OOS Sharpe", result=1.0, accepted=True)

    def test_missing_field_rejected(self):
        with pytest.raises(ValueError, match="missing field"):
            validate_entry({"id": 1, "timestamp": "t", "category": "strategy"})

    def test_non_boolean_accepted_rejected(self, ledger):
        with pytest.raises(ValueError, match="boolean"):
            append_entry(ledger, category="strategy", hypothesis="h", configuration={},
                         primaryMetric="x", result=1.0, accepted="yes")


class TestSummaryAndCounts:
    def test_counts_by_category_and_search_total(self, ledger):
        _add(ledger, n=2, category="strategy")
        _add(ledger, n=3, category="portfolio")
        _add(ledger, n=1, category="execution")
        _add(ledger, n=2, category="methodology")
        s = summarize(load_entries(ledger))
        assert s["by_category"]["strategy"] == 2
        assert s["recommended_trial_count"] == 6  # methodology excluded from search
        assert s["total_entries"] == 8

    def test_sharpe_valued_trials_filtered(self, ledger):
        _add(ledger, n=2, metric="OOS Sharpe")
        _add(ledger, n=1, metric="max drawdown")
        s = summarize(load_entries(ledger))
        assert s["sharpe_valued_trials"] == 2
        assert all(isinstance(x, float) for x in s["trial_sharpes"])

    def test_recommended_count_none_without_file(self, tmp_path):
        assert recommended_trial_count(tmp_path / "missing.jsonl") is None


class TestDsrBridge:
    def test_deflated_against_ledger(self, ledger, tmp_path):
        import random

        _add(ledger, n=5, category="portfolio", metric="OOS Sharpe",
             result=[1.0, 0.8, 1.2, 0.9, 1.1][0])
        rng = random.Random(5)
        rets = [rng.gauss(0.0006, 0.01) for _ in range(300)]
        res = deflated_sharpe_against_ledger(rets, path=ledger)
        assert res["available"]
        assert res["n_trials"] == 5
        assert 0.0 <= res["dsr"] <= 1.0
        assert "5 ledger experiments" in res["note"]

    def test_absent_ledger_reports_unavailable(self, tmp_path):
        res = deflated_sharpe_against_ledger([0.001] * 50, path=tmp_path / "none.jsonl")
        assert not res["available"]

    def test_fingerprint_counts_and_hashes(self, ledger):
        _add(ledger, n=3)
        n, sha = ledger_fingerprint(ledger)
        assert n == 3 and len(sha) == 64
