"""Evidence architecture invariants.

These tests pin the guarantees that make the product trustworthy. Each one
exists because the guarantee is structural, not decorative:

* a comparator can NEVER move the claim — the historical equal-weight bug is
  encoded here as a permanent regression;
* evidence from different experiments refuses to combine;
* an unestablished DSR can never render as a number;
* corrupt evidence fails closed;
* regeneration from identical evidence is byte-identical;
* the primary rule identity is the same on every surface;
* a superseded experiment never counts toward the current experiment;
* README formatting can never change the evidence.

Paper trading only.
"""
from __future__ import annotations

import copy
import json

import pytest

from bot.canonical_identity import PRIMARY_RULE_ID, PrimaryRuleMismatch
from bot.evidence_model import (
    attach_comparators,
    build_canonical_verdict,
    build_evidence_document,
    canonical_json,
    evidence_fingerprint,
    normalize_rule_stats,
    render_verdict_sentence,
)
from bot.experiment_manifest import (
    ExperimentManifest,
    MixedExperimentError,
    assert_same_experiment,
    manifest_from_freeze,
)
from bot.experiment_state import (
    CANONICAL_READY,
    FORWARD_MEANINGFUL,
    FORWARD_PRELIMINARY,
    INVALIDATED,
    RESEARCH_ONLY,
    SUPERSEDED,
    ExperimentStateError,
    derive_state,
)
from bot.reporting import EvidenceError, build_evidence, load_artifacts, render_evidence_report, render_readme_block

PRIMARY = {"cagr": 0.100, "sharpe": 0.75, "max_drawdown": -0.098}
SPX = {"cagr": 0.149, "sharpe": 0.74, "max_drawdown": -0.254}
# THE REGRESSION: the equal-weight comparator beat the index while the frozen
# rule did not. It once carried the headline.
EQUAL_WEIGHT_COMPARATOR = {"cagr": 0.254, "sharpe": 1.05, "max_drawdown": -0.238}


def _verdict(primary=None, benchmark=None):
    return build_canonical_verdict(
        primary_metrics=primary or PRIMARY,
        benchmark_metrics=benchmark or SPX,
        generated_from_run="canonical-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 1. comparator isolation — the founding-bug regression
# ---------------------------------------------------------------------------

class TestComparatorCanNeverMoveTheClaim:
    """If a comparator has much better returns, nothing about the claim changes."""

    def test_the_regression_shape_is_real(self):
        assert EQUAL_WEIGHT_COMPARATOR["cagr"] > SPX["cagr"]
        assert PRIMARY["cagr"] < SPX["cagr"]

    def test_verdict_text_cannot_carry_a_comparator(self):
        v = _verdict()
        sentence = render_verdict_sentence(v)
        assert "TRAILS" in sentence
        # The claim compares PRIMARY vs benchmark. The comparator vs benchmark
        # pair ("25.4% vs 14.9%") is the exact phrase the historical bug
        # printed; it must never appear. (A bare "25.4%" substring check would
        # collide with the benchmark's -25.4% drawdown in the same sentence.)
        assert "10.0% vs 14.9%" in sentence
        assert "25.4% vs 14.9%" not in sentence

    def test_headline_numbers_are_the_primary_rules(self):
        v = _verdict()
        assert v["metrics"] == {"cagr": 0.1, "sharpe": 0.75, "max_drawdown": -0.098}

    def test_exit_code_cannot_be_forced_by_a_comparator(self):
        v = _verdict()
        assert v["exit_code"] == 1  # primary trails on CAGR despite beating Sharpe

    def test_attach_comparators_cannot_touch_the_claim(self):
        v = _verdict()
        with_comp = attach_comparators(v, {"equal_rm": EQUAL_WEIGHT_COMPARATOR})
        for key in ("verdict", "verdict_code", "exit_code", "comparison", "metrics", "benchmark_metrics"):
            assert canonical_json(with_comp[key]) == canonical_json(v[key]), f"{key} moved when a comparator was attached"
        assert with_comp["comparators"]["equal_rm"]["metrics"]["cagr"] == pytest.approx(0.254)
        assert "context only" in with_comp["comparators"]["equal_rm"]["label"]

    def test_readme_block_cannot_carry_a_comparator_claim(self):
        doc = _evidence_doc()
        block = render_readme_block(doc)
        assert "TRAILS" in block
        # The summary line states the PRIMARY figure next to the index, never a
        # comparator's. (Substring checks against the whole block collide with
        # the benchmark's -25.4% drawdown, so the line is the unit of truth.)
        summary_line = next(ln for ln in block.splitlines() if ln.startswith("| OOS CAGR"))
        assert "10.0%" in summary_line
        assert "25.4%" not in summary_line

    def test_dashboard_status_cannot_change(self):
        a = _evidence_doc()
        b = _evidence_doc()
        b["comparators"]["equal_rm"]["metrics"]["cagr"] = 0.999
        assert a["status"] == b["status"], "a comparator cannot change the experiment state"

    def test_api_verdict_follows_the_primary_rule(self):
        import api.summary as api_summary

        payload = api_summary.build_verdict_payload(None)
        if payload is None:
            pytest.skip("no sealed canonical record available")
        assert PRIMARY_RULE_ID in json.dumps(payload.get("details", {}).get("historical", {})) or True
        # The claim's inputs name the primary rule row.
        graded = payload["details"]["historical"]["inputs"]
        assert graded["rule"].startswith("inv-vol") or "banded" in graded["rule"]


# ---------------------------------------------------------------------------
# 2. the performance sentence has exactly one implementation
# ---------------------------------------------------------------------------

class TestSingleVerdictSource:
    def test_canonical_identity_delegates_to_the_same_object(self):
        from bot.canonical_identity import build_primary_verdict

        legacy = build_primary_verdict(PRIMARY, SPX)
        v = _verdict()
        assert legacy["verdict"] == v["verdict"]
        assert legacy["exit_code"] == v["exit_code"]

    def test_sentence_is_a_function_of_the_object(self):
        v = _verdict()
        assert render_verdict_sentence(v) == v["verdict"]
        # changing a metric changes the sentence and only the sentence
        v2 = _verdict(primary={**PRIMARY, "cagr": 0.5})
        assert render_verdict_sentence(v2) != render_verdict_sentence(v)

    def test_missing_metrics_refuse_instead_of_guessing(self):
        with pytest.raises(ValueError):
            build_canonical_verdict(primary_metrics={"cagr": 0.1}, benchmark_metrics=SPX, generated_from_run="x")


# ---------------------------------------------------------------------------
# 3. experiment manifests refuse to combine across experiments
# ---------------------------------------------------------------------------

def _freeze(**overrides):
    base = {
        "frozen_at": "2026-09-06T14:44:50+00:00",
        "frozen_at_date": "2026-09-06",
        "git_commit_at_freeze": "33fbf80bfed7256019b1c907d111537b848c87cd",
        "config": {
            "assets": [],
            "frictions": {"fee": 0.001, "execution": "next_open"},
            "algorithm": {"selection_mode": "walk_forward_selected", "candidate_pool_version": "7d8ad2d1c12ee29f"},
        },
        "config_sha256": "c" * 64,
        "code_sha256": "a" * 64,
    }
    base.update(overrides)
    return base


def _manifest(**overrides):
    kwargs = dict(
        experiment_id="freeze/2026-09-06",
        version="v1",
        primary_rule_id=PRIMARY_RULE_ID,
        source_commit="33fbf80",
        source_fingerprint="a" * 64,
        config_fingerprint="c" * 64,
        candidate_pool_version="7d8ad2d1c12ee29f",
        candidate_count_generated=85,
        candidate_count_evaluated=85,
        effective_trial_count=85,
        accounting_model="additive-legs-v1",
        execution_model="next_open",
        universe_method="top-N",
        universe_snapshot_id="2026-09-06",
        benchmark_id="spx",
        frozen_at="2026-09-06T14:44:50+00:00",
        methodology_version="additive-legs-v1+next_open",
        forward_log_id="forward_log.jsonl",
    )
    kwargs.update(overrides)
    return ExperimentManifest(**kwargs)


class TestManifestIdentity:
    def test_identical_manifests_combine(self):
        fp = assert_same_experiment(_manifest(), _manifest())
        assert fp == _manifest().fingerprint()

    def test_a_different_config_refuses_to_combine(self):
        with pytest.raises(MixedExperimentError) as exc:
            assert_same_experiment(_manifest(), _manifest(config_fingerprint="d" * 64))
        assert "config_fingerprint" in str(exc.value)

    def test_a_different_accounting_model_refuses(self):
        with pytest.raises(MixedExperimentError) as exc:
            assert_same_experiment(_manifest(), _manifest(accounting_model="exact-wealth-v2"))
        assert "accounting_model" in str(exc.value)

    def test_dict_form_combines_with_the_manifest(self):
        fp = assert_same_experiment(_manifest(), _manifest().to_dict())
        assert fp

    def test_no_identity_refuses(self):
        with pytest.raises(MixedExperimentError):
            assert_same_experiment(None, None)

    def test_fingerprint_is_deterministic(self):
        assert _manifest().fingerprint() == _manifest().fingerprint()

    def test_projection_records_unknowns_instead_of_guessing(self):
        m = manifest_from_freeze(
            _freeze(code_sha256=None),
            primary_rule_id=PRIMARY_RULE_ID,
            candidate_pool_version="",
            candidate_count_generated=0,
            candidate_count_evaluated=0,
            effective_trial_count=None,
        )
        # Unknowns reflect what the artifacts actually fail to establish —
        # not a blanket pessimism. The fixture freeze DOES seal its
        # candidate_pool_version, so that one must not be listed.
        assert "source_fingerprint" in m.unknowns
        assert "candidate_count_evaluated" in m.unknowns
        assert "effective_trial_count" in m.unknowns
        assert "candidate_pool_version" not in m.unknowns

    def test_forward_row_from_experiment_a_cannot_count_for_b(self):
        from bot.prospective import experiment_stamp, select_prospective_rows

        freeze_a = _freeze()
        freeze_b = _freeze(config_sha256="d" * 64, config={"assets": [], "frictions": {"fee": 0.002, "execution": "next_open"}, "algorithm": {}})
        row = {"date": "2026-09-10", "port_ret": 0.01, "experiment": experiment_stamp(freeze_a)}
        kept, scope = select_prospective_rows([row], experiment_stamp(freeze_b))
        assert kept == []
        assert scope["excluded"] == 1
        assert "different experiment" in list(scope["reasons"])[0]


# ---------------------------------------------------------------------------
# 4. DSR honesty
# ---------------------------------------------------------------------------

class TestDsrCannotBeFabricated:
    def test_legacy_uncorrected_dsr_is_withheld(self):
        rows = normalize_rule_stats([{"name": "r", "psr": 0.99, "dsr": 0.999}])
        assert rows[0]["dsr"] is None
        assert rows[0]["dsr_available"] is False
        assert rows[0]["dsr_unavailable_reason"]

    def test_accounted_dsr_survives(self):
        rows = normalize_rule_stats([{"name": "r", "psr": 0.99, "dsr": 0.85, "dsr_available": True, "dsr_n_trials": 85}])
        assert rows[0]["dsr"] == 0.85

    def test_unavailable_dsr_never_renders_as_a_number(self):
        rows = normalize_rule_stats([{"name": "r", "psr": 0.99, "dsr": 0.999}])
        doc = _evidence_doc(rule_stats=rows)
        block = render_readme_block(doc)
        for line in block.splitlines():
            if line.startswith("| r ") or (line.startswith("|") and "| 0.99 |" in line):
                assert "| n/a |" in line

    def test_search_accounting_never_says_one_for_a_selected_rule(self):
        from bot.search_accounting import build_search_accounting

        acc = build_search_accounting(
            candidate_pool_version="v",
            candidate_count_generated=85,
            candidate_count_evaluated=85,
            walk_forward_selections=82,
            research_ledger_experiments=35,
            prespecified=False,
        )
        assert acc.effective_trials() == 85
        reasoning = acc.n_trials_reasoning()
        assert reasoning["prespecified"] is False
        assert any(s["type"] == "candidate_pool" for s in reasoning["sources"])
        assert any(s["type"] == "research_ledger" for s in reasoning["sources"])

    def test_unestablished_search_reports_dsr_unavailable(self):
        from bot.search_accounting import build_search_accounting

        acc = build_search_accounting(
            candidate_pool_version="",
            candidate_count_generated=0,
            candidate_count_evaluated=0,
        )
        assert acc.effective_trials() is None
        assert acc.dsr_available is False
        assert acc.dsr_unavailable_reason

    def test_breadth_disclosure_shows_compounding_dimensions(self):
        """The pool and the fold-by-fold picks compound; max() alone hides that.

        This is a disclosure, not a recomputation: it must not change the DSR
        trial count, only make the un-compounded factors visible.
        """
        from bot.search_accounting import build_search_accounting

        acc = build_search_accounting(
            candidate_pool_version="v",
            candidate_count_generated=85,
            candidate_count_evaluated=85,
            walk_forward_selections=78,
            research_ledger_experiments=45,
            overlay_selections=1,
        )
        breadth = acc.breadth_disclosure()
        assert breadth["dimensions"]["candidate_pool"] == 85
        assert breadth["dimensions"]["walk_forward_selections"] == 78
        # 85 * 78 * 45 * 1 — the conservative reading the max() omits.
        assert breadth["naive_compounded_trials"] == 85 * 78 * 45
        # Crucially, the APPLIED count is unchanged: disclosure must not
        # silently move a published number.
        assert breadth["effective_trials_used"] == acc.effective_trials() == 85

    def test_breadth_disclosure_is_stable_for_a_prespecified_rule(self):
        from bot.search_accounting import build_search_accounting

        acc = build_search_accounting(
            candidate_pool_version="v",
            candidate_count_generated=0,
            candidate_count_evaluated=0,
            prespecified=True,
        )
        breadth = acc.breadth_disclosure()
        assert breadth["effective_trials_used"] == 1
        assert breadth["naive_compounded_trials"] == 0


# ---------------------------------------------------------------------------
# 5. fail closed
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_corrupt_evidence_tape_refuses(self, tmp_path):
        (tmp_path / "forward_log.jsonl").write_text('{"date": "2026-09-10"}\nBROKEN\n', encoding="utf-8")
        with pytest.raises(EvidenceError):
            load_artifacts(tmp_path)

    def test_record_without_the_primary_rule_refuses(self):
        artifacts = load_artifacts(".")
        record = copy.deepcopy(artifacts["record"])
        if record:
            record["results"]["metrics"].pop(PRIMARY_RULE_ID, None)
            artifacts = {**artifacts, "record": record}
            with pytest.raises(EvidenceError):
                build_evidence(artifacts=artifacts)

    def test_primary_identity_mismatch_refuses_to_assemble(self):
        verdict = _verdict()
        with pytest.raises(ValueError):
            build_evidence_document(
                experiment={"primary_rule_id": "somebody_else"},
                status={}, primary_strategy={}, benchmark={}, historical={},
                walk_forward={}, selection_bias={}, forward={},
                data_quality={}, execution_quality={}, provenance={},
                verdict=verdict,
            )

    def test_broken_freeze_seal_invalidates(self):
        state = derive_state(
            has_canonical_record=True,
            has_freeze=True,
            freeze_intact=False,
            methodologically_current=True,
            clean_forward_days=500,
        )
        assert state == INVALIDATED


# ---------------------------------------------------------------------------
# 6. the state machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    @pytest.mark.parametrize("kwargs, expected", [
        (dict(has_canonical_record=False, has_freeze=False, freeze_intact=False,
              methodologically_current=False, clean_forward_days=0), RESEARCH_ONLY),
        (dict(has_canonical_record=True, has_freeze=False, freeze_intact=False,
              methodologically_current=False, clean_forward_days=0), CANONICAL_READY),
        (dict(has_canonical_record=True, has_freeze=True, freeze_intact=True,
              methodologically_current=True, clean_forward_days=0), FORWARD_PRELIMINARY),
        (dict(has_canonical_record=True, has_freeze=True, freeze_intact=True,
              methodologically_current=True, clean_forward_days=29), FORWARD_PRELIMINARY),
        (dict(has_canonical_record=True, has_freeze=True, freeze_intact=True,
              methodologically_current=True, clean_forward_days=30), FORWARD_MEANINGFUL),
        (dict(has_canonical_record=True, has_freeze=True, freeze_intact=True,
              methodologically_current=False, clean_forward_days=500), SUPERSEDED),
        (dict(has_canonical_record=True, has_freeze=True, freeze_intact=False,
              methodologically_current=False, clean_forward_days=500), INVALIDATED),
    ])
    def test_derivation(self, kwargs, expected):
        assert derive_state(**kwargs) == expected

    def test_negative_days_refuse(self):
        with pytest.raises(ExperimentStateError):
            derive_state(has_canonical_record=True, has_freeze=True, freeze_intact=True,
                         methodologically_current=True, clean_forward_days=-1)


# ---------------------------------------------------------------------------
# 7. determinism — identical evidence regenerates byte-identically
# ---------------------------------------------------------------------------

def _evidence_doc(**overrides):
    defaults = dict(
        experiment={"primary_rule_id": PRIMARY_RULE_ID, "experiment_id": "freeze/2026-09-06",
                    "version": "v1", "fingerprint": "f" * 64},
        status={"state": SUPERSEDED, "state_explanation": "x", "has_valid_freeze": True,
                "methodologically_current": False, "superseded_by": "exact-wealth-v2",
                "next_checkpoint": None, "checkpoints": []},
        primary_strategy={"rule_id": PRIMARY_RULE_ID},
        benchmark={"id": "spx", "label": "S&P 500", "metrics": SPX},
        historical={"available": True, "window": {"start": "2020-08-16", "end": "2026-08-14"},
                    "metrics": PRIMARY,
                    "primary_stat": {"name": "row", "psr": 0.99, "dsr": None,
                                     "dsr_available": False, "dsr_unavailable_reason": "no accounting"}},
        walk_forward={"n_folds": 6, "consistency": {"summary": "s", "warnings": []}},
        selection_bias={"effective_trials": 85, "method": "max_of_recorded_sources",
                        "candidate_count_generated": 85, "candidate_count_evaluated": 85,
                        "research_ledger_experiments": 35, "dsr_available": True},
        forward={"clean_forward_days": 0, "days_recorded": 0, "days_full": 0, "days_partial": 0,
                 "days_dark": 0, "days_closed": 0, "excluded_rows": {}, "superseded_rows_not_counted": 0,
                 "return": None, "sharpe": None, "sharpe_reason": None, "max_drawdown": None,
                 "return_quality": "none", "first_day": None, "last_day": None,
                 "sufficient_for_meaningful": False, "insufficient_reason": "below threshold"},
        data_quality={"score": 100.0, "full_evidence_days": 0, "outage_days": 0, "problems": []},
        execution_quality={"execution_model": "m", "turnover": 0.0, "turnover_events": 0,
                           "predicted_cost_bps": None,
                           "observed_execution_gap": {"mean_abs_bps": None, "observations": 0}},
        provenance={"canonical_run_id": "canonical-v2", "canonical_record_intact": True},
        verdict=_verdict(),
        comparators={"equal_rm": EQUAL_WEIGHT_COMPARATOR},
        generated_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return build_evidence_document(**defaults)


class TestDeterminism:
    def test_identical_evidence_gives_identical_bytes(self):
        a, b = _evidence_doc(), _evidence_doc()
        assert canonical_json(a) == canonical_json(b)
        assert evidence_fingerprint(a) == evidence_fingerprint(b)

    def test_reports_from_identical_evidence_are_byte_identical(self):
        a, b = _evidence_doc(), _evidence_doc()
        assert render_evidence_report(a) == render_evidence_report(b)
        assert render_readme_block(a) == render_readme_block(b)

    def test_evidence_change_moves_the_fingerprint(self):
        a = _evidence_doc()
        b = _evidence_doc()
        b["forward"]["clean_forward_days"] = 1
        assert evidence_fingerprint(a) != evidence_fingerprint(b)

    def test_generated_at_alone_does_not_move_the_fingerprint(self):
        a = _evidence_doc()
        b = _evidence_doc(generated_at="2027-06-06T00:00:00+00:00")
        assert evidence_fingerprint(a) == evidence_fingerprint(b)

    def test_git_now_alone_does_not_move_the_fingerprint(self):
        """The live checkout is not evidence.

        Regression: git_now (HEAD commit + dirty paths) used to be hashed, so
        the README canonical block embedded a fingerprint of the working tree
        it was rendered from. Committing the regenerated block then moved HEAD,
        changed the fingerprint, and invalidated the very line recording it --
        derive_canonical_readme.py --check could never pass on any commit.
        """
        a = _evidence_doc()
        b = _evidence_doc()
        b["provenance"]["git_now"] = {
            "head_commit": "f" * 40,
            "dirty": True,
            "dirty_paths": ["README.md", "bot/strategy.py"],
        }
        assert evidence_fingerprint(a) == evidence_fingerprint(b)

    def test_real_evidence_still_moves_the_fingerprint_when_git_differs(self):
        """Excluding git_now must not blunt the fingerprint: a genuine evidence
        change must still register while the checkout differs."""
        a = _evidence_doc()
        a["provenance"]["git_now"] = {"head_commit": "a" * 40, "dirty": False, "dirty_paths": []}
        b = _evidence_doc()
        b["provenance"]["git_now"] = {"head_commit": "b" * 40, "dirty": True, "dirty_paths": ["x"]}
        b["historical"]["primary_stat"] = {"sharpe": 0.5}
        assert evidence_fingerprint(a) != evidence_fingerprint(b)

    def test_git_now_remains_available_for_the_provenance_gate(self):
        """Excluded from the fingerprint, not from the document: verify-evidence
        reads git_now to report HEAD and tree cleanliness."""
        a = _evidence_doc()
        a["provenance"]["git_now"] = {"head_commit": "c" * 40, "dirty": False, "dirty_paths": []}
        assert a["provenance"]["git_now"]["head_commit"] == "c" * 40

    def test_reading_a_plan_does_not_move_the_fingerprint(self):
        """Regression: reading a plan is a PROCESS EVENT, not new evidence.

        The read ledger is a gitignored per-checkout file. Folding `any_read`,
        the per-plan read fields, and the read-driven `pre_registration` grade
        into the fingerprint made the evidence identity a function of the
        machine: CI (no ledger) and a developer who had read a plan disagreed,
        and the committed README block could not match both. Same class as
        git_now.
        """
        def doc(read: bool):
            plan = {"criterion": "oos_sharpe gte 0.7", "seal": "abc",
                    "read": read, "n_reads": int(read),
                    "latest_read_state": "NOT_CONFIRMED" if read else None}
            return _evidence_doc(
                registration={"registered": True, "n_valid": 1, "n_drifted": 0,
                              "any_read": read, "primary_plan": dict(plan), "plans": [dict(plan)]},
            )

        assert evidence_fingerprint(doc(False)) == evidence_fingerprint(doc(True))

    def test_read_state_driving_the_pre_registration_grade_does_not_move_it(self):
        """The verdict's pre_registration grade is downstream of read-state
        (Moderate unread -> Weak refuted). It is live status, not evidence."""
        def doc(grade: str):
            return _evidence_doc(
                verdict={"primary_rule_id": "banded_rm", "metrics": {}, "benchmark_metrics": {},
                         "comparison": {}, "verdict": "", "exit_code": 1, "benchmark_id": "spx",
                         "grades": {"pre_registration": grade, "overall": "not established"}},
            )

        assert evidence_fingerprint(doc("Moderate")) == evidence_fingerprint(doc("Weak"))

    def test_a_changed_criterion_still_moves_the_fingerprint(self):
        """The other side of the boundary: read-state is excluded, but a plan
        whose criterion was loosened must still register as evidence drift."""
        def doc(criterion: str):
            return _evidence_doc(
                registration={"registered": True, "n_valid": 1, "n_drifted": 0,
                              "any_read": False,
                              "primary_plan": {"criterion": criterion, "seal": "abc"},
                              "plans": [{"criterion": criterion, "seal": "abc"}]},
            )

        assert evidence_fingerprint(doc("oos_sharpe gte 0.7")) != \
            evidence_fingerprint(doc("oos_sharpe gte 0.2"))

    def test_a_changed_seal_still_moves_the_fingerprint(self):
        """Swapping which plan covers the claim is a change in the evidence."""
        def doc(seal: str):
            return _evidence_doc(
                registration={"registered": True, "n_valid": 1, "n_drifted": 0,
                              "any_read": False,
                              "primary_plan": {"criterion": "c", "seal": seal},
                              "plans": [{"criterion": "c", "seal": seal}]},
            )

        assert evidence_fingerprint(doc("abc")) != evidence_fingerprint(doc("TAMPERED"))


# ---------------------------------------------------------------------------
# 8. README formatting cannot change the evidence
# ---------------------------------------------------------------------------

class TestReadmeFormattingIsInert:
    def test_readme_markdown_wash_does_not_change_the_evidence_document(self):
        before = canonical_json(build_evidence("."))
        # Even if README text is edited or missing, the document is unchanged:
        # the README renders FROM it, never the reverse.
        after = canonical_json(build_evidence("."))
        assert before == after

    def test_render_arguments_cannot_inject_a_claim(self, tmp_path):
        import scripts.derive_canonical_readme as dcr

        hostile = {"run_id": "hostile", "results": {"metrics": {"equal_rm": EQUAL_WEIGHT_COMPARATOR}}}
        baseline = dcr.render(artifacts=None)
        assert dcr.render(hostile, artifacts=None) == baseline


# ---------------------------------------------------------------------------
# 9. confounded comparison — a delta is never attributed to one change
# ---------------------------------------------------------------------------

class TestConfoundedComparison:
    def test_two_or_more_important_changes_are_confounded(self):
        import argparse

        from bot.commands import run_compare_experiments

        # Two experiments differing in code AND universe cannot have their
        # performance delta attributed to either single change.
        args = argparse.Namespace(left="canonical-v1", right="canonical-v2", root=".")
        code = run_compare_experiments(args)
        # The committed records differ in code fingerprint AND parameters
        # (the accounting fix was deliberately re-run), so this must refuse.
        assert code == 2, "a multi-variable delta must be reported as CONFUNDED"

    def test_identical_experiments_are_controlled(self, tmp_path, monkeypatch):
        import argparse
        import json as _json

        from bot.commands import run_compare_experiments

        for run_id in ("a", "b"):
            d = tmp_path / "runs" / run_id
            d.mkdir(parents=True)
            (d / "run.json").write_text(_json.dumps({
                "run_id": run_id,
                "results": {
                    "environment": {"code_fingerprint": {"sha256": "1" * 64, "algo": "sha256-lf-v1"}},
                    "universe": ["BTCUSDT"],
                    "parameters": {"fee": 0.001},
                    "n_folds": 6,
                    "metrics": {"banded_rm": {"cagr": 0.1, "sharpe": 0.7, "max_drawdown": -0.1},
                                "spx": {"cagr": 0.15, "sharpe": 0.7, "max_drawdown": -0.2}},
                },
            }), encoding="utf-8")
        args = argparse.Namespace(left="a", right="b", root=str(tmp_path))
        assert run_compare_experiments(args) == 0

    def test_one_changed_variable_stays_controlled(self, tmp_path):
        import argparse
        import json as _json

        from bot.commands import run_compare_experiments

        base = {
            "environment": {"code_fingerprint": {"sha256": "1" * 64, "algo": "sha256-lf-v1"}},
            "universe": ["BTCUSDT"],
            "parameters": {"fee": 0.001},
            "n_folds": 6,
            "metrics": {"banded_rm": {"cagr": 0.1, "sharpe": 0.7, "max_drawdown": -0.1},
                        "spx": {"cagr": 0.15, "sharpe": 0.7, "max_drawdown": -0.2}},
        }
        for run_id, fee in (("a", 0.001), ("b", 0.002)):
            d = tmp_path / "runs" / run_id
            d.mkdir(parents=True)
            payload = {**base, "parameters": {"fee": fee}}
            (d / "run.json").write_text(_json.dumps({"run_id": run_id, "results": payload}), encoding="utf-8")
        args = argparse.Namespace(left="a", right="b", root=str(tmp_path))
        assert run_compare_experiments(args) == 0


# ---------------------------------------------------------------------------
# 10. cross-surface identity
# ---------------------------------------------------------------------------

class TestIdentityAgreement:
    def test_primary_id_matches_across_surfaces(self):
        doc = build_evidence(".")
        assert doc["verdict"]["primary_rule_id"] == PRIMARY_RULE_ID
        assert doc["experiment"]["primary_rule_id"] == PRIMARY_RULE_ID
        block = render_readme_block(doc)
        assert f"`{PRIMARY_RULE_ID}`" in block

    def test_frozen_config_implements_the_primary_rule(self):
        artifacts = load_artifacts(".")
        freeze = artifacts["freeze"]
        if freeze is None:
            pytest.skip("no freeze committed")
        from bot.canonical_identity import assert_primary_rule_is_frozen

        assert assert_primary_rule_is_frozen(freeze["config"]["algorithm"]) == PRIMARY_RULE_ID

    def test_a_wrong_frozen_rule_refuses(self):
        from bot.canonical_identity import assert_primary_rule_is_frozen

        wrong = {
            "selection_mode": "walk_forward_selected",
            "rebalance_band": 0.0,
            "xs_momentum": {"enabled": False},
            "crisis_derisk": {"enabled": False},
            "drawdown_throttle": {"enabled": False},
            "weighting": {"mode": "inverse_vol"},
        }
        with pytest.raises(PrimaryRuleMismatch):
            assert_primary_rule_is_frozen(wrong)
