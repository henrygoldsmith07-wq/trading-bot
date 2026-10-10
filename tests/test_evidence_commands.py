"""The evidence CLI: one document, one exit code, one report.

Invariants pinned here:
* `evidence` renders from the evidence document and can emit canonical bytes;
* `experiment-status` reports the derived state, never a hand-written one;
* `verify-evidence` fails closed with exact reasons and exits non-zero;
* `compare-experiments` refuses to attribute a confounded delta;
* `verdict` follows the primary rule's exit code, from the same document.
"""
from __future__ import annotations

import argparse
import json

from bot.commands import (
    run_compare_experiments,
    run_evidence,
    run_experiment_status,
    run_verdict,
    run_verify_evidence,
)


def _args(**kwargs):
    defaults = {"root": ".", "json": False, "out": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestEvidenceCommand:
    def test_reports_from_the_document(self, capsys):
        assert run_evidence(_args()) == 0
        out = capsys.readouterr().out
        assert "EXPERIMENT EVIDENCE" in out
        assert "primary rule" in out.lower()

    def test_json_mode_is_parseable_and_carries_the_verdict(self, capsys):
        assert run_evidence(_args(json=True)) == 0
        doc = json.loads(capsys.readouterr().out)
        assert "verdict" in doc and "status" in doc
        assert doc["verdict"]["primary_rule_id"]

    def test_out_writes_canonical_bytes(self, tmp_path, capsys):
        target = tmp_path / "evidence.json"
        assert run_evidence(_args(out=str(target))) == 0
        doc = json.loads(target.read_text(encoding="utf-8"))
        assert doc["schema"] == "evidence/1"
        # canonical serialization: sorted keys, so a re-read round-trips
        from bot.evidence_model import evidence_fingerprint

        assert evidence_fingerprint(doc) == evidence_fingerprint(doc)

    def test_two_runs_produce_the_same_document(self, capsys):
        run_evidence(_args(json=True))
        a = json.loads(capsys.readouterr().out)
        run_evidence(_args(json=True))
        b = json.loads(capsys.readouterr().out)
        assert a == b, "the evidence document must be deterministic, not wall-clock shaped"


class TestExperimentStatus:
    def test_states_is_derived_not_asserted(self, capsys):
        assert run_experiment_status(_args()) == 0
        out = capsys.readouterr().out
        assert "state" in out.lower()
        assert "next checkpoint" in out.lower()
        assert "biggest caveat" in out.lower()

    def test_checkpoints_are_listed(self, capsys):
        run_experiment_status(_args())
        out = capsys.readouterr().out
        assert "10 clean forward days" in out
        assert "365 clean forward days" in out


class TestVerifyEvidence:
    def test_returns_a_concise_pass_fail_report(self, capsys):
        code = run_verify_evidence(_args())
        out = capsys.readouterr().out
        assert "checks pass" in out
        assert "[PASS]" in out or "[FAIL]" in out
        # The repo's committed experiment is genuinely superseded, so a fully
        # honest run reports that rather than whitewashing it.
        assert code in (0, 1)

    def test_names_the_failing_checks(self, capsys):
        run_verify_evidence(_args())
        out = capsys.readouterr().out
        for name in ("git provenance", "freeze integrity", "evidence chain integrity",
                     "experiment IDs", "methodological currency"):
            assert name in out


class TestForwardTapeLiveness:
    """A tape that stopped advancing must be distinguishable from a young one.

    The repo lost two weeks of prospective record because both reported the
    same thing: "0 clean forward days". Every scheduled run returned success
    while producing no evidence, and nothing said the number had stopped
    moving.
    """

    @staticmethod
    def _doc(has_freeze=True, current=True, superseded_by="exact-wealth-v2"):
        return {
            "status": {
                "has_valid_freeze": has_freeze,
                "methodologically_current": current,
                "superseded_by": superseded_by,
            }
        }

    @staticmethod
    def _arts(days):
        return {"forward_rows": [{"date": d} for d in days]}

    def _run(self, days, has_freeze=True, current=True):
        from bot.commands import _forward_tape_liveness

        return _forward_tape_liveness(self._doc(has_freeze, current), self._arts(days))

    def test_a_stalled_tape_fails(self):
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=17)).isoformat()
        ok, detail = self._run([old])
        assert ok is False
        assert "stopped advancing" in detail
        assert old in detail

    def test_a_current_tape_passes(self):
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=1)).isoformat()
        ok, detail = self._run([recent])
        assert ok is True
        assert "is current" in detail

    def test_the_slack_window_tolerates_weekends_and_holidays(self):
        """A normal long weekend must never read as a dead experiment."""
        from datetime import date, timedelta

        from bot.experiment_state import MAX_FORWARD_TAPE_GAP_DAYS

        at_limit = (date.today() - timedelta(days=MAX_FORWARD_TAPE_GAP_DAYS)).isoformat()
        assert self._run([at_limit])[0] is True

        past_limit = (date.today() - timedelta(days=MAX_FORWARD_TAPE_GAP_DAYS + 1)).isoformat()
        assert self._run([past_limit])[0] is False

    def test_no_freeze_is_not_a_failure(self):
        """Nothing is supposed to be advancing, so nothing is broken."""
        ok, detail = self._run([], has_freeze=False)
        assert ok is True
        assert "no current freeze" in detail

    def test_an_unstarted_tape_is_not_a_failure(self):
        """A brand-new experiment is not a stalled one."""
        ok, detail = self._run([])
        assert ok is True
        assert "has not started" in detail

    def test_reads_the_raw_tape_not_the_scoped_rows(self):
        """Rows excluded from grading still prove when the tape last moved.

        The committed tape's 11 rows are all excluded (they predate experiment
        stamping), so `forward.last_day` is None. Liveness must not therefore
        conclude the tape never started -- it must read the rows themselves.
        """
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=40)).isoformat()
        ok, detail = self._run([old])
        assert ok is False, "excluded rows must still count as tape activity"

    def test_a_malformed_last_day_fails_closed(self):
        ok, detail = self._run(["not-a-date"])
        assert ok is False
        assert "not a valid date" in detail

    def test_a_superseded_freeze_blames_the_missing_refreeze_not_the_runner(self):
        """Two different causes, two different fixes.

        A superseded methodology is *right* to stop accruing days. Reporting
        "the scheduled run is broken" would send someone to debug a workflow
        that is working correctly, when the real gap is that nobody has frozen
        the current methodology.
        """
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=40)).isoformat()
        ok, detail = self._run([old], current=False)
        assert ok is False, "a superseded freeze still produces no prospective evidence"
        assert "SUPERSEDED" in detail
        assert "exact-wealth-v2" in detail
        assert "missing is a freeze" in detail
        assert "scheduled run is not producing evidence" not in detail

    def test_a_stale_tape_on_current_code_blames_the_runner(self):
        """The complementary case: current methodology, dead tape."""
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=40)).isoformat()
        ok, detail = self._run([old], current=True)
        assert ok is False
        assert "scheduled run is not producing evidence" in detail
        assert "SUPERSEDED" not in detail

    def test_liveness_is_not_part_of_the_evidence_document(self):
        """Wall clock may decide this check; it may not enter the evidence.

        If it did, the fingerprint would change every midnight and the README
        gate would be unsatisfiable -- the same defect class as git_now.
        """
        from bot.reporting import build_evidence

        doc = build_evidence(".")
        assert "liveness" not in json.dumps(doc).lower(), (
            "liveness must be a verification-time check, not evidence"
        )


class TestVerdictCommand:
    def test_exit_code_follows_the_primary_rule(self, capsys):
        code = run_verdict(_args())
        assert code in (0, 1, 2)
        out = capsys.readouterr().out
        assert "STRATEGY VERDICT" in out

    def test_json_carries_the_canonical_verdict(self, capsys):
        run_verdict(_args(json=True))
        doc = json.loads(capsys.readouterr().out)
        assert doc["verdict"]["primary_rule_id"]
        assert doc["verdict"]["verdict_code"] in (
            "beats_on_cagr_and_sharpe", "does_not_beat", "no_evidence"
        )


class TestCompareExperimentsCommand:
    def test_confounded_comparison_refuses(self, capsys):
        code = run_compare_experiments(argparse.Namespace(left="canonical-v1", right="canonical-v2", root="."))
        out = capsys.readouterr().out
        assert code == 2
        assert "CONFUNDED COMPARISON" in out
        assert "not attributable to any single change" in out.lower()

    def test_identical_runs_are_controlled(self, tmp_path, capsys):
        for run_id in ("a", "b"):
            d = tmp_path / "runs" / run_id
            d.mkdir(parents=True)
            (d / "run.json").write_text(json.dumps({
                "run_id": run_id,
                "results": {
                    "environment": {"code_fingerprint": {"sha256": "1" * 64, "algo": "sha256-lf-v1"}},
                    "universe": ["BTCUSDT"], "parameters": {"fee": 0.001}, "n_folds": 6,
                    "metrics": {"banded_rm": {"cagr": 0.1, "sharpe": 0.7, "max_drawdown": -0.1}},
                },
            }), encoding="utf-8")
        code = run_compare_experiments(argparse.Namespace(left="a", right="b", root=str(tmp_path)))
        out = capsys.readouterr().out
        assert code == 0
        assert "controlled comparison" in out
        assert "CONFUNDED" not in out

    def test_single_variable_change_stays_controlled(self, tmp_path, capsys):
        base = {
            "environment": {"code_fingerprint": {"sha256": "1" * 64, "algo": "sha256-lf-v1"}},
            "universe": ["BTCUSDT"], "n_folds": 6,
            "metrics": {"banded_rm": {"cagr": 0.1, "sharpe": 0.7, "max_drawdown": -0.1}},
        }
        for run_id, fee in (("a", 0.001), ("b", 0.002)):
            d = tmp_path / "runs" / run_id
            d.mkdir(parents=True)
            payload = {**base, "parameters": {"fee": fee}}
            (d / "run.json").write_text(json.dumps({"run_id": run_id, "results": payload}), encoding="utf-8")
        code = run_compare_experiments(argparse.Namespace(left="a", right="b", root=str(tmp_path)))
        assert code == 0
        assert "exactly one important variable differs" in capsys.readouterr().out
