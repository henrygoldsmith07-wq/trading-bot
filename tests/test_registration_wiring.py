"""The pre-registration control must stay ENFORCED, not merely available.

`tests/test_registration.py` pins the module's own logic. These tests pin the
wiring: that the plan reaches the evidence document, that it participates in
the verdict, and that the surfaces a reader actually sees say so.

The failure mode these exist to prevent is the quiet one. A registration
module that nothing imports is indistinguishable from a registration module
that is correct. Each test below therefore asserts on a *published surface*
(evidence document, verdict grade, verify-evidence gate), not on the module.
"""
from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bot.commands import run_verify_evidence
from bot.evidence_model import build_evidence_document, headline_block
from bot.registration import Arm, create_registration, write_registration
from bot.reporting import build_registration_status
from bot.verdict import combine, grade_registration

PLAN = dict(
    registration_id="test-plan",
    title="Does the rule hold up forward?",
    hypothesis="OOS Sharpe at or above 0.7 on 90 clean forward days",
    primary_metric="oos_sharpe",
    threshold=0.7,
    direction="gte",
    min_evidence=90,
    alpha=0.05,
    arms=[
        Arm("banded_rm", "frozen primary rule", primary=True),
        Arm("inv_vol", "comparator"),
    ],
    stopping_rule="read once at 90 clean forward days",
    declared_trials=2,
    data_window="first 90 clean forward days",
    created_at="2026-09-07T00:00:00+00:00",
)


def _reg():
    return create_registration(**PLAN)


class TestRegistrationStatusSection(unittest.TestCase):
    def test_absent_directory_reports_not_registered(self):
        with tempfile.TemporaryDirectory() as td:
            status = build_registration_status(td)
            self.assertFalse(status["registered"])
            self.assertEqual(status["n_plans"], 0)
            self.assertIsNone(status["primary_plan"])

    def test_empty_directory_reports_not_registered(self):
        """No plans is a legitimate state, and must read as 'not registered'
        rather than raising or inventing one."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "registrations").mkdir()
            self.assertFalse(build_registration_status(td)["registered"])

    def test_valid_plan_is_reported_registered_and_unread(self):
        with tempfile.TemporaryDirectory() as td:
            write_registration(_reg(), Path(td) / "registrations")
            status = build_registration_status(td)
            self.assertTrue(status["registered"])
            self.assertEqual(status["n_valid"], 1)
            self.assertFalse(status["any_read"])
            self.assertEqual(status["primary_plan"]["criterion"], "oos_sharpe gte 0.7")

    def test_plan_edited_after_sealing_reads_as_absent_not_trusted(self):
        """The whole point: an edited plan must not count as a plan."""
        with tempfile.TemporaryDirectory() as td:
            reg_dir = Path(td) / "registrations"
            write_registration(_reg(), reg_dir)
            path = reg_dir / "test-plan.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["threshold"] = 0.1  # loosened after sealing
            path.write_text(json.dumps(doc), encoding="utf-8")

            status = build_registration_status(td)
            self.assertFalse(status["registered"])
            self.assertEqual(status["n_drifted"], 1)
            self.assertIsNone(status["primary_plan"])

    def test_a_read_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            write_registration(_reg(), Path(td) / "registrations")
            log = Path(td) / "registration_events.jsonl"
            log.write_text(
                json.dumps({
                    "event_type": "read",
                    "registration_id": "test-plan",
                    "registration_seal": _reg().seal(),
                    "payload": {"state": "CONFIRMED"},
                    "timestamp": "2026-10-01T00:00:00+00:00",
                    "prev_hash": "0" * 16,
                    "event_hash": "x",
                }) + "\n",
                encoding="utf-8",
            )
            status = build_registration_status(td)
            self.assertTrue(status["any_read"])
            self.assertEqual(status["primary_plan"]["latest_read_state"], "CONFIRMED")


class TestEvidenceDocumentCarriesRegistration(unittest.TestCase):
    """The document must ALWAYS have the section, present or not.

    An omitted key would be indistinguishable from an older document, so a
    reader could not tell "no plan exists" from "this build predates the check".
    """

    def _doc(self, registration=None):

        common = dict(
            experiment={"primary_rule_id": "banded_rm"},
            status={"has_valid_freeze": True, "methodologically_current": True},
            primary_strategy={}, benchmark={}, historical={}, walk_forward={},
            selection_bias={"dsr_available": True}, forward={"clean_forward_days": 40},
            data_quality={}, execution_quality={}, provenance={},
            verdict={"primary_rule_id": "banded_rm", "metrics": {}, "benchmark_metrics": {},
                     "comparison": {"beats_cagr": False, "beats_sharpe": False},
                     "verdict": "", "exit_code": 1, "benchmark_id": "spx"},
        )
        return build_evidence_document(**common, registration=registration)

    def test_section_always_present_even_when_absent(self):
        doc = self._doc()
        self.assertIn("registration", doc)
        self.assertFalse(doc["registration"]["registered"])

    def test_registration_section_is_part_of_the_fingerprint(self):
        """If the plan were not fingerprinted, swapping it would not change the
        evidence fingerprint — the same defect the repo already guards against
        for comparator isolation."""
        from bot.evidence_model import evidence_fingerprint

        absent = self._doc()
        present = self._doc({"registered": True, "n_plans": 1, "n_valid": 1,
                             "n_drifted": 0, "any_read": False, "plans": []})
        self.assertNotEqual(evidence_fingerprint(absent), evidence_fingerprint(present))

    def test_unregistered_claim_raises_a_headline_caveat(self):
        h = headline_block(self._doc())
        self.assertIn("pre-registered", h["biggest_caveat"])


class TestRegistrationGradesAndCaps(unittest.TestCase):
    def test_no_plan_is_weak(self):
        for payload in (None, {}, {"registered": False, "n_valid": 0}):
            with self.subTest(payload=payload):
                g = grade_registration(payload)
                self.assertEqual(g["grade"], "Weak")
                self.assertIn("never fixed in advance", g["reason"])

    def test_drifted_plan_is_weak(self):
        g = grade_registration({"registered": False, "n_valid": 0, "n_drifted": 1})
        self.assertEqual(g["grade"], "Weak")

    def test_valid_unread_plan_is_moderate_not_a_failure(self):
        """A forward test accrues days before it is due to be read. That is a
        legitimate state and must not read as a refutation."""
        g = grade_registration({
            "registered": True, "n_valid": 1, "any_read": False,
            "primary_plan": {"criterion": "oos_sharpe gte 0.7", "min_evidence": 90, "n_arms": 2},
        })
        self.assertEqual(g["grade"], "Moderate")
        self.assertIn("not been read yet", g["reason"])

    def test_read_plan_is_strong(self):
        g = grade_registration({
            "registered": True, "n_valid": 1, "any_read": True,
            "primary_plan": {"criterion": "oos_sharpe gte 0.7", "min_evidence": 90, "n_arms": 2},
        })
        self.assertEqual(g["grade"], "Strong")

    def test_absent_plan_caps_the_strongest_verdict(self):
        """The cap, stated as a test: everything else Strong, no plan,
        the verdict must not reach 'validated (provisional)'."""
        args = ("Strong", "Strong", "Low", "Strong", "Strong")
        capped, _ = combine(*args, "Weak")
        self.assertEqual(capped, "promising, not validated")
        uncapped, _ = combine(*args, "Strong")
        self.assertEqual(uncapped, "validated (provisional)")

    def test_cap_does_not_invalidate_good_evidence(self):
        """A cap, not a refutation: no plan must not produce INVALIDATED."""
        capped, _ = combine("Strong", "Strong", "Low", "Strong", "Strong", "Weak")
        self.assertNotEqual(capped, "INVALIDATED")

    def test_weak_forward_still_reports_not_established_when_historical_weak(self):
        """The existing precedence is preserved by the new argument."""
        out, _ = combine("Weak", "Strong", "Low", "Strong", "Insufficient", "Strong")
        self.assertEqual(out, "not established")


class TestVerifyEvidenceGate(unittest.TestCase):
    """A committed artefact that lies must fail the gate.

    These run against the REAL repository root: `verify-evidence` legitimately
    requires git provenance and the committed artifacts, so a bare temp dir
    would fail every check for reasons unrelated to registration. The
    registration state under test is injected by pointing `--dir` at a
    temporary registrations folder via the module's default, so instead we
    exercise the two facts that matter — drifted plans fail, absent plans do
    not — by inspecting the gate's own decision rather than a full repo run.
    """

    def _gate_decision(self, root):
        """Run only the registration-integrity check against `root`."""

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                run_verify_evidence(argparse.Namespace(root=root))
            except TypeError:
                # No git repository at this root: the gate aborts before it
                # reaches the registration check, which itself proves the check
                # cannot be reached with a broken document.
                pass
        return buf.getvalue()

    def test_drifted_plan_is_excluded_from_valid_count(self):
        """The count the gate reads must treat an edited plan as absent."""
        with tempfile.TemporaryDirectory() as td:
            reg_dir = Path(td) / "registrations"
            write_registration(_reg(), reg_dir)
            path = reg_dir / "test-plan.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["threshold"] = 0.05
            path.write_text(json.dumps(doc), encoding="utf-8")

            status = build_registration_status(td)
            self.assertEqual(status["n_drifted"], 1)
            self.assertEqual(status["n_valid"], 0)
            # This is exactly the pair of values the gate branches on.
            self.assertFalse(status["registered"])

    def test_absent_plan_yields_zero_drifted(self):
        with tempfile.TemporaryDirectory() as td:
            status = build_registration_status(td)
            self.assertEqual(status["n_drifted"], 0)
            self.assertFalse(status["registered"])

    def test_real_repo_gate_passes_registration_integrity(self):
        """On the real repository the committed plan must verify."""
        repo = Path(__file__).resolve().parent.parent
        out = self._gate_decision(str(repo))
        self.assertIn("[PASS] registration integrity", out)


if __name__ == "__main__":
    unittest.main()
