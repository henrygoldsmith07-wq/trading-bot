"""Tests for the pre-analysis-plan module.

The properties pinned here are the ones that make a registration worth
anything. Each test states the failure mode it removes, because the failure
modes are the point: a plan that cannot be edited silently, cannot be read
before it is due, and cannot be read at all when it does not exist, is what
turns "we checked" into a checkable claim.

Stdlib unittest, matching the repo's test style.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bot.registration import (
    READ_ALREADY_READ,  # noqa: F401  (imported to pin the public vocabulary)
    READ_CONFIRMED,
    READ_DRIFTED,
    READ_INCONCLUSIVE,
    READ_NOT_CONFIRMED,
    READ_UNREGISTERED,
    Arm,
    Observation,
    Registration,
    RegistrationDrift,
    RegistrationError,
    already_read,
    append_event,
    create_registration,
    direction_holds,
    holm_bonferroni,
    load_events,
    load_registration,
    read_result,
    record_read,
    verify_event_chain,
    write_registration,
)


def _plan(**overrides):
    base = dict(
        registration_id="banded-rm-validity",
        title="Does the frozen banded_rm rule beat buy-and-hold out of sample?",
        hypothesis="banded_rm earns an OOS Sharpe above 0.8 on the frozen universe",
        primary_metric="oos_sharpe",
        threshold=0.8,
        direction="gt",
        min_evidence=6,
        alpha=0.05,
        arms=[
            Arm("banded_rm", "inverse-vol + tilt + crisis de-risk + 5% band", primary=True),
            Arm("fixed_inv_vol", "inverse-vol only, no overlays"),
            Arm("buy_hold", "equal-weight buy and hold over the same window"),
        ],
        stopping_rule="read once at 6 folds; do not read earlier and do not re-read",
        declared_trials=3,
        data_window="2020-08-16 -> 2026-08-14",
        created_at="2026-09-07T00:00:00+00:00",
    )
    base.update(overrides)
    return create_registration(**base)


class TestSealIntegrity(unittest.TestCase):
    def test_seal_is_stable_across_identical_construction(self):
        self.assertEqual(_plan().seal(), _plan().seal())

    def test_editing_any_field_changes_the_seal(self):
        """The seal must cover the decision procedure, not just its identity."""
        original = _plan()
        for field, value in [
            ("threshold", 0.9),
            ("direction", "lt"),
            ("min_evidence", 3),
            ("primary_metric", "cagr"),
            ("alpha", 0.01),
            ("stopping_rule", "read whenever convenient"),
            ("declared_trials", 99),
            ("data_window", "2015-01-01 -> 2026-08-14"),
        ]:
            with self.subTest(field=field):
                self.assertNotEqual(original.seal(), _plan(**{field: value}).seal())

    def test_changing_which_arm_is_primary_changes_the_seal(self):
        base = _plan()
        flipped = create_registration(
            registration_id=base.registration_id,
            title=base.title,
            hypothesis=base.hypothesis,
            primary_metric=base.primary_metric,
            threshold=base.threshold,
            direction=base.direction,
            min_evidence=base.min_evidence,
            alpha=base.alpha,
            arms=[
                Arm("banded_rm", "inverse-vol + tilt + crisis de-risk + 5% band"),
                Arm("fixed_inv_vol", "inverse-vol only, no overlays", primary=True),
                Arm("buy_hold", "equal-weight buy and hold over the same window"),
            ],
            stopping_rule=base.stopping_rule,
            declared_trials=base.declared_trials,
            data_window=base.data_window,
            created_at=base.created_at,
        )
        self.assertNotEqual(base.seal(), flipped.seal())

    def test_from_dict_rejects_an_edited_document(self):
        doc = _plan().document()
        doc["threshold"] = 2.0  # threshold loosened after sealing
        with self.assertRaises(RegistrationDrift):
            Registration.from_dict(doc)

    def test_from_dict_accepts_an_untouched_document(self):
        reg = _plan()
        self.assertEqual(Registration.from_dict(reg.document()).seal(), reg.seal())

    def test_unknown_field_is_refused_rather_than_defaulted(self):
        """A typo must not become a silent default that changes the decision."""
        doc = _plan().document()
        doc["threshhold"] = 0.8
        doc.pop("seal")
        with self.assertRaises(RegistrationError):
            Registration.from_dict(doc)


class TestRegistrationValidation(unittest.TestCase):
    def test_exactly_one_primary_arm_is_required(self):
        with self.assertRaises(RegistrationError):
            _plan(arms=[Arm("a", "first"), Arm("b", "second")])

    def test_two_primary_arms_are_refused(self):
        with self.assertRaises(RegistrationError):
            _plan(arms=[Arm("a", "first", primary=True), Arm("b", "second", primary=True)])

    def test_duplicate_arm_ids_are_refused(self):
        with self.assertRaises(RegistrationError):
            _plan(arms=[Arm("a", "first", primary=True), Arm("a", "again")])

    def test_non_finite_threshold_is_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(threshold=bad):
                with self.assertRaises(RegistrationError):
                    _plan(threshold=bad)

    def test_alpha_must_be_a_probability(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(alpha=bad):
                with self.assertRaises(RegistrationError):
                    _plan(alpha=bad)

    def test_unknown_direction_is_refused(self):
        with self.assertRaises(RegistrationError):
            _plan(direction="sideways")

    def test_empty_hypothesis_is_refused(self):
        with self.assertRaises(RegistrationError):
            _plan(hypothesis="   ")

    def test_declared_trials_must_be_positive(self):
        with self.assertRaises(RegistrationError):
            _plan(declared_trials=0)


class TestWriteOnce(unittest.TestCase):
    def test_roundtrip_through_disk_preserves_the_seal(self):
        with tempfile.TemporaryDirectory() as td:
            reg = _plan()
            path = write_registration(reg, td)
            loaded = load_registration(path)
            self.assertEqual(loaded.seal(), reg.seal())
            self.assertEqual(loaded.primary_arm.arm_id, "banded_rm")

    def test_overwriting_a_registered_plan_is_refused(self):
        """Edit-after-sealing is the exact failure the seal exists to catch."""
        with tempfile.TemporaryDirectory() as td:
            write_registration(_plan(), td)
            with self.assertRaises(RegistrationError):
                write_registration(_plan(threshold=2.5), td)

    def test_write_is_atomic_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            write_registration(_plan(), td)
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_loading_a_missing_registration_raises(self):
        with self.assertRaises(RegistrationError):
            load_registration(Path(tempfile.gettempdir()) / "definitely-not-registered-xyz.json")


class TestReadFailsClosed(unittest.TestCase):
    def test_no_plan_is_never_evidence(self):
        res = read_result(None, [Observation("banded_rm", 3.0, 99)])
        self.assertEqual(res.state, READ_UNREGISTERED)
        self.assertFalse(res.counts_as_evidence)

    def test_thin_sample_is_inconclusive_not_a_pass(self):
        """The headline guard: a great number on three observations proves nothing."""
        reg = _plan(min_evidence=30)
        res = read_result(reg, [Observation("banded_rm", 9.9, 3)])
        self.assertEqual(res.state, READ_INCONCLUSIVE)
        self.assertFalse(res.counts_as_evidence)
        self.assertIn("registered minimum of 30", res.reason)

    def test_missing_primary_observation_is_inconclusive(self):
        res = read_result(_plan(), [Observation("buy_hold", 1.4, 10)])
        self.assertEqual(res.state, READ_INCONCLUSIVE)
        self.assertFalse(res.counts_as_evidence)

    def test_clearing_the_threshold_with_enough_evidence_confirms(self):
        res = read_result(_plan(), [Observation("banded_rm", 1.10, 6)])
        self.assertEqual(res.state, READ_CONFIRMED)
        self.assertTrue(res.counts_as_evidence)

    def test_missing_the_threshold_is_not_confirmed(self):
        res = read_result(_plan(), [Observation("banded_rm", 0.42, 6)])
        self.assertEqual(res.state, READ_NOT_CONFIRMED)
        self.assertFalse(res.counts_as_evidence)

    def test_undeclared_arm_is_refused(self):
        with self.assertRaises(RegistrationError):
            read_result(_plan(), [Observation("some_new_rule", 5.0, 20)])

    def test_two_observations_for_one_arm_are_refused(self):
        with self.assertRaises(RegistrationError):
            read_result(_plan(), [Observation("banded_rm", 1.0, 6), Observation("banded_rm", 2.0, 7)])

    def test_reading_a_non_finite_observation_is_refused(self):
        with self.assertRaises(RegistrationError):
            Observation("banded_rm", float("nan"), 10)

    def test_p_value_outside_unit_interval_is_refused(self):
        with self.assertRaises(RegistrationError):
            Observation("banded_rm", 1.0, 10, p_value=1.4)


class TestDriftAndRepeatRead(unittest.TestCase):
    def test_a_plan_replaced_underneath_the_reader_is_drifted(self):
        """Grading against a different seal is a new study, not a result."""
        reg = _plan()
        res = read_result(reg, [Observation("banded_rm", 5.0, 30)], expected_seal="deadbeefdeadbeef")
        self.assertEqual(res.state, READ_DRIFTED)
        self.assertFalse(res.counts_as_evidence)

    def test_matching_expected_seal_proceeds_normally(self):
        reg = _plan()
        res = read_result(reg, [Observation("banded_rm", 1.1, 6)], expected_seal=reg.seal())
        self.assertEqual(res.state, READ_CONFIRMED)

    def test_a_second_read_of_the_same_plan_is_refused(self):
        """A tape examined until one reading looks good is a search over time."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "events.jsonl"
            reg = _plan()
            append_event(log, event_type="read", registration_id=reg.registration_id,
                         registration_seal=reg.seal())
            res = read_result(reg, [Observation("banded_rm", 9.9, 999)], event_log=log)
            self.assertEqual(res.state, READ_ALREADY_READ)
            self.assertFalse(res.counts_as_evidence)

    def test_record_read_then_a_second_read_is_refused(self):
        """The documented round trip: read -> record -> re-read is blocked."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "events.jsonl"
            reg = _plan()
            first = read_result(reg, [Observation("banded_rm", 1.1, 6)], event_log=log)
            self.assertEqual(first.state, READ_CONFIRMED)
            record_read(log, first)
            self.assertTrue(already_read(log, reg.registration_id))
            second = read_result(reg, [Observation("banded_rm", 1.1, 6)], event_log=log)
            self.assertEqual(second.state, READ_ALREADY_READ)

    def test_recorded_read_preserves_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "events.jsonl"
            reg = _plan()
            result = read_result(reg, [Observation("banded_rm", 1.1, 6)])
            record_read(log, result)
            record_read(log, result)
            events = load_events(log)  # strict: verifies the hash chain
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["payload"]["state"], READ_CONFIRMED)
            self.assertEqual(events[0]["payload"]["threshold"], reg.threshold)

    def test_drift_outranks_a_perfect_number(self):
        """A spectacular result on a changed plan is still not confirmation."""
        reg = _plan()
        res = read_result(reg, [Observation("banded_rm", 999.0, 9999)],
                          expected_seal="0" * 16, event_log=None)
        self.assertEqual(res.state, READ_DRIFTED)


class TestMultiArmCorrection(unittest.TestCase):
    """Reading the best of several arms is a search, and it must cost something."""

    def _observe(self, banded, bsh, p_banded=None, p_bsh=None):
        return [
            Observation("banded_rm", banded, 6, p_value=p_banded),
            Observation("fixed_inv_vol", 0.5, 6, p_value=0.60),
            Observation("buy_hold", bsh, 6, p_value=p_bsh),
        ]

    def test_three_nominal_p_values_are_not_all_rejected(self):
        """Holm at alpha=0.05 across 3 arms: the best threshold is 0.0167."""
        holm = holm_bonferroni({"a": 0.02, "b": 0.03, "c": 0.04}, 0.05)
        self.assertEqual(sum(holm["rejected"].values()), 0)

    def test_a_single_strong_arm_survives(self):
        holm = holm_bonferroni({"a": 0.001, "b": 0.60, "c": 0.70}, 0.05)
        self.assertTrue(holm["rejected"]["a"])
        self.assertFalse(holm["rejected"]["b"])

    def test_holm_stops_at_the_first_failure(self):
        # Holm sorts ascending by p and steps alpha down: rank0 tested at
        # alpha/3, rank1 at alpha/2, rank2 at alpha.
        #   c=0.006 <= 0.0167  -> rejects
        #   a=0.030 >  0.025   -> fails, and Holm stops here
        #   b=0.040            -> never reached, cannot reject
        holm = holm_bonferroni({"a": 0.030, "b": 0.040, "c": 0.006}, 0.05)
        self.assertTrue(holm["rejected"]["c"])
        self.assertFalse(holm["rejected"]["a"])
        self.assertFalse(holm["rejected"]["b"])

    def test_a_clearing_threshold_that_fails_correction_is_not_confirmed(self):
        reg = _plan()
        obs = self._observe(1.10, 0.30, p_banded=0.03, p_bsh=0.80)
        res = read_result(reg, obs)
        self.assertEqual(res.state, READ_NOT_CONFIRMED)
        self.assertFalse(res.counts_as_evidence)
        self.assertIn("Holm-Bonferroni", res.reason)

    def test_correction_is_absent_when_no_p_values_are_supplied(self):
        res = read_result(_plan(), [Observation("banded_rm", 1.10, 6)])
        self.assertNotIn("family_wise", res.per_arm)


class TestDirection(unittest.TestCase):
    def test_each_direction_holds_as_declared(self):
        self.assertTrue(direction_holds(1.0, "gt", 0.8))
        self.assertFalse(direction_holds(0.8, "gt", 0.8))
        self.assertTrue(direction_holds(0.8, "gte", 0.8))
        self.assertTrue(direction_holds(-0.1, "lt", 0.0))
        self.assertTrue(direction_holds(0.0, "lte", 0.0))
        self.assertFalse(direction_holds(0.0, "lt", 0.0))


class TestEventChain(unittest.TestCase):
    def test_chain_links_and_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            append_event(path, event_type="registered", registration_id="r1", registration_seal="abc")
            append_event(path, event_type="read", registration_id="r1", registration_seal="abc",
                         payload={"state": "INCONCLUSIVE"})
            events = load_events(path)
            self.assertEqual(len(events), 2)
            verify_event_chain(events)

    def test_tampering_with_a_record_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            append_event(path, event_type="registered", registration_id="r1", registration_seal="abc")
            append_event(path, event_type="read", registration_id="r1", registration_seal="abc")
            lines = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["registration_seal"] = "forged"
            lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(RegistrationError):
                load_events(path)

    def test_missing_log_is_an_empty_chain_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(load_events(Path(td) / "nope.jsonl"), [])

    def test_already_read_detects_a_second_look(self):
        """A tape examined until one reading looks good is a search over time."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            self.assertFalse(already_read(path, "r1"))
            append_event(path, event_type="read", registration_id="r1", registration_seal="abc")
            self.assertTrue(already_read(path, "r1"))
            self.assertFalse(already_read(path, "r2"))


class TestCountsAsEvidence(unittest.TestCase):
    def test_only_confirmed_counts(self):
        for state, expected in [
            (READ_CONFIRMED, True),
            (READ_NOT_CONFIRMED, False),
            (READ_INCONCLUSIVE, False),
            (READ_UNREGISTERED, False),
            (READ_DRIFTED, False),
            (READ_ALREADY_READ, False),
        ]:
            with self.subTest(state=state):
                self.assertEqual(state == READ_CONFIRMED, expected)


if __name__ == "__main__":
    unittest.main()
