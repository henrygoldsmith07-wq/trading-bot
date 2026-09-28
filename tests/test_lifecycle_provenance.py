"""Order-lifecycle timestamp provenance and experiment versioning.

Committed forward rows contain a REAL unit-error bug: `intent_ts` was computed
by adding a millisecond-scale constant to a seconds-scale timestamp, landing
~2.7 years in the future (2029-06-02 for a 2026-09-07 session). Those bytes are
append-only evidence and are never rewritten. What must happen instead is that
the corruption is DETECTED, reported, and kept out of the statistics it would
otherwise contaminate.
"""
import json

import pytest

from bot.experiments import (
    ACCOUNTING_MODEL_ADDITIVE,
    ACCOUNTING_MODEL_EXACT,
    CURRENT_ACCOUNTING_MODEL,
    current_freeze_status,
    describe_freeze,
)
from bot.lifecycle import (
    LIFECYCLE_FIELDS,
    REASON_FUTURE,
    REASON_OUT_OF_ORDER,
    REASON_UNPARSEABLE,
    audit_log_lifecycle,
    lifecycle_problems,
    order_is_trustworthy,
    trusted_orders,
)
from bot.prospective import load_log, slippage_stats


def _order(**over):
    base = {
        "symbol": "BTCUSDT",
        "signal_generated_ts": "2026-09-06T23:59:59+00:00",
        "intent_ts": "2026-09-07T00:00:00+00:00",
        "submitted_ts": "2026-09-07T23:47:10+00:00",
        "fill_ts": "2026-09-07T23:47:10+00:00",
    }
    base.update(over)
    return base


class TestLifecycleOrdering:
    def test_well_formed_order_has_no_problems(self):
        assert lifecycle_problems(_order(), "2026-09-07") == []
        assert order_is_trustworthy(_order(), "2026-09-07")

    def test_late_signal_after_intent_is_flagged(self):
        o = _order(signal_generated_ts="2026-09-08T00:00:00+00:00")
        assert REASON_OUT_OF_ORDER in lifecycle_problems(o, "2026-09-07")

    def test_fill_before_submission_is_flagged(self):
        o = _order(fill_ts="2026-09-07T00:00:00+00:00")
        assert REASON_OUT_OF_ORDER in lifecycle_problems(o, "2026-09-07")

    def test_naive_timestamps_are_rejected(self):
        assert lifecycle_problems(_order(signal_generated_ts="2026-09-06T23:59:59"), "2026-09-07")

    def test_unparseable_timestamps_are_rejected(self):
        assert REASON_UNPARSEABLE in lifecycle_problems(_order(fill_ts="not-a-time"), "2026-09-07")

    def test_missing_fields_are_tolerated_for_legacy_rows(self):
        o = {"symbol": "X", "signal_generated_ts": "2026-09-06T23:59:59+00:00"}
        assert lifecycle_problems(o, "2026-09-07") == []

    def test_every_lifecycle_field_is_checked(self):
        assert set(LIFECYCLE_FIELDS) == {
            "signal_generated_ts", "intent_ts", "submitted_ts", "fill_ts"
        }


class TestLegacyFutureIntent:
    """The committed corruption: intent ~1000 days after the session."""

    LEGACY = _order(intent_ts="2029-06-02T00:00:00+00:00")

    def test_legacy_intent_is_flagged_as_future(self):
        problems = lifecycle_problems(self.LEGACY, "2026-09-07")
        assert REASON_FUTURE in problems
        assert not order_is_trustworthy(self.LEGACY, "2026-09-07")

    def test_legacy_row_is_flagged_even_without_a_session_date(self):
        # Out-of-order alone still catches it: a 2029 intent follows a 2026 signal.
        assert REASON_OUT_OF_ORDER in lifecycle_problems(self.LEGACY)

    def test_a_delayed_but_honest_run_is_still_trusted(self):
        # Submitted six hours late, still inside its own session: legitimate.
        o = _order(submitted_ts="2026-09-07T18:00:00+00:00", fill_ts="2026-09-07T18:00:01+00:00")
        assert order_is_trustworthy(o, "2026-09-07")

    def test_seconds_versus_milliseconds_is_the_documented_cause(self):
        # 86_400_000 seconds is ~2.7 years: a ms constant added to a seconds
        # timestamp. Proof this is a unit error, not a future-dated session.
        from datetime import datetime

        skew = (datetime.fromisoformat("2029-06-02T00:00:00+00:00")
                - datetime.fromisoformat("2026-09-07T00:00:00+00:00")).total_seconds()
        assert 900 * 86_400 < skew < 1100 * 86_400


class TestCommittedForwardTape:
    """The real committed log must be detected, preserved, and quarantined."""

    def test_committed_log_contains_the_known_corruption(self):
        audit = audit_log_lifecycle(load_log("forward_log.jsonl"))
        assert audit["n_orders"] > 0
        assert audit["n_orders_flagged"] == audit["n_orders"]
        assert REASON_FUTURE in audit["flagged_by_reason"]

    def test_committed_log_is_not_mutated_by_auditing(self):
        before = open("forward_log.jsonl", "rb").read()
        audit_log_lifecycle(load_log("forward_log.jsonl"))
        assert open("forward_log.jsonl", "rb").read() == before

    def test_corrupted_orders_are_excluded_from_slippage_stats(self):
        stats = slippage_stats(load_log("forward_log.jsonl"))
        # every order in the tape is untrustworthy, so none may inform slippage
        assert stats["count"] == 0
        assert stats["mean_abs_bps"] is None
        assert stats["excluded_lifecycle_orders"] > 0
        assert REASON_FUTURE in stats["lifecycle_problems"]

    def test_trusted_orders_is_empty_for_the_committed_tape(self):
        assert trusted_orders(load_log("forward_log.jsonl")) == []

    def test_mixed_tape_keeps_only_the_trustworthy_orders(self):
        entries = [{
            "date": "2026-09-07",
            "orders": [_order(symbol="GOOD"),
                       _order(symbol="BAD", intent_ts="2029-01-01T00:00:00+00:00")],
        }]
        assert [o["symbol"] for o in trusted_orders(entries)] == ["GOOD"]
        audit = audit_log_lifecycle(entries)
        assert audit["n_orders"] == 2 and audit["n_orders_trusted"] == 1


class TestExperimentVersioning:
    """v1 is preserved and honestly labelled as superseded."""

    def test_committed_freeze_is_recognised_as_v1_and_superseded(self):
        manifest = json.loads(open("freeze.json", encoding="utf-8").read())
        d = describe_freeze(manifest)
        assert d["experiment_version"] == "v1"
        assert d["accounting_model"] == ACCOUNTING_MODEL_ADDITIVE
        assert d["methodologically_current"] is False
        assert d["superseded_by"] == ACCOUNTING_MODEL_EXACT
        assert d["supersede_reason"]

    def test_committed_freeze_is_byte_untouched(self):
        manifest = json.loads(open("freeze.json", encoding="utf-8").read())
        assert "accounting_model" not in manifest      # never rewritten in place
        assert "experiment_version" not in manifest
        assert manifest["frozen_at_date"] == "2026-09-06"

    def test_a_v2_manifest_is_current(self):
        d = describe_freeze({
            "experiment_version": "v2",
            "accounting_model": ACCOUNTING_MODEL_EXACT,
            "frozen_at_date": "2026-10-01",
        })
        assert d["methodologically_current"] is True
        assert d["status"] == "current"
        assert d["superseded_by"] is None
        assert CURRENT_ACCOUNTING_MODEL == ACCOUNTING_MODEL_EXACT

    def test_status_text_says_superseded_not_invalid(self):
        text = current_freeze_status({"experiment_version": "v1", "frozen_at_date": "2026-09-06"})
        assert "SUPERSEDED" in text
        assert "NOT of the corrected implementation" in text

    def test_status_text_says_current_for_v2(self):
        assert "methodologically current" in current_freeze_status(
            {"experiment_version": "v2", "accounting_model": ACCOUNTING_MODEL_EXACT}
        )


class TestSlippageStatsWithGoodData:
    def test_a_day_with_good_orders_keeps_all_its_asset_observations(self):
        entries = [{
            "date": "2026-09-07",
            "assets": {"A": {"slippage_bps": 12.0}, "B": {"slippage_bps": -8.0}},
            "orders": [_order(symbol="A")],
        }]
        stats = slippage_stats(entries)
        assert stats["count"] == 2
        assert stats["mean_abs_bps"] == pytest.approx(10.0)
        assert stats["excluded_lifecycle_orders"] == 0

    def test_a_day_with_corrupt_orders_drops_that_day_only(self):
        entries = [
            {"date": "2026-09-07",
             "assets": {"A": {"slippage_bps": 12.0}},
             "orders": [_order(symbol="A")]},
            {"date": "2026-09-08",
             "assets": {"A": {"slippage_bps": 99.0}},
             "orders": [_order(symbol="A", intent_ts="2029-01-01T00:00:00+00:00")]},
        ]
        stats = slippage_stats(entries)
        assert stats["count"] == 1
        assert stats["mean_abs_bps"] == pytest.approx(12.0)
        assert stats["excluded_lifecycle_orders"] == 1

    def test_entries_without_orders_keep_their_observations(self):
        # Older rows have no `orders` array, so there is no lifecycle metadata
        # to invalidate: their asset-level slippage must still count.
        entries = [{"date": "2026-09-07", "assets": {"A": {"slippage_bps": 5.0}}}]
        stats = slippage_stats(entries)
        assert stats["count"] == 1
        assert stats["excluded_lifecycle_orders"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
