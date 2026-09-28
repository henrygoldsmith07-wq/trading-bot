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

from api.summary import build_forward_summary
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
)
from bot.prospective import load_log, slippage_stats
from bot.verdict import build_verdict, grade_forward


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

    def test_no_order_in_the_committed_tape_is_trustworthy(self):
        assert audit_log_lifecycle(load_log("forward_log.jsonl"))["n_orders_trusted"] == 0

    def test_mixed_tape_counts_only_the_trustworthy_orders(self):
        entries = [{
            "date": "2026-09-07",
            "orders": [_order(symbol="GOOD"),
                       _order(symbol="BAD", intent_ts="2029-01-01T00:00:00+00:00")],
        }]
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


class TestSupersededExperimentIsNotProspectiveEvidence:
    """A superseded tape must not be graded as current prospective evidence.

    This is the gap that let the dashboard advertise 11 forward days of a v1
    experiment whose accounting model the repo had already corrected.
    """

    def _forward(self, **over):
        base = {
            "available": True, "started": True, "n_days_recorded": 11,
            "code_verified": True, "evidence_verified": True, "parameter_changes": 0,
            "data_outages": 0, "methodologically_current": False,
            "experiment_version": "v1", "superseded_by": "exact-wealth-v2",
        }
        base.update(over)
        return base

    def _verdict(self, forward):
        return build_verdict(
            canonical_rule_stats=[{"name": "+ tilt + crisis, banded 5% rebalance",
                                   "dsr": 0.99, "psr": 0.99, "cagr": 0.1, "max_drawdown": -0.1}],
            canonical_per_asset=[{"sharpe": 1.0}],
            canonical_n_folds=6, pool_size=85, ledger_search_n=34,
            cost_report=None, forward=forward,
        )

    def test_superseded_days_are_not_counted(self):
        f = self._verdict(self._forward())["details"]["forward"]
        assert f["grade"] == "Insufficient"
        assert f["inputs"]["methodologically_current"] is False
        assert f["inputs"]["days_recorded_not_counted"] == 11

    def test_superseded_reason_names_the_experiment_and_successor(self):
        reason = self._verdict(self._forward())["details"]["forward"]["reason"]
        assert "v1" in reason and "exact-wealth-v2" in reason and "SUPERSEDED" in reason

    def test_superseded_label_does_not_advertise_uncounted_days(self):
        # The label is the dashboard hero: it must not read "11 trading days"
        # directly above a reason saying those days were not counted.
        v = self._verdict(self._forward())
        assert v["verdict"]["prospective_forward_evidence"] == "Insufficient — 0 trading days"
        assert "11" not in v["verdict"]["prospective_forward_evidence"]

    def test_superseded_is_not_compromised(self):
        # Nothing is tampered or false, so this must NOT print INVALIDATED --
        # that would be its own kind of dishonesty about a merely
        # un-revalidated system.
        assert self._verdict(self._forward())["verdict"]["overall"] != "invalidated"

    def test_missing_currency_field_fails_closed(self):
        fwd = self._forward()
        del fwd["methodologically_current"]  # a producer bug must not award evidence
        assert self._verdict(fwd)["details"]["forward"]["grade"] == "Insufficient"

    def test_current_experiment_still_counts_its_days(self):
        f = self._verdict(self._forward(methodologically_current=True,
                                        experiment_version="v2",
                                        superseded_by=None))["details"]["forward"]
        assert f["grade"] == "Insufficient"  # 11 days is still < 30
        assert "days_recorded_not_counted" not in f["inputs"]
        assert f["inputs"]["days_recorded"] == 11

    def test_grade_forward_defaults_to_current_for_direct_callers(self):
        assert grade_forward(days_recorded=200)["grade"] == "Strong"

    def test_broken_seal_outranks_supersession(self):
        g = grade_forward(days_recorded=200, code_verified=False, methodologically_current=False)
        assert g["grade"] == "COMPROMISED"


class TestSupersessionReachesTheDashboard:
    def test_payload_declares_the_experiment(self):
        f = build_forward_summary()
        for key in ("experiment_version", "accounting_model",
                    "methodologically_current", "superseded_by"):
            assert key in f, f"dashboard payload must declare {key}"

    def test_committed_freeze_is_reported_as_superseded(self):
        f = build_forward_summary()
        assert f["experiment_version"] == "v1"
        assert f["accounting_model"] == "additive-legs-v1"
        assert f["methodologically_current"] is False
        assert f["superseded_by"] == "exact-wealth-v2"


class TestTurnoverBasisAgreesWithCosts:
    """`stats["turnover"]` must reconcile with the fees actually charged."""

    def _run(self, opens, target, closes=None):
        from bot.engine import run_strategy

        day, base = 86_400_000, 1_600_000_000_000
        n = len(closes) if closes is not None else len(opens)
        px = closes or [100.0] * n
        candles = [{"open_time": base + i * day, "open": opens[i], "close": px[i],
                    "high": max(opens[i], px[i]), "low": min(opens[i], px[i])}
                   for i in range(n)]
        return run_strategy(candles, lambda c, i: target, fee=0.002,
                            execution="next_open", start_index=1)

    def test_turnover_reconciles_with_bar_costs(self):
        res = self._run([100.0, 105.0, 95.0, 110.0, 90.0, 100.0, 100.0], 0.5)
        assert res["turnover"] == pytest.approx(sum(res["bar_costs"]) / 0.002, abs=1e-12)

    def test_turnover_is_not_the_undrifted_weight_difference(self):
        res = self._run([100.0, 105.0, 95.0, 110.0, 90.0, 100.0, 100.0], 0.5)
        undrifted = sum(abs(res["weights"][i] - (res["weights"][i - 1] if i else 0.0))
                        for i in range(len(res["weights"])))
        assert res["turnover"] > undrifted

    def test_no_gap_turnover_is_unchanged(self):
        # open == previous close on every bar, so nothing drifts and the
        # traded notional is exactly the undrifted weight difference.
        px = [100.0, 100.0, 100.0, 105.0, 105.0, 110.0]
        opens = [px[0]] + px[:-1]  # every open equals the prior close
        res = self._run(opens, 1.0, closes=px)
        undrifted = sum(abs(res["weights"][i] - (res["weights"][i - 1] if i else 0.0))
                        for i in range(len(res["weights"])))
        assert res["turnover"] == pytest.approx(undrifted, abs=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
