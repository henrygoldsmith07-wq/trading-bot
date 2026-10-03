"""Execution quality: modelled cost, observed gap and data error stay separate.

Invariants pinned here:
* the three cost categories are never merged into one "slippage" figure;
* predicted cost is never guessed — no cost model means None, not zero;
* turnover is measured on weights and hand-checkable;
* null slippage observations go to "unknown", never to a fake bucket.
"""
from __future__ import annotations

import pytest

from bot.execution_quality import TURNOVER_EVENT_EPSILON, diagnostics


def _row(date: str, weights: dict[str, float], slippage: dict[str, float | None] | None = None, **extra):
    assets = {}
    for sym, w in weights.items():
        block: dict = {"weight": w}
        if slippage is not None and sym in slippage:
            block["slippage_bps"] = slippage[sym]
        assets[sym] = block
    return {"date": date, "assets": assets, "outages": [], "missed_fills": [], **extra}


class TestSeparation:
    def test_three_categories_never_merge(self):
        out = diagnostics([_row("2026-09-01", {"A": 1.0})], execution_model="exact-wealth-v2")
        sep = out["separation"]
        assert set(sep) == {"modelled_cost", "observed_gap", "data_quality_error"}
        assert "predicted_cost_bps" in sep["modelled_cost"]
        assert "mean_abs_bps" in sep["observed_gap"]
        assert set(sep["data_quality_error"]) == {
            "outage_days", "skipped_fills", "stale_data_events", "lifecycle_validation_failures"
        }

    def test_predicted_cost_is_never_guessed(self):
        out = diagnostics([_row("2026-09-01", {"A": 1.0})], execution_model="m")
        assert out["predicted_cost_bps"] is None
        assert out["separation"]["modelled_cost"]["source"] == "no cost model supplied"

    def test_predicted_cost_passes_through_verbatim(self):
        out = diagnostics([_row("2026-09-01", {"A": 1.0})],
                          execution_model="m", cost_model={"predicted_cost_bps": 12.5})
        assert out["predicted_cost_bps"] == 12.5
        assert out["separation"]["modelled_cost"]["predicted_cost_bps"] == 12.5


class TestTurnover:
    def test_two_day_book_is_hand_computable(self):
        # Day 1: A=1.0 (from zero) -> 0.5 one-way. Day 2: A=0.6, B=0.4 -> |0.4| + |0.4| / 2 = 0.4.
        rows = [_row("2026-09-01", {"A": 1.0}), _row("2026-09-02", {"A": 0.6, "B": 0.4})]
        out = diagnostics(rows, execution_model="m")
        assert out["turnover"] == pytest.approx(0.5 + 0.4)
        assert out["turnover_per_day"] == pytest.approx(0.45)

    def test_no_turnover_when_weights_still(self):
        rows = [_row("2026-09-01", {"A": 1.0}), _row("2026-09-02", {"A": 1.0})]
        out = diagnostics(rows, execution_model="m")
        assert out["turnover"] == pytest.approx(0.5)  # only the initial entry
        assert out["turnover_events"] == 1

    def test_event_threshold_is_respected(self):
        assert TURNOVER_EVENT_EPSILON > 0
        rows = [_row("2026-09-01", {"A": 1.0}), _row("2026-09-02", {"A": 1.0 - TURNOVER_EVENT_EPSILON / 2})]
        out = diagnostics(rows, execution_model="m")
        assert out["turnover_events"] == 1  # drift below the epsilon is not a trade

    def test_empty_entries_are_zero_not_a_crash(self):
        out = diagnostics([], execution_model="m")
        assert out["turnover"] == 0.0
        assert out["turnover_events"] == 0


class TestObservedGap:
    def test_null_slippage_is_unknown_not_a_bucket(self):
        out = diagnostics([_row("2026-09-01", {"A": 1.0}, slippage={"A": None})], execution_model="m")
        buckets = {b["bucket"]: b["count"] for b in out["slippage_distribution"]}
        assert buckets["unknown"] == 1
        assert buckets["0-1bps"] == 0
        assert out["observed_execution_gap"]["observations"] == 0

    def test_bucket_boundaries(self):
        rows = [_row("2026-09-01", {"A": 1.0},
                     slippage={"A": 0.5})]
        out = diagnostics(rows, execution_model="m")
        buckets = {b["bucket"]: b["count"] for b in out["slippage_distribution"]}
        assert buckets["0-1bps"] == 1

        for value, expected in ((1.0, "1-5bps"), (4.9, "1-5bps"), (5.0, "5-10bps"),
                                (9.9, "5-10bps"), (10.0, "10-25bps"), (25.0, "25bps+"), (90.0, "25bps+")):
            out = diagnostics([_row("2026-09-01", {"A": 1.0}, slippage={"A": value})], execution_model="m")
            buckets = {b["bucket"]: b["count"] for b in out["slippage_distribution"]}
            assert buckets[expected] == 1, f"{value} should land in {expected}"

    def test_gap_stats_use_magnitudes(self):
        rows = [_row("2026-09-01", {"A": 1.0}, slippage={"A": -12.0}),
                _row("2026-09-02", {"A": 1.0}, slippage={"A": 2.0})]
        out = diagnostics(rows, execution_model="m")
        gap = out["observed_execution_gap"]
        assert gap["observations"] == 2
        assert gap["mean_abs_bps"] == pytest.approx(7.0)


class TestQualityErrors:
    def test_skipped_fills_and_outages_counted(self):
        row = _row("2026-09-01", {"A": 1.0},
                   outages=[{"symbol": "A"}], missed_fills=[{"s": 1}, {"s": 2}])
        out = diagnostics([row], execution_model="m")
        assert out["skipped_fills"] == 2
        assert out["separation"]["data_quality_error"]["outage_days"] == 1

    def test_lifecycle_failures_counted(self):
        row = _row("2026-09-01", {"A": 1.0},
                   orders=[{"symbol": "A", "side": "BUY",
                            "signal_generated_ts": "2026-09-02T00:00:00+00:00",
                            "intent_ts": "2026-09-01T00:00:00+00:00",
                            "submitted_ts": "2026-09-01T00:00:01+00:00",
                            "fill_ts": "2026-09-01T00:00:02+00:00"}])
        out = diagnostics([row], execution_model="m")
        assert out["lifecycle_validation_failures"] >= 1

    def test_delayed_signals_are_not_guessed(self):
        row = _row("2026-09-01", {"A": 1.0}, orders=[{"symbol": "A", "side": "BUY"}])
        out = diagnostics([row], execution_model="m")
        assert out["delayed_signals"] == 0
        assert any("timestamps" in n for n in out["notes"])


class TestDeterminism:
    def test_identical_inputs_give_identical_reports(self):
        rows = [_row("2026-09-01", {"A": 1.0}, slippage={"A": 3.0}),
                _row("2026-09-02", {"A": 0.5, "B": 0.5}, slippage={"A": 1.0, "B": None})]
        assert diagnostics(rows, execution_model="m") == diagnostics(rows, execution_model="m")
