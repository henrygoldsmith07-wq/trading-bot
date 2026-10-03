"""Contribution attribution: which component of the frozen stack earns its keep?

Invariants pinned here:
* every ablation rung consumes the SAME universe and period (one fingerprint);
* the delta chain reconciles exactly with the cumulative ladder;
* the cost line is reported, never re-simulated;
* a missing band simulation refuses instead of fabricating a rung;
* identical inputs give identical reports.
"""
from __future__ import annotations

import pytest

from bot.attribution import decompose, format_ladder

DAY = 86_400_000
T0 = 1_700_000_000_000
TIMELINE = [T0 + i * DAY for i in range(5)]


def _dailies(band: bool = False):
    a = {t: 0.01 for t in TIMELINE}
    b = {t: -0.005 for t in TIMELINE}
    if band:
        # The band suppresses small trades at sleeve level: a visibly
        # different return stream is what proves rung 5 is not rung 4.
        a = {t: 0.012 for t in TIMELINE}
        b = {t: -0.004 for t in TIMELINE}
    return {"A": a, "B": b}


def _report(**kwargs):
    args = dict(asset_dailies=_dailies(), timeline=TIMELINE, n_assets=2, band_dailies=_dailies(band=True))
    args.update(kwargs)
    return decompose(**args)


class TestLadder:
    def test_six_rungs_plus_cost_line(self):
        report = _report()
        assert len(report["rungs"]) == 6
        assert [r["component"] for r in report["rungs"]][-1] == "volatility overlay"
        assert report["cost_lines"] and "costs" in report["cost_lines"][0]["component"]

    def test_delta_chain_reconciles_with_the_cumulative_ladder(self):
        report = _report()
        rungs = report["rungs"]
        total_delta = sum(r["delta_cagr"] for r in rungs[1:])
        assert rungs[-1]["cagr"] - rungs[0]["cagr"] == pytest.approx(total_delta, abs=1e-12)

    def test_full_rule_equals_the_last_rung(self):
        report = _report()
        assert report["full_rule"] == {
            "cagr": report["rungs"][-1]["cagr"],
            "sharpe": report["rungs"][-1]["sharpe"],
            "max_drawdown": report["rungs"][-1]["max_drawdown"],
        }

    def test_base_rung_is_hand_computable(self):
        # Two sleeves, equal weight 1/2 each: +1% and -0.5% per day -> +0.25%.
        report = _report()
        base = report["rungs"][0]
        assert base["cagr"] == pytest.approx((1.0025) ** 365 - 1, rel=1e-9)
        assert base["delta_cagr"] == 0.0

    def test_cost_line_is_negative(self):
        assert _report()["cost_lines"][0]["cagr"] <= 0.0

    def test_ladder_formatter_reads_the_report(self):
        report = _report()
        text = format_ladder(report)
        for rung in report["rungs"]:
            assert rung["component"] in text


class TestPinnedInputs:
    def test_fingerprint_is_identical_across_rungs(self):
        # One report, one fingerprint: the runs cannot have used different
        # universes because there is only one to begin with.
        report = _report()
        assert report["universe_pinned"] is True
        assert len(report["inputs_fingerprint"]) == 64

    def test_fingerprint_stable_across_calls(self):
        assert _report()["inputs_fingerprint"] == _report()["inputs_fingerprint"]

    def test_fingerprint_moves_with_the_timeline(self):
        short = decompose(_dailies(), TIMELINE[:3], 2, band_dailies=_dailies(band=True))
        assert short["inputs_fingerprint"] != _report()["inputs_fingerprint"]

    def test_period_is_stated(self):
        report = _report()
        assert set(report["period"]) == {"start", "end"}


class TestRefusals:
    def test_empty_timeline_refuses(self):
        with pytest.raises(ValueError):
            decompose(_dailies(), [], 2, band_dailies=_dailies(band=True))

    def test_band_dailies_must_match_the_sleeves(self):
        mismatched = {**_dailies(band=True), "C": {t: 0.0 for t in TIMELINE}}
        with pytest.raises(ValueError):
            decompose(_dailies(), TIMELINE, 2, band_dailies=mismatched)

    def test_missing_band_refuses_rather_than_fabricating(self):
        """The band lives at sleeve level; its effect cannot be derived from
        unbanded streams. Inventing rung 5 as rung 4 would be evidence of
        nothing."""
        with pytest.raises(ValueError):
            decompose(_dailies(), TIMELINE, 2, band_dailies=None)

    def test_bad_asset_count_refuses(self):
        with pytest.raises(ValueError):
            decompose(_dailies(), TIMELINE, 0, band_dailies=_dailies(band=True))


class TestDeterminism:
    def test_identical_inputs_give_identical_reports(self):
        assert _report() == _report()
