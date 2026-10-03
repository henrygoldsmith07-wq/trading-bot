"""Walk-forward consistency diagnostics: is the edge consistent or just lucky?

Invariants pinned here:
* every fold is reported — worst folds are never dropped;
* a dominant single asset / fold / regime is flagged, not smoothed away;
* empty inputs degrade honestly instead of dividing by zero;
* the same input always produces the same report.
"""
from __future__ import annotations

import pytest

from bot.walkforward_diagnostics import (
    analyse_asset_dispersion,
    analyse_concentration,
    analyse_folds,
    consistency_report,
)

FOLDS = [
    {"label": "fold-1", "start": "2020-08-16", "end": "2021-08-16",
     "daily_returns": [0.001] * 10, "strategy": "TrendVol", "train_sharpe": 1.0},
    {"label": "fold-2", "start": "2021-08-16", "end": "2022-08-16",
     "daily_returns": [-0.001] * 10, "strategy": "SmaCrossover", "train_sharpe": 0.5},
    {"label": "fold-3", "start": "2022-08-16", "end": "2023-08-16",
     "daily_returns": [0.002] * 10, "strategy": "TrendVol", "train_sharpe": 0.8},
]


class TestFolds:
    def test_every_fold_is_reported(self):
        out = analyse_folds(FOLDS)
        assert out["n_folds"] == 3
        assert [f["label"] for f in out["folds"]] == ["fold-1", "fold-2", "fold-3"]

    def test_worst_fold_is_the_losings_one(self):
        out = analyse_folds(FOLDS)
        assert out["worst_fold"]["label"] == "fold-2"
        assert out["worst_fold"]["return"] < 0

    def test_profitable_share_counts_losers(self):
        out = analyse_folds(FOLDS)
        assert out["pct_profitable_folds"] == pytest.approx(2 / 3)

    def test_fold_return_compounds_daily(self):
        out = analyse_folds([{"label": "f", "start": "a", "end": "b", "daily_returns": [0.1, 0.1]}])
        assert out["folds"][0]["return"] == pytest.approx(1.21 - 1.0)

    def test_empty_input_is_honest_not_a_crash(self):
        out = analyse_folds([])
        assert out["n_folds"] == 0
        assert out["worst_fold"] is None
        assert any("no fold" in n for n in out["notes"])

    def test_deterministic(self):
        assert analyse_folds(FOLDS) == analyse_folds(FOLDS)

    def test_single_fold_is_not_consistency_evidence(self):
        out = analyse_folds(FOLDS[:1])
        assert any("single fold" in n for n in out["notes"])


class TestDispersion:
    def test_share_positive(self):
        out = analyse_asset_dispersion([
            {"symbol": "A", "cagr": 0.1, "sharpe": 1.0},
            {"symbol": "B", "cagr": -0.1, "sharpe": -1.0},
            {"symbol": "C", "cagr": 0.2, "sharpe": 0.5},
        ])
        assert out["assets"] == 3
        assert out["positive_sharpe_assets"] == 2
        assert out["pct_positive_sharpe"] == pytest.approx(2 / 3)

    def test_empty_is_zero_not_divide_by_zero(self):
        out = analyse_asset_dispersion([])
        assert out["assets"] == 0
        assert out["pct_positive_sharpe"] == 0.0


class TestConcentration:
    def test_single_asset_dominance_is_flagged(self):
        out = analyse_concentration(asset_contributions={"A": 0.9, "B": 0.05, "C": 0.05})
        assert out["largest_asset_contribution"]["name"] == "A"
        assert any("concentrated" in w for w in out["warnings"])

    def test_boundary_at_the_flag_share(self):
        # The flag is on the LARGEST name's share of positive contributions,
        # and it fires only above the threshold. Three names are needed to put
        # the largest share genuinely below 0.5 (two names summing to 1.0 make
        # the larger one >= 0.5 by construction).
        just_below = {"A": 0.50, "B": 0.25, "C": 0.25}
        above = {"A": 0.51, "B": 0.25, "C": 0.24}
        assert analyse_concentration(asset_contributions=just_below)["warnings"] == []
        assert analyse_concentration(asset_contributions=above)["warnings"]

    def test_negative_contributions_do_not_cancel_positives(self):
        out = analyse_concentration(asset_contributions={"A": 0.5, "B": -0.5})
        # share is over POSITIVE contributions only
        assert out["largest_asset_contribution"]["share_of_positive"] == pytest.approx(1.0)

    def test_all_negative_is_warned_not_divided(self):
        out = analyse_concentration(asset_contributions={"A": -0.5, "B": -0.2})
        assert out["largest_asset_contribution"]["share_of_positive"] == 0.0
        assert any("no positive" in w for w in out["warnings"])

    def test_nothing_supplied_is_all_none(self):
        out = analyse_concentration()
        assert out["largest_asset_contribution"] is None
        assert out["herfindahl_asset"] is None


class TestConsistencyReport:
    def _report(self, *, n_folds=4, pct=0.8, assets=5, fold_share=0.0, asset_share=0.0, regime_share=0.0):
        folds = analyse_folds([
            {"label": f"f{i}", "start": "a", "end": "b", "daily_returns": [0.001] * 5}
            for i in range(n_folds)
        ]) if n_folds else analyse_folds([])
        disp = analyse_asset_dispersion([{"symbol": f"S{i}", "cagr": 0.1, "sharpe": 0.5} for i in range(assets)])
        if assets:
            disp["pct_positive_sharpe"] = pct
        conc = analyse_concentration(
            asset_contributions={"A": asset_share, "B": 1 - asset_share} if asset_share else None,
            fold_contributions={"fold-1": fold_share, "fold-2": 1 - fold_share} if fold_share else None,
            regime_contributions={"crisis": regime_share, "calm": 1 - regime_share} if regime_share else None,
        )
        return consistency_report(folds, disp, conc)

    def test_spread_by_default(self):
        assert self._report()["dominant_source"] == "spread_across_assets_and_folds"

    def test_one_fold_dominates(self):
        out = self._report(fold_share=0.9)
        assert out["dominant_source"] == "one_fold"
        assert out["flags"]["one_fold"]

    def test_one_asset_dominates(self):
        out = self._report(asset_share=0.9)
        assert out["dominant_source"] == "one_asset"
        assert out["flags"]["one_asset"]

    def test_one_regime_dominates(self):
        out = self._report(regime_share=0.9)
        assert out["dominant_source"] == "one_regime"

    def test_fold_outranks_asset(self):
        out = self._report(fold_share=0.9, asset_share=0.9)
        assert out["dominant_source"] == "one_fold"

    def test_minority_positive_is_flagged(self):
        out = self._report(pct=0.3)
        assert out["flags"]["minority_of_assets_positive"]

    def test_insufficient_data_wins(self):
        out = self._report(n_folds=0)
        assert out["dominant_source"] == "insufficient_data"
        assert out["summary"]
