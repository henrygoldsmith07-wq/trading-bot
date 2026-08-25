"""Verdict engine: deterministic evidence grading, auditable at every tier."""
import pytest

from bot.verdict import (
    build_verdict,
    combine,
    format_verdict,
    grade_costs,
    grade_forward,
    grade_historical,
    grade_robustness,
    grade_selection_bias,
)

RULES = [
    {"name": "inv-vol (selected underlying)", "cagr": 0.108, "sharpe": 0.70,
     "max_drawdown": -0.122, "psr": 0.997, "dsr": 0.997},
    {"name": "+ tilt + crisis, banded 5% rebalance", "cagr": 0.100, "sharpe": 0.75,
     "max_drawdown": -0.097, "psr": 0.997, "dsr": 0.997},
]
PER_ASSET = [{"symbol": f"S{i}", "sharpe": 0.6} for i in range(11)] + [
    {"symbol": "S11", "sharpe": -0.2}, {"symbol": "S12", "sharpe": 0.1}
]


class TestHistorical:
    def test_strong_capped_by_high_selection_risk(self):
        res = grade_historical(RULES, "banded 5% rebalance", selection_risk_grade="High")
        # PSR/DSR look excellent at trial-count=1, but the search correction
        # is unresolved -> capped to Moderate
        assert res["grade"] == "Moderate"
        assert "capped" in res["reason"]

    def test_uncapped_when_selection_risk_lower(self):
        res = grade_historical(RULES, "banded", selection_risk_grade="Moderate")
        assert res["grade"] == "Strong"

    def test_weak_dsr_stays_weak_even_without_cap(self):
        weak = [{**r, "psr": 0.85, "dsr": 0.82} for r in RULES]
        assert grade_historical(weak, "banded", "Low")["grade"] == "Weak"

    def test_headline_missing_is_insufficient(self):
        assert grade_historical(RULES, "nonexistent", "Low")["grade"] == "Insufficient"


class TestRobustness:
    def test_share_positive_drives_grade(self):
        strong = [{"sharpe": 1.0}] * 10 + [{"sharpe": -0.1}]
        assert grade_robustness(strong, n_folds=6)["grade"] == "Strong"
        half = [{"sharpe": 1.0}, {"sharpe": -1.0}]
        assert grade_robustness(half, n_folds=6)["grade"] == "Weak"

    def test_few_folds_cap_strong(self):
        strong = [{"sharpe": 1.0} for _ in range(12)]
        assert grade_robustness(strong, n_folds=2)["grade"] == "Moderate"

    def test_empty_record_insufficient(self):
        assert grade_robustness([], n_folds=None)["grade"] == "Insufficient"


class TestSelectionBias:
    def test_wide_search_is_high(self):
        res = grade_selection_bias(pool_size=85, ledger_search_n=29)
        assert res["grade"] == "High"

    def test_ledger_informed_dsr_clearing_bar_downgrades(self):
        res = grade_selection_bias(pool_size=85, ledger_search_n=29, ledger_informed_dsr=0.96)
        assert res["grade"] == "Moderate"

    def test_single_predeclared_is_low(self):
        assert grade_selection_bias(pool_size=1, ledger_search_n=None)["grade"] == "Low"


class TestCosts:
    def test_insufficient_sample(self):
        assert grade_costs(5, None, sufficient=False)["grade"] == "Insufficient"
        assert "30" in grade_costs(5, None, sufficient=False)["reason"]

    def test_small_error_strong(self):
        assert grade_costs(50, mean_error_bp=1.9, sufficient=True)["grade"] == "Strong"

    def test_large_error_weak(self):
        assert grade_costs(50, mean_error_bp=-20.4, sufficient=True)["grade"] == "Weak"


class TestForward:
    def test_one_day_insufficient_with_label(self):
        f = grade_forward(days_recorded=1, code_verified=True, parameter_changes=0)
        assert f["grade"] == "Insufficient"

    @pytest.mark.parametrize("days,expected", [
        (24, "Insufficient"), (60, "Weak"), (120, "Moderate"), (200, "Strong"),
    ])
    def test_thresholds(self, days, expected):
        assert grade_forward(days, True, 0)["grade"] == expected

    def test_compromised_overrides(self):
        for kw in ({"code_verified": False}, {"parameter_changes": 3}):
            res = grade_forward(days_recorded=400, **kw)
            assert res["grade"] == "COMPROMISED"

    def test_outage_ratio_caps(self):
        res = grade_forward(days_recorded=100, code_verified=True, outage_days=25)
        assert res["grade"] in ("Weak", "Moderate")
        assert "20%" in res["reason"]


class TestCombine:
    def test_compromised_invalidates_everything(self):
        overall, _ = combine("Strong", "Strong", "Strong", "Strong", "COMPROMISED")
        assert overall == "INVALIDATED"

    def test_short_forward_promising_not_validated(self):
        overall, _ = combine("Moderate", "Moderate", "High", "Insufficient", "Insufficient")
        assert overall == "promising, not validated"

    def test_weak_history_not_established(self):
        overall, _ = combine("Weak", "Weak", "High", "Insufficient", "Insufficient")
        assert overall == "not established"

    def test_full_validation_requires_all_moderate_plus_strong_forward(self):
        overall, _ = combine("Strong", "Strong", "Moderate", "Moderate", "Strong")
        assert overall == "validated (provisional)"

    def test_partial_between_states(self):
        overall, _ = combine("Strong", "Moderate", "Moderate", "Moderate", "Moderate")
        assert overall == "partially supported"


class TestBuildVerdict:
    def test_realistic_shape(self):
        v = build_verdict(
            canonical_rule_stats=RULES,
            canonical_per_asset=PER_ASSET,
            canonical_n_folds=6,
            pool_size=85,
            ledger_search_n=29,
            cost_report={"n_turnover_events": 11, "error_bp": -20.41, "sufficient": False},
            forward={"available": True, "started": True, "n_days_recorded": 24,
                     "code_verified": True, "parameter_changes": 0, "data_outages": 3},
        )
        vd = v["verdict"]
        assert vd["historical_evidence"] == "Moderate"      # capped
        assert vd["selection_bias_risk"] == "High"
        assert vd["cost_robustness"] == "Insufficient"
        assert "24 trading days" in vd["prospective_forward_evidence"]
        assert vd["overall"] == "promising, not validated"
        text = format_verdict(v)
        assert "STRATEGY VERDICT" in text
        assert "OVERALL: promising, not validated" in text

    def test_compromised_forward_invalidates(self):
        v = build_verdict(
            canonical_rule_stats=RULES, canonical_per_asset=PER_ASSET,
            canonical_n_folds=6, pool_size=1, ledger_search_n=None,
            cost_report={"n_turnover_events": 60, "error_bp": 1.0, "sufficient": True},
            forward={"available": True, "started": True, "n_days_recorded": 200,
                     "code_verified": False, "parameter_changes": 0, "data_outages": 0},
        )
        assert v["verdict"]["overall"] == "INVALIDATED"
