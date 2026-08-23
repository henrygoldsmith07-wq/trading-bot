"""Tests for research-methodology extensions."""
import math
import random

import pytest

from bot.metrics import max_drawdown
from bot.research import (
    bayesian_sharpe,
    circular_block_bootstrap_indices,
    drawdown_confidence_intervals,
    expanded_bootstrap,
    mc_future_paths,
    moving_block_bootstrap_indices,
    portfolio_dsr,
    probability_of_underperformance,
    sequence_risk,
    spa_test,
)


def _seeded_iid(n=600, mu=0.0005, sd=0.02, seed=7):
    rng = random.Random(seed)
    return [rng.gauss(mu, sd) for _ in range(n)]


class TestBootstrapSamplers:
    def test_lengths_and_ranges(self):
        for sampler in (circular_block_bootstrap_indices, moving_block_bootstrap_indices):
            idx = sampler(50, block=10, rng=random.Random(1))
            assert len(idx) == 50
            assert all(0 <= i < 50 for i in idx)

    def test_circular_preserves_local_order(self):
        idx = circular_block_bootstrap_indices(100, block=20, rng=random.Random(3))
        # at least one adjacent pair must be consecutive (block structure)
        assert any(idx[i + 1] == (idx[i] + 1) % 100 for i in range(len(idx) - 1))

    def test_deterministic_given_seed(self):
        a = circular_block_bootstrap_indices(40, 8, random.Random(9))
        b = circular_block_bootstrap_indices(40, 8, random.Random(9))
        assert a == b


class TestSpaTest:
    def test_clear_edge_rejected_null(self):
        edge = [r + 0.002 for r in _seeded_iid()]  # one strong candidate
        noise = [_seeded_iid(seed=s) for s in range(1, 6)]
        res = spa_test([edge] + noise, n_boot=150)
        assert res["p_value"] <= 0.10
        assert res["best_stat"] > 0

    def test_all_noise_no_rejection(self):
        rows = [_seeded_iid(seed=s) for s in range(4)]
        res = spa_test(rows, n_boot=100)
        assert res["p_value"] > 0.05

    def test_degenerate_streams_report_p1(self):
        res = spa_test([[0.0] * 100], n_boot=25)
        assert res["p_value"] == 1.0


class TestExpandedBootstrap:
    def test_three_schemes_with_ordered_cis(self):
        res = expanded_bootstrap(_seeded_iid(), n_boot=200)
        assert set(res) == {"stationary", "circular", "moving"}
        for scheme in res.values():
            lo, hi = scheme["cagr_ci"]
            assert lo <= hi
            lo, hi = scheme["sharpe_ci"]
            assert lo <= hi
            lo, hi = scheme["mdd_ci"]
            assert lo <= hi


class TestDrawdownCIs:
    def test_actual_inside_band_structure(self):
        rets = _seeded_iid()
        eq = [1.0]
        for r in rets:
            eq.append(eq[-1] * (1 + r))
        res = drawdown_confidence_intervals(rets, n_boot=300)
        assert res["actual_mdd"] == pytest.approx(max_drawdown(eq))
        lo, hi = res["mdd_90_ci"]
        assert hi <= 0.0 and lo <= hi  # drawdowns are non-positive; worst is smallest
        assert res["mdd_worst"] <= res["mdd_median"] <= 0.0
        assert 0.0 <= res["time_under_water_median"] <= 1.0


class TestProbabilityOfUnderperformance:
    def test_dominant_strategy_never_underperforms(self):
        bench = _seeded_iid()
        strat = [b + 0.001 for b in bench]
        res = probability_of_underperformance(strat, bench, n_boot=300)
        assert res["p_underperform_cagr"] == pytest.approx(0.0, abs=0.01)

    def test_dominated_strategy_always_underperforms(self):
        bench = _seeded_iid()
        strat = [b - 0.001 for b in bench]
        res = probability_of_underperformance(strat, bench, n_boot=300)
        assert res["p_underperform_cagr"] >= 0.99

    def test_misaligned_raises(self):
        with pytest.raises(ValueError, match="aligned"):
            probability_of_underperformance([0.01] * 50, [0.01] * 49)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="30"):
            probability_of_underperformance([0.01] * 10, [0.005] * 10)

    def test_portfolio_dsr_passes_through(self):
        from bot.stats_validation import dsr

        rets = _seeded_iid(mu=0.001)
        assert portfolio_dsr(rets, [1.0], 2) == dsr(rets, [1.0], 2)


class TestBayesianSharpe:
    def test_positive_series_has_high_posterior(self):
        res = bayesian_sharpe(_seeded_iid(mu=0.002, sd=0.01), draws=4000)
        lo, hi = res["ci_90"]
        assert lo < hi
        assert res["prob_above_benchmark"] > 0.95
        assert res["posterior_mean"] > 0

    def test_zero_mean_series_is_centered_on_zero(self):
        res = bayesian_sharpe(_seeded_iid(mu=0.0, sd=0.02, n=900), draws=4000)
        assert abs(res["median"]) < 1.0
        assert res["prob_above_benchmark"] < 0.99

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            bayesian_sharpe([0.01, 0.02])


class TestSequenceRisk:
    def test_trending_series_low_loss_probability(self):
        rng = random.Random(5)
        rets = [0.001 + 0.01 * rng.gauss(0, 1) for _ in range(800)]
        res = sequence_risk(rets, horizon_days=126, n_shuffles=30)
        assert res["observed_median"] > 0.05  # positive drift shows up
        assert res["n_windows"] > 5

    def test_gap_between_observed_and_shuffle_is_finite(self):
        rets = _seeded_iid(n=700)
        res = sequence_risk(rets, horizon_days=180, n_shuffles=20)
        assert math.isfinite(res["sequence_risk_gap"])
        assert abs(res["sequence_risk_gap"]) < 1.0

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            sequence_risk([0.01] * 10, horizon_days=100)


class TestMcFuturePaths:
    def test_percentiles_ordered_and_probabilities_sane(self):
        res = mc_future_paths(_seeded_iid(mu=0.0008), horizon_days=200, n_paths=800)
        assert 0.0 <= res["p_loss"] <= 1.0
        assert res["terminal_p05"] <= res["terminal_median"] <= res["terminal_p95"]
        assert res["path_mdd_worst"] <= res["path_mdd_median"] <= 0.0

    def test_positive_drift_lowers_loss_prob(self):
        bad = mc_future_paths(_seeded_iid(mu=-0.001, seed=1), horizon_days=200, n_paths=500)
        good = mc_future_paths(_seeded_iid(mu=0.001, seed=1), horizon_days=200, n_paths=500)
        assert good["p_loss"] < bad["p_loss"]
