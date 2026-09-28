"""Tests for research-methodology extensions."""
import json
import math
import os
import random
import subprocess
import sys

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
from bot.stats_validation import stationary_bootstrap_indices


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

    def test_pvalue_resolution_is_bounded_by_resample_count(self):
        edge = [0.01 + (0.0001 if i % 2 else -0.0001) for i in range(200)]
        res = spa_test([edge], n_boot=9, seed=3)
        assert res["p_value"] >= 0.1

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

    def test_seed_is_reproducible_across_python_hash_seeds(self):
        code = (
            "import json; "
            "from bot.research import expanded_bootstrap; "
            "returns = [((i * 37) % 101 - 50) / 10000 for i in range(120)]; "
            "print(json.dumps(expanded_bootstrap(returns, n_boot=20, seed=17), sort_keys=True))"
        )
        outputs = []
        for hash_seed in ("1", "987654"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs.append(json.loads(result.stdout))
        assert outputs[0] == outputs[1]


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


class TestMcFuturePaths:
    """The forward Monte Carlo must resample the ENTIRE history."""

    def test_late_history_can_affect_generated_paths(self):
        # 600 observations, 252-day horizon. Under the old (buggy) sampler only
        # indices [0, 252) were reachable, so moving the SAME values into the
        # second half of the series changed nothing. It must change now.
        early_big = [0.01] * 300 + [-0.01] * 300
        late_big = [-0.01] * 300 + [0.01] * 300
        a = mc_future_paths(early_big, horizon_days=252, n_paths=300, block=5, seed=1)
        b = mc_future_paths(late_big, horizon_days=252, n_paths=300, block=5, seed=1)
        assert a["terminal_median"] != pytest.approx(b["terminal_median"], rel=1e-12)

    def test_sampling_domain_covers_the_whole_sample(self):
        # Direct check on the index sampler: with a domain far larger than the
        # number of draws, indices must reach the tail of the sample.
        rng = random.Random(4)
        seen = set()
        for _ in range(400):
            seen.update(stationary_bootstrap_indices(30, 5, rng, domain=1500))
        assert max(seen) > 251, "sampler never reached observations past index 251"
        assert min(seen) < 251

    def test_domain_defaults_to_length_for_backwards_compatibility(self):
        rng = random.Random(4)
        assert all(0 <= i < 50 for i in stationary_bootstrap_indices(50, 5, rng))

    def test_reports_its_provenance(self):
        r = mc_future_paths([0.01, -0.01] * 300, horizon_days=60, n_paths=100, block=4, seed=2)
        assert r["n_source_observations"] == 600
        assert r["n_paths"] == 100 and r["block"] == 4

    def test_deterministic_under_a_fixed_seed(self):
        rets = [(i % 7) / 100.0 for i in range(200)]
        kw = dict(horizon_days=40, n_paths=80, block=6, seed=99)
        assert mc_future_paths(rets, **kw) == mc_future_paths(rets, **kw)
        assert mc_future_paths(rets, **kw) != mc_future_paths(rets, **{**kw, "seed": 100})

    def test_horizon_longer_than_the_sample_is_allowed(self):
        # Blocks wrap, which is what makes this a forecast, not an error.
        r = mc_future_paths([0.01, -0.01, 0.02] * 10, horizon_days=200, n_paths=50, block=3, seed=5)
        assert 0.0 <= r["p_loss"] <= 1.0
        assert r["horizon_days"] == 200

    def test_single_observation_sample_works(self):
        assert mc_future_paths([0.01], horizon_days=10, n_paths=20, block=2, seed=1)["terminal_median"] > 0.0

    def test_empty_returns_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            mc_future_paths([], horizon_days=10, n_paths=10)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_returns_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            mc_future_paths([0.01, bad], horizon_days=10, n_paths=10)

    @pytest.mark.parametrize("horizon", [0, -1, 2.5, True, None])
    def test_invalid_horizon_rejected(self, horizon):
        with pytest.raises(ValueError, match="horizon_days"):
            mc_future_paths([0.01] * 50, horizon_days=horizon, n_paths=10)

    @pytest.mark.parametrize("n_paths", [0, -5, 1.5, True])
    def test_invalid_path_count_rejected(self, n_paths):
        with pytest.raises(ValueError, match="n_paths"):
            mc_future_paths([0.01] * 50, n_paths=n_paths)

    @pytest.mark.parametrize("block", [0, -3, 2.5, True])
    def test_invalid_block_rejected(self, block):
        with pytest.raises(ValueError, match="block"):
            mc_future_paths([0.01] * 50, n_paths=10, block=block)

    def test_all_probabilities_are_in_range(self):
        r = mc_future_paths([(i % 5 - 2) / 100.0 for i in range(300)],
                            horizon_days=100, n_paths=200, block=5, seed=8)
        assert 0.0 <= r["p_loss"] <= 1.0
        assert r["terminal_p05"] <= r["terminal_median"] <= r["terminal_p95"]


class TestAnalyticUnderperformanceTail:
    """The Normal approximation must use the correct tail."""

    def test_large_positive_gap_gives_low_probability(self):
        bench = _seeded_iid()
        res = probability_of_underperformance([b + 0.004 for b in bench], bench, n_boot=200)
        assert res["observed_sharpe_gap"] > 0
        assert res["analytic_p_underperform_sharpe"] < 0.05

    def test_large_negative_gap_gives_high_probability(self):
        bench = _seeded_iid()
        res = probability_of_underperformance([b - 0.004 for b in bench], bench, n_boot=200)
        assert res["observed_sharpe_gap"] < 0
        assert res["analytic_p_underperform_sharpe"] > 0.95

    def test_zero_gap_is_about_one_half(self):
        bench = _seeded_iid()
        res = probability_of_underperformance(list(bench), list(bench), n_boot=200)
        assert res["analytic_p_underperform_sharpe"] == pytest.approx(0.5, abs=0.05)

    def test_analytic_probability_is_monotone_in_the_gap(self):
        bench = _seeded_iid()
        # Deltas are listed from a big strategy edge DOWN to a big deficit, so
        # the probability of underperformance must INCREASE along the list.
        ps = [
            probability_of_underperformance([b + d for b in bench], bench, n_boot=100)["analytic_p_underperform_sharpe"]
            for d in (0.004, 0.002, 0.0, -0.002, -0.004)
        ]
        assert ps == sorted(ps)
        assert all(0.0 <= p <= 1.0 for p in ps)

    def test_the_three_quantities_are_distinct_and_labelled(self):
        # Empirical probability, CI, and analytic approximation answer
        # different questions and must not be conflated.
        bench = _seeded_iid()
        res = probability_of_underperformance([b + 0.001 for b in bench], bench, n_boot=200)
        lo, hi = res["sharpe_gap_ci"]
        assert lo <= hi                      # a CI is ordered, a probability is not
        assert 0.0 <= res["analytic_p_underperform_sharpe"] <= 1.0
        assert 0.0 <= res["p_underperform_sharpe"] <= 1.0
        # the CI is a 90% interval, so it is wide; a probability is not
        assert hi - lo > 0.0

    def test_misaligned_series_rejected(self):
        with pytest.raises(ValueError, match="aligned"):
            probability_of_underperformance([0.01] * 50, [0.01] * 49)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_series_rejected(self, bad):
        good = _seeded_iid(50)
        with pytest.raises(ValueError, match="finite"):
            probability_of_underperformance([bad] + good[1:], good, n_boot=20)

    def test_invalid_bootstrap_counts_rejected(self):
        bench = _seeded_iid(60)
        with pytest.raises(ValueError, match="n_boot"):
            probability_of_underperformance(list(bench), list(bench), n_boot=0)
        with pytest.raises(ValueError, match="block"):
            probability_of_underperformance(list(bench), list(bench), n_boot=20, block=0)

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="30"):
            probability_of_underperformance([0.01] * 10, [0.005] * 10)


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


class TestMcFuturePathDistributions:
    def test_percentiles_ordered_and_probabilities_sane(self):
        res = mc_future_paths(_seeded_iid(mu=0.0008), horizon_days=200, n_paths=800)
        assert 0.0 <= res["p_loss"] <= 1.0
        assert res["terminal_p05"] <= res["terminal_median"] <= res["terminal_p95"]
        assert res["path_mdd_worst"] <= res["path_mdd_median"] <= 0.0

    def test_positive_drift_lowers_loss_prob(self):
        bad = mc_future_paths(_seeded_iid(mu=-0.001, seed=1), horizon_days=200, n_paths=500)
        good = mc_future_paths(_seeded_iid(mu=0.001, seed=1), horizon_days=200, n_paths=500)
        assert good["p_loss"] < bad["p_loss"]
