import math
import random

import pytest

from bot.metrics import calmar, downside_deviation, expected_shortfall, kurtosis, skewness, sortino, var_hist
from bot.stats_validation import (
    bootstrap_metrics,
    dsr,
    expected_max_sharpe_annual,
    parameter_stability,
    psr,
    reality_check,
    shuffle_test,
    stationary_bootstrap_indices,
    start_end_sensitivity,
)


def _rand(n, mu, sd, seed=7):
    rng = random.Random(seed)
    return [rng.gauss(mu, sd) for _ in range(n)]


# ---------- moments / tail metrics ----------

def test_skew_kurtosis_on_known_shapes():
    assert skewness([1.0, 2.0, 3.0, 3.0, 2.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    left_skew = [1.0, 1.1, 1.2, 1.3, -5.0]
    assert skewness(left_skew) < 0
    fat_tails = [0.0] * 20 + [10.0] + [-10.0]
    assert kurtosis(fat_tails) > 3.0
    assert kurtosis(_rand(500, 0, 1)) == pytest.approx(3.0, abs=0.5)


def test_var_and_es_ordering():
    rets = ([-0.10, -0.08, -0.05, -0.04] + [-0.01, 0.0, 0.01, 0.02, 0.03, 0.05] * 10)
    v95 = var_hist(rets, 0.95)
    es95 = expected_shortfall(rets, 0.95)
    assert v95 == -0.04  # 3rd worst of 64 samples: int(0.05*64)=3
    assert es95 <= v95  # ES is at least as pessimistic
    assert es95 == pytest.approx((-0.10 + -0.08 + -0.05) / 3)


def test_var_descriptive_only_known_quantile():
    rets = list(range(-10, 11))  # -10..10 as returns
    assert var_hist(rets, 0.95) == -9


def test_sortino_rewards_positive_skew():
    good = [0.01, 0.01, 0.01, -0.01]
    bad = [0.03, 0.03, 0.03, -0.05]
    assert sortino(bad, 365) < sortino(good, 365)


def test_downside_deviation_zero_when_no_losses():
    assert downside_deviation([0.01, 0.02], 365) == 0.0


def test_calmar_formula():
    assert calmar(0.30, -0.15) == pytest.approx(2.0)
    assert calmar(0.30, 0.0) == 0.0


# ---------- PSR / DSR ----------

def test_psr_high_for_consistent_edge():
    rets = _rand(2000, 0.002, 0.01)
    assert psr(rets) > 0.99


def test_psr_about_half_for_noise():
    rets = _rand(2000, 0.0, 0.01)
    assert 0.2 < psr(rets) < 0.8


def test_psr_low_for_bad_returns():
    rets = _rand(2000, -0.002, 0.01)
    assert psr(rets) < 0.01


def test_expected_max_sharpe_grows_with_trials_and_variance():
    small = expected_max_sharpe_annual([0.1, 0.3], 2)
    bigger_var = expected_max_sharpe_annual([0.0, 1.0], 2)
    more_trials = expected_max_sharpe_annual([0.1, 0.3], 100)
    assert bigger_var > small
    assert more_trials > small


def test_dsr_below_psr():
    rets = _rand(1500, 0.0012, 0.01)
    trials = [0.0, 0.5, 1.0, 1.5, 2.0] * 10
    assert dsr(rets, trials, 50) < psr(rets)


# ---------- bootstrap ----------

def test_stationary_bootstrap_shape_and_determinism():
    idx = stationary_bootstrap_indices(500, 20, random.Random(1))
    assert len(idx) == 500
    assert all(0 <= i < 500 for i in idx)
    idx2 = stationary_bootstrap_indices(500, 20, random.Random(1))
    assert idx == idx2


def test_bootstrap_ci_contains_point_estimate_ballpark():
    rets = _rand(800, 0.001, 0.02, seed=3)
    boot = bootstrap_metrics(rets, n_boot=200, seed=5)
    assert boot["cagr_ci"][0] < boot["cagr_ci"][1]
    assert boot["sharpe_ci"][0] < boot["sharpe_ci"][1]
    assert boot["mdd_p95"] <= boot["mdd_median"] <= 0.0  # both negative, p95 worse


# ---------- Reality Check ----------

def test_reality_check_superior_candidate_low_pvalue():
    rng = random.Random(11)
    n = 1000
    streams = [_rand(n, 0.0, 0.01, seed=100 + i) for i in range(20)]  # noise
    streams.append(_rand(n, 0.0015, 0.01, seed=99))  # one real edge
    rc = reality_check(streams, n_boot=50, seed=13)
    assert 0.0 <= rc["p_value"] <= 1.0
    assert rc["p_value"] < 0.2  # the edge stands out of the max-null


def test_reality_check_all_noise_high_pvalue():
    streams = [_rand(600, 0.0, 0.01, seed=200 + i) for i in range(15)]
    rc = reality_check(streams, n_boot=50, seed=17)
    assert rc["p_value"] > 0.05


# ---------- shuffle / start-end ----------

def test_shuffle_test_detects_sequencing_protection():
    # alternating wins/losses: actual path never suffers consecutive losses,
    # so its drawdown is shallower than most random reorderings
    seq = [0.01, -0.004] * 400
    res = shuffle_test(seq, n_boot=200, seed=5)
    assert res["dd_percentile"] > 0.9  # almost no shuffle is as shallow


def test_shuffle_test_detects_clustered_crash():
    # all losses together at the end: actual drawdown far deeper than shuffles
    seq = [0.005] * 700 + [-0.01] * 300
    res = shuffle_test(seq, n_boot=200, seed=5)
    assert res["dd_percentile"] < 0.1


def test_shuffle_test_flat_series_mid_percentile():
    rets = _rand(500, 0.0005, 0.01)
    res = shuffle_test(rets, n_boot=200, seed=9)
    assert 0.05 < res["dd_percentile"] < 0.95


def test_start_end_sensitivity_shape():
    rets = _rand(1000, 0.001, 0.02)
    rows = start_end_sensitivity(rets)
    assert len(rows) >= 4
    for row in rows:
        assert row["trim_start"] >= 0 and row["trim_end"] >= 0
        assert row["sharpe"] == row["sharpe"]  # not NaN


# ---------- parameter stability ----------

def test_parameter_stability_flat_vs_spiky_grid():
    flat = {(i, j): {"sharpe": 1.0} for i in range(5) for j in range(4)}
    spiky = {(i, j): {"sharpe": 1.0 if (i, j) == (2, 2) else 0.0} for i in range(5) for j in range(4)}
    fs = parameter_stability(flat)
    ss = parameter_stability(spiky)
    assert fs["mean_neighbor_delta"] == 0.0
    assert ss["mean_neighbor_delta"] > 0.0
    assert fs["share_above_half_max"] == 1.0
    assert ss["share_above_half_max"] < 0.2


# ---------- walk-forward statistical plumbing ----------

def test_walk_forward_records_trials_and_exposure():
    from bot.strategy import TrendVol
    from bot.walkforward import walk_forward

    closes = [100.0 * (1.002 ** max(0, i - 200)) if i > 200 else 100.0 for i in range(1100)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    wf = walk_forward(candles, candidates=[TrendVol(50, 20, 0.4), TrendVol(75, 20, 0.4)], train_days=365, test_days=180)
    # trial_sharpes: 2 candidates x each fold evaluated on selection slice
    assert len(wf["trial_sharpes"]) >= 2
    assert 0.0 <= wf["exposure"] <= 1.0
    assert wf["turnover"] >= 0.0


def test_embargo_changes_selection_window_not_oos_timeline():
    from bot.strategy import TrendVol
    from bot.walkforward import walk_forward

    closes = [100.0 * (1.002 ** max(0, i - 200)) if i > 200 else 100.0 for i in range(1100)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    wf_plain = walk_forward(candles, candidates=[TrendVol(50, 20, 0.4)], train_days=365, test_days=180)
    wf_emb = walk_forward(candles, candidates=[TrendVol(50, 20, 0.4)], train_days=365, test_days=180, embargo_days=30)
    # single candidate: OOS days identical (embargo only trims selection data)
    assert sorted(wf_plain["daily"]) == sorted(wf_emb["daily"])


def test_nested_selection_picks_something_and_runs():
    from bot.strategy import TrendVol
    from bot.walkforward import nested_selection_fn, walk_forward_at

    closes = [100.0 * (1.002 ** max(0, i - 200)) if i > 200 else 100.0 for i in range(1600)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    folds = [(candles[800]["open_time"], candles[1000]["open_time"])]
    sel = nested_selection_fn(inner_train_days=365, inner_test_days=182, purge_days=100, embargo_days=30)
    wf = walk_forward_at(candles, folds, candidates=[TrendVol(50, 20, 0.4), TrendVol(100, 20, 0.4)], selection_fn=sel)
    assert wf["n_folds"] == 1
    assert wf["cagr"] > 0.0


def test_purged_inner_folds_have_gaps():
    from bot.walkforward import _purged_inner_folds

    closes = [100.0] * 1500
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    folds = _purged_inner_folds(candles, train_days=365, test_days=180, purge_days=200)
    assert folds
    for _, train_end, test_start, _ in folds:
        gap_days = (test_start - train_end)
        assert gap_days >= 190  # >= purge_days of calendar rows dropped


def test_fixed_candidate_streams_all_candidates():
    from bot.strategy import TrendVol
    from bot.walkforward import fixed_candidate_streams

    closes = [100.0 * (1.002 ** max(0, i - 200)) if i > 200 else 100.0 for i in range(1200)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    folds = [(candles[700]["open_time"], candles[900]["open_time"])]
    cands = [TrendVol(50, 20, 0.4), TrendVol(100, 20, 0.4)]
    streams = fixed_candidate_streams(candles, folds, cands, fee=0.001)
    assert set(streams) == {repr(c) for c in cands}
    for s in streams.values():
        assert len(s) == 200
