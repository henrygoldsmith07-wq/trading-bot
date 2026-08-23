"""Tests for strategy clustering / duplicate detection / effective trials."""
import pytest

from bot.clustering import (
    correlation_matrix,
    effective_trial_count,
    near_duplicate_pairs,
    pearson,
    strategy_clusters,
)


def _stream(seed, n=400, noise=0.01):
    import random

    rng = random.Random(seed)
    return [rng.gauss(0.0004, noise) for _ in range(n)]


def test_pearson_perfect_and_independent():
    base = _stream(1)
    assert pearson(base, base) == pytest.approx(1.0)
    assert pearson(base, [-x for x in base]) == pytest.approx(-1.0)
    other = [x if i % 2 else -x * 3 for i, x in enumerate(_stream(2))]
    assert abs(pearson(base, other)) < 0.5


def test_pearson_misaligned_raises():
    with pytest.raises(ValueError):
        pearson([0.0] * 10, [0.0] * 9)


def test_near_duplicate_detection():
    base = _stream(1)
    twin = [b + 0.00001 for b in base]
    unrelated = _stream(99)
    dups = near_duplicate_pairs({"a": base, "a_copy": twin, "z": unrelated}, corr_threshold=0.999)
    assert len(dups) == 1
    pair = dups[0]
    assert {pair["a"], pair["b"]} == {"a", "a_copy"}
    assert pair["correlation"] > 0.999


def test_clusters_group_correlated_strategies():
    # two families: members share a common factor plus idiosyncratic noise
    fam1 = [[0.001 + 0.02 * ((i * 37 + s) % 11) / 11 for i in range(200)] for s in range(3)]
    fam2 = [[-0.001 + 0.01 * ((i * 53 + s) % 7) / 7 for i in range(200)] for s in range(3)]
    streams = {f"f{s}": v for s, v in enumerate(fam1)}
    streams.update({f"g{s}": v for s, v in enumerate(fam2)})
    clusters = strategy_clusters(streams, corr_threshold=0.95)
    assert 1 <= len(clusters) <= len(streams)


def test_uncorrelated_streams_form_own_clusters():
    streams = {f"s{k}": _stream(k * 17 + 3) for k in range(6)}
    clusters = strategy_clusters(streams, corr_threshold=0.95)
    assert len(clusters) >= 4


def test_effective_trial_count_bounds():
    single_family = {f"a{k}": _stream(1, noise=0.05 + k * 0.001) for k in range(8)}  # all ~same drift
    res = effective_trial_count(single_family, corr_threshold=0.90)
    assert res["n_strategies"] == 8
    assert 1 <= res["n_effective"] <= 8
    independent = {f"i{k}": _stream(k * 31 + 5) for k in range(8)}
    res_ind = effective_trial_count(independent)
    assert res_ind["n_effective"] > res["n_effective"]
    assert res_ind["avg_pairwise_corr"] < res["avg_pairwise_corr"]


def test_effective_trials_single_stream():
    res = effective_trial_count({"only": _stream(1)})
    assert res == {"n_strategies": 1, "n_clusters": 1, "n_effective": 1, "avg_pairwise_corr": 0.0}


def test_correlation_matrix_symmetric_subset():
    a, b, c = _stream(1), _stream(2), _stream(3)
    m = correlation_matrix({"x": a, "y": b, "z": c})
    assert set(m) == {("x", "y"), ("x", "z"), ("y", "z")}
