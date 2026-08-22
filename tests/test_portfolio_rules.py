import math

import pytest

from bot.portfolio_rules import avg_pairwise_corr, combine_portfolio_rule, tilt_multipliers


def _flat_timeline(n, start=1_600_000_000_000):
    return [start + i * 86_400_000 for i in range(n)]


def _warm(n, r):
    return [r] * n


# ---------- tilt multipliers ----------

def test_tilt_orders_by_rank_and_preserves_gross():
    ranks = {"A": 0.30, "B": 0.10, "C": -0.05, "D": None}
    m = tilt_multipliers(ranks, max_tilt=0.5)
    assert m["D"] == 1.0  # no history: neutral
    assert m["A"] > m["B"] > m["C"]
    assert m["A"] <= 1.5 + 1e-9
    assert m["C"] >= 0.5 - 1e-9
    assert sum(m[s] for s in "ABC") == pytest.approx(3.0, rel=1e-9)


def test_tilt_single_ranked_asset_is_neutral():
    assert tilt_multipliers({"A": 0.2, "B": None}) == {"A": 1.0, "B": 1.0}


def test_tilt_zero_band_is_flat():
    m = tilt_multipliers({"A": 0.3, "B": 0.0, "C": -0.1}, max_tilt=0.0)
    assert set(m.values()) == {1.0}


# ---------- correlation ----------

def test_avg_pairwise_corr_detects_co_movement():
    n = 80
    a = [0.01 * ((-1) ** i) for i in range(n)]
    b = a  # perfectly correlated
    c = [x * -1 for x in a]  # perfectly negatively correlated with a
    corr = avg_pairwise_corr({"A": a, "B": b, "C": c}, 60)
    assert corr == pytest.approx((1.0 + -1.0 + -1.0) / 3.0, abs=1e-9)


def test_avg_pairwise_corr_insufficient_history():
    assert avg_pairwise_corr({"A": [0.1, 0.2], "B": [0.1, 0.2]}, 60) is None
    assert avg_pairwise_corr({"A": [0.1] * 5}, 3) is None


# ---------- combined rule ----------

def test_rule_beats_top_asset_when_momentum_persists():
    # STAR trends up, DOG trends down; tilt should load STAR over DOG
    n = 200
    timeline = _flat_timeline(n)
    star = {}
    dog = {}
    for i, t in enumerate(timeline):
        star[t] = 0.002 if i % 2 == 0 else 0.001
        dog[t] = -0.0015 if i % 2 == 0 else -0.001
    tilted = combine_portfolio_rule({"STAR": star, "DOG": dog}, timeline, 2, use_crisis=False)
    untilted = combine_portfolio_rule({"STAR": star, "DOG": dog}, timeline, 2, use_tilt=False, use_crisis=False)
    assert sum(tilted) > sum(untilted)


def test_rule_crisis_derisk_reduces_exposure_when_correlated():
    n = 200
    timeline = _flat_timeline(n)
    # two assets moving identically (corr = 1 > threshold)
    a = {t: 0.01 * ((-1) ** i) for i, t in enumerate(timeline)}
    b = {t: 0.01 * ((-1) ** i) for i, t in enumerate(timeline)}
    derisked = combine_portfolio_rule({"A": a, "B": b}, timeline, 2, use_tilt=False, use_crisis=True, corr_window=60, corr_threshold=0.6, derisk=0.5)
    full = combine_portfolio_rule({"A": a, "B": b}, timeline, 2, use_tilt=False, use_crisis=False)
    assert abs(derisked[-1]) < abs(full[-1])


def test_rule_survivorship_exposure_shrinks():
    n = 120
    timeline = _flat_timeline(n)
    a = {t: 0.01 for t in timeline[:60]}
    b = {t: 0.01 for t in timeline}
    out = combine_portfolio_rule({"A": a, "B": b}, timeline, 2, use_tilt=False, use_crisis=False)
    assert out[30] == pytest.approx(0.01, rel=1e-6)
    assert out[90] == pytest.approx(0.005, rel=1e-6)


def test_rule_no_lookahead():
    n = 150
    timeline = _flat_timeline(n)
    base = {"A": {t: 0.001 for t in timeline}, "B": {t: -0.001 for t in timeline}}
    shocked = {"A": dict(base["A"]), "B": dict(base["B"])}
    shocked["A"][timeline[-1]] = 5.0  # huge final-day shock
    out1 = combine_portfolio_rule(base, timeline, 2, use_crisis=False)
    out2 = combine_portfolio_rule(shocked, timeline, 2, use_crisis=False)
    assert out1[:-1] == out2[:-1]


def test_rule_warmup_equals_inverse_vol_off():
    # with <vol_window history, weights are equal: A and B cancel
    n = 10
    timeline = _flat_timeline(n)
    a = {t: 0.01 for t in timeline}
    b = {t: -0.01 for t in timeline}
    out = combine_portfolio_rule({"A": a, "B": b}, timeline, 2, use_tilt=False, use_crisis=False)
    assert all(r == pytest.approx(0.0) for r in out)
