"""Tests for ablation studies and regime-conditioned performance."""
import random

import pytest

from bot.ablation import family_ablation, family_of, overlay_ablation
from bot.regimes import regime_conditioned_performance
from bot.strategy import BuyHold, SmaCrossover, TrendVol, build_candidates


def _candles(n=1400, seed=11):
    rng = random.Random(seed)
    closes = []
    price = 100.0
    drift = 0.0
    for i in range(n):
        if i % 250 == 0:
            drift = rng.choice([0.0015, -0.0008, 0.0002])
        price *= max(0.2, 1 + drift + rng.gauss(0, 0.02))
        closes.append(price)
    return [
        {"open_time": 86_400_000 * i, "open": c, "high": c, "low": c, "close": c}
        for i, c in enumerate(closes)
    ]


def test_family_of_labels():
    assert family_of(TrendVol(50, 20, 0.3)) == "TrendVol"
    assert family_of(SmaCrossover(10, 50)) == "SmaCross"
    assert family_of(BuyHold()) == "BuyHold"


def test_family_ablation_runs_and_reports_baseline():
    candles = _candles()
    from bot.walkforward import absolute_folds

    folds = absolute_folds(candles, train_days=365, test_days=180)
    pool = [BuyHold(), TrendVol(25, 20, 0.4), SmaCrossover(10, 50)]
    rows = family_ablation(candles, folds, pool, fee=0.0)
    assert rows[0]["omitted_family"].startswith("(none")
    assert rows[0]["sharpe_delta"] == pytest.approx(0.0)
    omitted = {r["omitted_family"] for r in rows[1:]}
    assert omitted == {"TrendVol", "SmaCross", "BuyHold"}
    assert all("n_remaining" in r and r["n_remaining"] == 2 for r in rows[1:])


def test_family_ablation_sorted_most_valuable_first():
    candles = _candles()
    from bot.walkforward import absolute_folds

    folds = absolute_folds(candles, train_days=365, test_days=180)
    rows = family_ablation(candles, folds, build_candidates()[:12], fee=0.001)
    deltas = [r["sharpe_delta"] for r in rows[1:]]  # omit the pinned baseline row
    assert deltas == sorted(deltas)


def test_overlay_ablation_variants_present():
    timeline = list(range(86_400_000, 86_400_000 * 120, 86_400_000))
    rng = random.Random(3)
    dailies = {
        f"s{k}": {t: rng.gauss(0.0005, 0.02) for t in timeline} for k in range(4)
    }
    out = overlay_ablation(dailies, timeline, n_assets=4)
    names = set(out)
    assert "equal_weight" in names
    assert "inv_vol" in names
    assert "inv_vol+tilt" in names
    assert "inv_vol+tilt+crisis|vol_targeted" in names
    for m in out.values():
        assert m["final"] > 0


def test_overlay_vol_target_reduces_drawdown_on_volatile_stream():
    rng = random.Random(9)
    timeline = list(range(86_400_000, 86_400_000 * 200, 86_400_000))
    # one violent asset: big up/down cycles
    dailies = {"boom": {t: (0.03 if (i // 10) % 2 else -0.04) + rng.gauss(0, 0.002) for i, t in enumerate(timeline)}}
    out = overlay_ablation(dailies, timeline, n_assets=1, target_vol=0.15)
    raw = out["inv_vol"]["max_drawdown"]
    targeted = out["inv_vol|vol_targeted"]["max_drawdown"]
    assert targeted > raw  # less negative with the risk overlay


def test_regime_conditioned_performance_buckets_all_days():
    rng = random.Random(4)
    timeline = list(range(86_400_000 * 50, 86_400_000 * 150, 86_400_000))
    labels = {t: ("bull" if i % 2 else "bear") for i, t in enumerate(timeline)}
    rets_by_day = {
        t: (0.01 if labels[t] == "bull" else -0.005) + rng.gauss(0, 0.002) for t in timeline
    }
    out = regime_conditioned_performance(rets_by_day, timeline, labels)
    assert set(out) == {"bear", "bull"}
    assert out["bull"]["hit_rate"] == pytest.approx(1.0)
    assert out["bull"]["sharpe"] > out["bear"]["sharpe"]
    assert sum(v["days"] for v in out.values()) == len(timeline)
