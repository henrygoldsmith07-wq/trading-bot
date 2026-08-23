"""Tests for extended transaction-cost models."""
import math

import pytest

from bot.costs import (
    CostParams,
    median_daily_quote_volume,
    passes_liquidity_filter,
    realized_vol_series,
    square_root_impact_fraction,
    tiered_taker_fee,
    vol_cost_multiplier,
)
from bot.engine import run_strategy
from bot.strategy import BuyHold, TrendVol


def _candles(closes, volumes=None):
    vols = volumes or [1000.0] * len(closes)
    return [
        {"open_time": 86_400_000 * i, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": v}
        for i, (c, v) in enumerate(zip(closes, vols, strict=True))
    ]


def test_realized_vol_series_matches_definition():
    closes = [100.0, 110.0, 99.0, 105.0, 120.0, 90.0]
    rv = realized_vol_series(closes, window=3, periods_per_year=365)
    # index 3 (0-based): uses log returns of days 1..3
    rets = [math.log(closes[j] / closes[j - 1]) for j in range(1, 4)]
    m = sum(rets) / 3
    var = sum((r - m) ** 2 for r in rets) / 2
    assert rv[3] == pytest.approx(math.sqrt(var * 365))
    for i in range(3):
        assert math.isnan(rv[i])


def test_vol_multiplier_clamped_and_monotone():
    assert vol_cost_multiplier(0.60) == pytest.approx(1.0)
    assert vol_cost_multiplier(float("nan")) == 1.0
    low = vol_cost_multiplier(0.10, floor_mult=0.5)
    high = vol_cost_multiplier(6.0, cap_mult=3.0)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(3.0)
    mid1 = vol_cost_multiplier(0.30)
    mid2 = vol_cost_multiplier(0.60)
    mid3 = vol_cost_multiplier(1.20)
    assert mid1 < mid2 < mid3


def test_impact_scales_with_sqrt_participation():
    base = square_root_impact_fraction(0.1, daily_vol=0.03, adv_to_equity=100.0, impact_k=1.0)
    four_x = square_root_impact_fraction(0.4, daily_vol=0.03, adv_to_equity=100.0, impact_k=1.0)
    assert base > 0
    assert four_x == pytest.approx(base * 2.0)  # 4x turnover -> sqrt(4)=2x impact
    assert square_root_impact_fraction(0.0, 0.03, 100.0) == 0.0
    assert square_root_impact_fraction(0.1, 0.0, 100.0) == 0.0
    assert square_root_impact_fraction(0.1, 0.03, 0.0) == 0.0


def test_tiered_fee_picks_highest_reached_tier():
    tiers = [(0.0, 0.001), (100.0, 0.0008), (1000.0, 0.0005)]
    assert tiered_taker_fee(50.0, tiers) == 0.001
    assert tiered_taker_fee(500.0, tiers) == 0.0008
    assert tiered_taker_fee(9999.0, tiers) == 0.0005


def test_median_quote_volume_and_filter():
    candles = _candles([100.0] * 40, volumes=[10_000.0] * 40)
    med = median_daily_quote_volume(candles, lookback=30)
    assert med == pytest.approx(1_000_000.0)
    ok, why = passes_liquidity_filter(candles, min_median_quote_volume=500_000.0)
    assert ok and "1,000,000" in why
    ok, why = passes_liquidity_filter(candles, min_median_quote_volume=5_000_000.0)
    assert not ok and "<" in why


def test_no_volume_data_passes_vacuously():
    candles = [{"open_time": i, "close": 5.0} for i in range(10)]
    ok, why = passes_liquidity_filter(candles, min_median_quote_volume=10.0)
    assert ok and "no volume" in why


def _trending(n=400):
    return _candles([100.0 * (1.003 if i % 7 else 0.998) ** i for i in range(n)])


def test_default_cost_params_leave_results_unchanged():
    data = _trending()
    kw = dict(fee=0.001, spread_bps=5.0, slippage_bps=5.0)
    a = run_strategy(data, TrendVol(50, 20, 0.3).weight_at, **kw)
    b = run_strategy(data, TrendVol(50, 20, 0.3).weight_at, cost_params=None, **kw)
    c = run_strategy(data, TrendVol(50, 20, 0.3).weight_at, cost_params=CostParams(), **kw)
    assert a["final"] == b["final"] == c["final"]


def test_vol_dependent_costs_reduce_returns_when_volatile():
    data = _trending()
    strat = BuyHold().weight_at  # no turnover: use something that rebalances
    strat = TrendVol(50, 20, 0.3).weight_at
    flat = run_strategy(data, strat, fee=0.0, spread_bps=10.0, slippage_bps=0.0)
    scaled = run_strategy(
        data,
        strat,
        fee=0.0,
        spread_bps=10.0,
        slippage_bps=0.0,
        cost_params=CostParams(spread_vol_scale=1.0),
    )
    assert scaled["final"] < flat["final"]


def test_market_impact_charged_only_on_turnover():
    data = _trending()
    tv = TrendVol(50, 20, 0.3)
    with_impact = run_strategy(
        data,
        tv.weight_at,
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        cost_params=CostParams(impact_k=50.0, adv_to_equity=10.0),
    )
    without_impact = run_strategy(data, tv.weight_at, fee=0.0)
    assert with_impact["turnover"] > 0  # strategy actually trades
    assert with_impact["final"] < without_impact["final"]
    # buy-and-hold turns over exactly once (after the vol warmup); impact must bite there
    bh_with = run_strategy(
        data,
        BuyHold().weight_at,
        fee=0.0,
        start_index=25,
        cost_params=CostParams(impact_k=50.0, adv_to_equity=10.0),
    )
    bh_without = run_strategy(data, BuyHold().weight_at, fee=0.0, start_index=25)
    assert bh_with["final"] < bh_without["final"]


def test_cost_params_roundtrip_kwargs():
    cp = CostParams(spread_vol_scale=1.0, impact_k=2.0, adv_to_equity=50.0)
    assert cp.active()
    kwargs = cp.to_kwargs()
    assert set(kwargs) == set(CostParams.__slots__)
    rebuilt = CostParams(**kwargs)
    assert rebuilt.to_kwargs() == kwargs
