import math

from bot.__main__ import _vol_overlay


def test_calm_market_stays_fully_invested():
    rets = [0.001] * 60
    out = _vol_overlay(rets, target=0.25, window=20, fee=0.0015)
    # zero realized variance -> weight 1, no ongoing cost
    for i in range(1, len(out)):
        assert out[i] == 0.001


def test_high_vol_market_scales_down():
    rets = [0.05, -0.05] * 60
    out = _vol_overlay(rets, target=0.25, window=20, fee=0.0)
    # annualized vol ~ 0.05*sqrt(365) ~ 95% -> weight ~0.26
    scaled = [abs(r) for r in out[25:]]
    assert all(s <= 0.05 * 0.3 for s in scaled)


def test_no_lookahead_future_changes_do_not_affect_past():
    rets = [0.01 * ((-1) ** i) for i in range(80)]
    a = _vol_overlay(rets, target=0.2, window=20, fee=0.0015)
    rets2 = list(rets)
    rets2[-1] = 5.0  # huge final-day shock
    b = _vol_overlay(rets2, target=0.2, window=20, fee=0.0015)
    assert a[:-1] == b[:-1]


def test_weight_capped_at_one():
    # tiny volatility below target -> weight would exceed 1 without the cap
    rets = [0.0001, -0.0001] * 60
    out = _vol_overlay(rets, target=0.25, window=20, fee=0.0)
    assert all(abs(r) <= 0.0001 + 1e-12 for r in out)


def test_warmup_period_unscaled():
    rets = [0.05, -0.05] * 8  # only 16 days: no scaling during warmup
    out = _vol_overlay(rets, target=0.01, window=20, fee=0.0)
    assert out[0] == 0.05
    assert out[1] == -0.05
