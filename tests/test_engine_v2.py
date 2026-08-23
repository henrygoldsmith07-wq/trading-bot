import pytest

from bot.engine import run_strategy
from bot.portfolio_rules import combine_portfolio_rule


def _candles(closes, start_ms=0):
    return [{"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)]


def test_rebalance_band_suppresses_small_changes():
    # target drifts 1%/day: with a 5% band, no trades fire for 4 days
    targets = iter([0.01, 0.02, 0.03, 0.04, 0.05, 0.10])

    def w(candles, i):
        return next(targets)

    candles = _candles([100.0] * 7)
    res = run_strategy(candles, w, fee=0.01, rebalance_band=0.05)
    # weights stay 0 until the target crosses the band (0.05... day 5 target 0.10 fires)
    assert res["weights"][0] == 0.0
    assert res["weights"][3] == 0.0
    assert res["weights"][5] == pytest.approx(0.10)
    assert res["turnover"] == pytest.approx(0.10)


def test_rebalance_band_off_by_default():
    calls = []

    def w(candles, i):
        calls.append(i)
        return 0.5

    res = run_strategy(_candles([100.0] * 5), w, fee=0.01)
    assert res["turnover"] == pytest.approx(0.5)  # single 0->0.5 trade


def test_banding_never_exceeds_target():
    candles = _candles([100.0] * 20)
    res = run_strategy(candles, lambda c, i: min(0.99, 0.1 * i), fee=0.01, rebalance_band=0.05)
    assert all(w <= 0.99 + 1e-9 for w in res["weights"])


def _flat_timeline(n, start=1_600_000_000_000):
    return [start + i * 86_400_000 for i in range(n)]


def test_dd_throttle_cuts_exposure_in_drawdown():
    # portfolio that ramps up then bleeds: throttle should reduce later losses
    n = 240
    timeline = _flat_timeline(n)
    a = {}
    for i, t in enumerate(timeline):
        a[t] = 0.004 if i < 60 else -0.004  # +27% run-up, then sustained bleed
    plain = combine_portfolio_rule({"A": a}, timeline, 1, use_tilt=False, use_crisis=False, use_dd_throttle=False)
    throttled = combine_portfolio_rule({"A": a}, timeline, 1, use_tilt=False, use_crisis=False, use_dd_throttle=True, dd_trigger=-0.10, dd_exit=-0.05, throttle=0.5)
    # during the sustained drawdown the throttled stream must lose less
    assert sum(throttled[80:]) > sum(plain[80:])


def test_dd_throttle_hysteresis_and_recovery():
    # after the drawdown recovers above -5%, full exposure returns
    n = 300
    timeline = _flat_timeline(n)
    a = {}
    for i, t in enumerate(timeline):
        a[t] = (-0.004 if 60 <= i < 100 else 0.006)
    throttled = combine_portfolio_rule({"A": a}, timeline, 1, use_tilt=False, use_crisis=False, use_dd_throttle=True, dd_trigger=-0.10, dd_exit=-0.05, throttle=0.5)
    plain = combine_portfolio_rule({"A": a}, timeline, 1, use_tilt=False, use_crisis=False)
    # by the end, equity has recovered: returns should match again (throttle off)
    assert throttled[-1] == pytest.approx(plain[-1])


def test_dd_throttle_uses_only_past_equity():
    n = 200
    timeline = _flat_timeline(n)
    base = {t: 0.001 for t in timeline}
    shocked = dict(base)
    shocked[timeline[-1]] = -0.9  # catastrophic final day
    t1 = combine_portfolio_rule({"A": dict(base)}, timeline, 1, use_tilt=False, use_crisis=False, use_dd_throttle=True)
    t2 = combine_portfolio_rule({"A": shocked}, timeline, 1, use_tilt=False, use_crisis=False, use_dd_throttle=True)
    assert t1[:-1] == t2[:-1]
