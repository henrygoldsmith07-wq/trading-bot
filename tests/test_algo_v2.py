import pytest

from bot.strategy import DualMomentum, TSMom, build_candidates, risk_ensemble, strategy_from_spec, strategy_to_spec
from bot.walkforward import combine_portfolio_invvol


def _candles(closes, start_ms=1_600_000_000_000):
    return [{"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)]


def _rising(n=400, growth=1.004):
    return _candles([100.0 * growth ** i for i in range(n)])


def test_tsmom_long_in_uptrend_flat_in_downtrend():
    up = TSMom(63, 20, 0.35).weight(_rising())
    down_closes = [c["close"] for c in reversed(_rising())]
    down = TSMom(63, 20, 0.35).weight(_candles(down_closes))
    assert up > 0.0
    assert down == 0.0


def test_tsmom_short_history_zero():
    assert TSMom(63, 20, 0.35).weight(_candles([1.0, 2.0])) == 0.0


def test_tsmom_vol_target_caps_weight():
    calm = _candles([100.0 * 1.001 ** i for i in range(400)])
    wild_closes = []
    px = 100.0
    for i in range(400):
        px *= 1.05 if i % 2 == 0 else 0.9524
        wild_closes.append(px)
    wild = _candles(wild_closes)
    # wild series is roughly flat over 63d but noisy: momentum sign varies; just check bounds
    w1 = TSMom(63, 20, 0.35).weight(calm)
    w2 = TSMom(63, 20, 0.35).weight(wild)
    assert 0.0 <= w2 <= 1.0
    assert w1 == 1.0  # calm uptrend: vol target not binding


def test_dual_momentum_fractional_weight():
    # flat, then a big run-up, then a mild pullback: the 63d horizon is
    # negative while 126d and 252d are positive -> fractional base
    closes = [100.0] * 100
    for _ in range(100):
        closes.append(closes[-1] * 1.01)
    for _ in range(100):
        closes.append(closes[-1] * 0.999)
    w = DualMomentum((63, 126, 252), 20, 0.5).weight(_candles(closes))
    assert 0.0 < w < 1.0


def test_dual_momentum_all_horizons_up():
    w = DualMomentum((63, 126), 20, 0.5).weight(_rising())
    assert w > 0.5  # both horizons positive -> base 1.0, scaled by vol


def test_risk_ensemble_spec_roundtrip():
    ens = risk_ensemble()
    spec = strategy_to_spec(ens)
    assert spec["type"] == "Ensemble"
    rebuilt = strategy_from_spec(spec)
    assert repr(rebuilt) == repr(ens)


def test_pool_expanded():
    pool = build_candidates()
    assert len(pool) >= 80
    kinds = {type(c).__name__ for c in pool}
    assert {"TSMom", "DualMomentum", "Ensemble", "TrendVol"} <= kinds


def _closes_up(n=400, growth=1.004):
    return [100.0 * growth ** i for i in range(n)]


@pytest.mark.parametrize("series", ["up", "down", "choppy"])
def test_new_strategies_bounded(series):
    closes = {
        "up": _closes_up(),
        "down": list(reversed(_closes_up())),
        "choppy": [100.0 * (1.02 if i % 3 else 0.985) ** i for i in range(400)],
    }[series]
    data = _candles(closes)
    for s in [TSMom(63, 20, 0.35), DualMomentum((63, 126, 252), 20, 0.3), risk_ensemble()]:
        w = s.weight(data)
        assert 0.0 <= w <= 1.0


# ---------- inverse-vol portfolio ----------

def _flat_timeline(n, start=1_600_000_000_000):
    return [start + i * 86_400_000 for i in range(n)]


def test_invvol_weights_low_vol_asset_more():
    n = 120
    timeline = _flat_timeline(n)
    calm = {t: 0.001 * ((-1) ** (i % 3)) for i, t in enumerate(timeline)}  # tiny vol
    wild = {t: 0.02 * ((-1) ** i) for i, t in enumerate(timeline)}  # 20x vol
    out = combine_portfolio_invvol({"CALM": calm, "WILD": wild}, timeline, n_assets=2)
    equal_share = 0.5 * calm[timeline[-1]] + 0.5 * wild[timeline[-1]]
    # inverse-vol tilts toward CALM: portfolio return closer to calm's than equal-weight would be
    assert abs(out[-1] - calm[timeline[-1]]) < abs(equal_share - calm[timeline[-1]])


def test_invvol_cap_prevents_domination():
    n = 120
    timeline = _flat_timeline(n)
    # CALM: tiny alternating returns (minuscule vol). Six wilds in cancelling
    # +/- pairs: identical vol, daily mean exactly zero. With the cap, CALM's
    # weight is 2/7 and the portfolio tracks 2/7 * calm exactly.
    calm = {t: 0.00001 * (1.0 if i % 2 == 0 else -1.0) for i, t in enumerate(timeline)}
    wilds = {}
    for k in range(3):
        wilds[f"W{k}a"] = {t: 0.01 * (1.0 if (i + k) % 2 == 0 else -1.0) for i, t in enumerate(timeline)}
        wilds[f"W{k}b"] = {t: 0.01 * (-1.0 if (i + k) % 2 == 0 else 1.0) for i, t in enumerate(timeline)}
    dailies = {"CALM": calm, **wilds}
    out = combine_portfolio_invvol(dailies, timeline, n_assets=7, max_multiple_of_equal=2.0)
    expected = 2.0 / 7.0 * calm[timeline[-1]]
    assert out[-1] == pytest.approx(expected, rel=0.02)


def test_invvol_warmup_is_equal_weight():
    timeline = _flat_timeline(5)
    a = {t: 0.01 for t in timeline}
    b = {t: -0.01 for t in timeline}
    out = combine_portfolio_invvol({"A": a, "B": b}, timeline, n_assets=2)
    assert all(r == pytest.approx(0.0) for r in out)  # equal weights cancel


def test_invvol_missing_asset_sits_in_cash():
    timeline = _flat_timeline(60)
    a = {t: 0.01 for t in timeline[:30]}
    b = {t: 0.01 for t in timeline}
    out = combine_portfolio_invvol({"A": a, "B": b}, timeline, n_assets=2)
    # first 30 days: exposure 2/2 -> 1% ; last 30 days: 1/2 exposure -> 0.5%
    assert out[10] == pytest.approx(0.01)
    assert out[40] == pytest.approx(0.005)
