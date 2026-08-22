from bot.engine import run_strategy


def _candles(closes, start_ms=0):
    return [
        {"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def test_full_weight_tracks_buy_and_hold_before_fees():
    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c, i: 1.0, fee=0.0)
    assert abs(res["final"] - 1.21) < 1e-9


def test_fees_reduce_equity():
    candles = _candles([100, 110, 121])
    no_fee = run_strategy(candles, lambda c, i: 1.0, fee=0.0)
    with_fee = run_strategy(candles, lambda c, i: 1.0, fee=0.01)
    assert with_fee["final"] < no_fee["final"]


def test_zero_weight_stays_flat():
    candles = _candles([100, 50, 200])
    res = run_strategy(candles, lambda c, i: 0.0, fee=0.01)
    assert abs(res["final"] - 1.0) < 1e-9
    assert res["max_drawdown"] == 0.0


def test_weights_clamped_to_unit_interval():
    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c, i: 5.0, fee=0.0)
    # clamped to 1.0 -> identical to buy and hold
    assert abs(res["final"] - 1.21) < 1e-9
    res_neg = run_strategy(candles, lambda c, i: -3.0, fee=0.01)
    assert abs(res_neg["final"] - 1.0) < 1e-9


def test_turnover_charged_on_weight_changes():
    import pytest

    candles = _candles([100, 100, 100, 100, 100])
    res = run_strategy(candles, lambda c, i: 1.0 if i % 2 else 0.0, fee=0.01)
    # weights flip 0->1->0->1->0 over 4 return days: 4 unit changes of exposure
    assert res["turnover"] == 4.0
    assert res["final"] == pytest.approx((1.0 - 0.01) ** 4)


def test_no_lookahead_weight_never_sees_current_day():
    candles = _candles([100, 110, 121, 133])
    seen = []

    def w(cs, i):
        seen.append(cs[i - 1]["close"])
        return 1.0

    run_strategy(candles, w, fee=0.0)
    # the last close the strategy ever saw is candle n-2, never the final one
    assert seen[-1] == 121


def test_return_days_aligned_with_returns():
    import pytest

    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c, i: 1.0, fee=0.0)
    assert [t for t, _ in res["return_days"]] == [c["open_time"] for c in candles[1:]]
    assert res["return_days"][0][1] == pytest.approx(0.1)


def test_exposure_and_stats_present():
    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c, i: 0.5, fee=0.001)
    assert 0.0 <= res["exposure"] <= 1.0
    assert res["vol"] >= 0.0
    assert res["sharpe"] >= 0.0


def test_insufficient_candles_raises():
    import pytest

    with pytest.raises(ValueError):
        run_strategy(_candles([1.0]), lambda c, i: 1.0)
