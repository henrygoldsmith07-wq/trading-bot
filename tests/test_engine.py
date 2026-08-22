from bot.engine import run_strategy


def _candles(closes, start_ms=0):
    return [
        {"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def test_full_weight_tracks_buy_and_hold_before_fees():
    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c: 1.0, fee=0.0)
    assert abs(res["final"] - 1.21) < 1e-9


def test_fees_reduce_equity():
    candles = _candles([100, 110, 121])
    no_fee = run_strategy(candles, lambda c: 1.0, fee=0.0)
    with_fee = run_strategy(candles, lambda c: 1.0, fee=0.01)
    assert with_fee["final"] < no_fee["final"]


def test_zero_weight_stays_flat():
    candles = _candles([100, 50, 200])
    res = run_strategy(candles, lambda c: 0.0, fee=0.01)
    assert abs(res["final"] - 1.0) < 1e-9
    assert res["max_drawdown"] == 0.0


def test_weights_clamped_to_unit_interval():
    candles = _candles([100, 110, 121])
    res = run_strategy(candles, lambda c: 5.0, fee=0.0)
    # clamped to 1.0 -> identical to buy and hold
    assert abs(res["final"] - 1.21) < 1e-9


def test_no_lookahead_weight_only_sees_past():
    # weight_fn receives candles[:i]; make it assert it cannot see candle i
    candles = _candles([100, 110, 121, 133])
    seen = []

    def w(cs):
        seen.append(cs[-1]["close"])
        return 1.0

    run_strategy(candles, w, fee=0.0)
    # the last close the strategy ever saw is candle n-2, never the final one
    assert seen[-1] == 121
