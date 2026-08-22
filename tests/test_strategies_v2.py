from bot.strategy import Ensemble, MacdTrend, RsiDipBuy, SmaCrossover, TrendVol, sma


def _candles(closes):
    return [{"close": c} for c in closes]


def _rising(n=400, start=100.0, growth=1.005):
    return [start * growth ** i for i in range(n)]


def test_weights_bounded_0_1():
    up = _candles(_rising())
    down = _candles(list(reversed(_rising())))
    strategies = [
        SmaCrossover(20, 100),
        TrendVol(100, 20, 0.4),
        RsiDipBuy(2, 65, 200),
        MacdTrend(),
        Ensemble([TrendVol(100, 20, 0.4), MacdTrend()]),
    ]
    for s in strategies:
        for data in (up, down):
            w = s.weight(data)
            assert 0.0 <= w <= 1.0, f"{s!r} weight {w} out of bounds"


def test_short_windows_return_zero():
    for s in [TrendVol(100, 20, 0.4), RsiDipBuy(2, 65, 200), MacdTrend()]:
        assert s.weight(_candles([1, 2, 3])) == 0.0


def test_trendvol_long_in_uptrend():
    assert TrendVol(100, 20, 0.4).weight(_candles(_rising())) > 0.0


def test_trendvol_flat_in_downtrend():
    assert TrendVol(100, 20, 0.4).weight(_candles(list(reversed(_rising())))) == 0.0


def test_trendvol_vol_target_scales_down_in_high_vol():
    calm = [100 * 1.005 ** i for i in range(200)]
    wild = []
    px = 100.0
    for i in range(200):
        px *= 1.05 if i % 2 == 0 else 0.9524
        wild.append(px)
    low_vol_w = TrendVol(100, 20, 0.40).weight(_candles(calm))
    high_vol_w = TrendVol(100, 20, 0.40).weight(_candles(wild))
    assert high_vol_w < low_vol_w


def test_sma_crossover_weight_matches_signal_state():
    closes = [10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 15]
    s = SmaCrossover(3, 5)
    assert s.signal(_candles(closes)).value == "buy"
    assert s.weight(_candles(closes)) == 1.0


def test_default_candidates_instantiable():
    from bot.strategy import default_candidates

    assert len(default_candidates()) >= 10
