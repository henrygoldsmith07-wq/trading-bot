from bot.strategy import SmaCrossover, Signal


def _candles(closes):
    return [{"close": c} for c in closes]


def test_golden_cross_buys():
    s = SmaCrossover(fast=3, slow=5)
    # downtrend then sharp uptrend -> fast crosses above slow
    closes = [10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 15]
    assert s.signal(_candles(closes)) is Signal.BUY


def test_death_cross_sells():
    s = SmaCrossover(fast=3, slow=5)
    closes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2]
    assert s.signal(_candles(closes)) is Signal.SELL


def test_hold_when_not_enough_data():
    s = SmaCrossover(fast=3, slow=5)
    assert s.signal(_candles([1, 2, 3])) is Signal.HOLD


def test_hold_in_steady_uptrend():
    s = SmaCrossover(fast=3, slow=5)
    closes = [1, 2, 3, 4, 5, 6]  # fast already above slow, no new cross
    assert s.signal(_candles(closes)) is Signal.HOLD
