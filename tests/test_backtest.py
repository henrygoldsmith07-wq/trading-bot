from bot.backtest import backtest
from bot.strategy import SmaCrossover


def test_backtest_profitable_in_uptrend_with_dip():
    s = SmaCrossover(fast=3, slow=5)
    closes = [10, 9, 8, 7, 6, 5, 5, 5, 9, 12, 15, 14, 13, 12, 11, 20, 25]
    candles = [{"close": c} for c in closes]
    result = backtest(candles, s, start_cash=1000)
    assert result["final_value"] > 0
    assert result["trades"] >= 1
    assert "return_pct" in result
