"""Trading strategies.

Each strategy takes a list of candles (dicts with 'close' prices) and returns
a Signal: "buy", "sell", or "hold".
"""
from __future__ import annotations

from enum import Enum
from statistics import mean


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return mean(closes[-period:])


class SmaCrossover:
    """Buy when the fast SMA crosses above the slow SMA, sell on the reverse."""

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow

    def signal(self, candles: list[dict]) -> Signal:
        closes = [c["close"] for c in candles]
        if len(closes) < self.slow + 1:
            return Signal.HOLD

        fast_now = sma(closes, self.fast)
        slow_now = sma(closes, self.slow)
        prev_closes = closes[:-1]
        fast_prev = sma(prev_closes, self.fast)
        slow_prev = sma(prev_closes, self.slow)

        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal.BUY
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal.SELL
        return Signal.HOLD
