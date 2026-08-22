"""Trading strategies.

Two interfaces coexist:
- Signal strategies (SmaCrossover): signal(candles) -> "buy"/"sell"/"hold"
- Weight strategies: weight(candles) -> target exposure in [0, 1], long-only.

Weight strategies are what the daily engine and walk-forward validation use.
Every weight strategy is a pure function of the candle window it is given and
only looks at a bounded tail of it, so there is no lookahead by construction.
"""
from __future__ import annotations

import math
from enum import Enum
from statistics import mean, stdev


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return mean(closes[-period:])


def _tail(closes: list[float], size: int) -> list[float]:
    return closes[-size:]


def rsi(closes: list[float], period: int) -> float:
    """Simple-average RSI over the last `period` changes."""
    window = closes[-(period + 1):]
    if len(window) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(1, len(window)):
        ch = window[i] - window[i - 1]
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def ema_series(values: list[float], period: int) -> list[float]:
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


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

    def weight(self, candles: list[dict]) -> float:
        """Long while fast SMA is above slow SMA (stateless crossover proxy)."""
        closes = [c["close"] for c in candles]
        if len(closes) < self.slow:
            return 0.0
        return 1.0 if sma(closes, self.fast) > sma(closes, self.slow) else 0.0


class TrendVol:
    """Long while price is above its SMA, sized to a target annualized volatility.

    The classic time-series-momentum building block: direction from a trend
    filter, position size from inverse realized volatility (capped at 1x so
    the bot stays unlevered).
    """

    PERIODS_PER_YEAR = 365

    def __init__(self, lookback: int = 100, vol_window: int = 20, target_vol: float = 0.40):
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight(self, candles: list[dict]) -> float:
        need = self.lookback + self.vol_window + 1
        closes = [c["close"] for c in candles]
        if len(closes) < need:
            return 0.0
        closes = _tail(closes, need)
        if closes[-1] <= sma(closes, self.lookback):
            return 0.0
        rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(len(closes) - self.vol_window, len(closes))
        ]
        realized = stdev(rets) * math.sqrt(self.PERIODS_PER_YEAR)
        if realized <= 0:
            return 1.0
        return min(1.0, self.target_vol / realized)

    def __repr__(self):
        return f"TrendVol({self.lookback},{self.target_vol:.2f})"


class RsiDipBuy:
    """Hold while RSI is below its exit level and the long-term trend is up.

    A stateless proxy for the classic "buy the dip in an uptrend" rule: the
    position is on whenever short-term momentum has not yet recovered past
    `exit_above` and price sits above its long SMA.
    """

    def __init__(self, period: int = 2, exit_above: float = 65.0, trend_filter: int = 200):
        self.period = period
        self.exit_above = exit_above
        self.trend_filter = trend_filter

    def weight(self, candles: list[dict]) -> float:
        closes = [c["close"] for c in candles]
        need = max(self.trend_filter, self.period + 1)
        if len(closes) < need:
            return 0.0
        closes = _tail(closes, need)
        if closes[-1] <= sma(closes, self.trend_filter):
            return 0.0
        return 1.0 if rsi(closes, self.period) < self.exit_above else 0.0

    def __repr__(self):
        return f"RsiDipBuy({self.period},{self.exit_above:.0f})"


class MacdTrend:
    """Long while the MACD histogram is positive."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def weight(self, candles: list[dict]) -> float:
        warmup = 5 * (self.slow + self.signal)
        closes = [c["close"] for c in candles]
        if len(closes) < self.slow + self.signal + 5:
            return 0.0
        closes = _tail(closes, warmup)
        ema_fast = ema_series(closes, self.fast)
        ema_slow = ema_series(closes, self.slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = ema_series(macd_line, self.signal)
        return 1.0 if macd_line[-1] > signal_line[-1] else 0.0

    def __repr__(self):
        return f"MacdTrend({self.fast},{self.slow},{self.signal})"


class Ensemble:
    """Average of member strategies' weights — diversifies regime bets."""

    def __init__(self, members: list):
        self.members = members

    def weight(self, candles: list[dict]) -> float:
        ws = [m.weight(candles) for m in self.members]
        return max(0.0, min(1.0, sum(ws) / len(ws)))

    def __repr__(self):
        return "Ensemble(" + ",".join(repr(m) for m in self.members) + ")"


class BuyHold:
    """Always fully invested — the benchmark any strategy must beat."""

    def weight(self, candles: list[dict]) -> float:
        return 1.0

    def __repr__(self):
        return "BuyHold"


def default_candidates() -> list:
    """Candidate pool for walk-forward selection."""
    diversified_trend = Ensemble([TrendVol(50, 20, 0.30), TrendVol(100, 20, 0.30), TrendVol(200, 20, 0.30)])
    balanced = Ensemble([TrendVol(50, 20, 0.40), TrendVol(100, 20, 0.25), RsiDipBuy(2, 65, 200)])
    return [
        BuyHold(),
        SmaCrossover(20, 100),
        TrendVol(50, 20, 0.40),
        TrendVol(50, 20, 0.25),
        TrendVol(100, 20, 0.40),
        TrendVol(100, 20, 0.25),
        TrendVol(100, 20, 0.60),
        TrendVol(150, 20, 0.40),
        TrendVol(200, 20, 0.40),
        TrendVol(200, 20, 0.60),
        TrendVol(75, 20, 0.30),
        RsiDipBuy(2, 65, 200),
        RsiDipBuy(3, 60, 150),
        MacdTrend(),
        Ensemble([TrendVol(100, 20, 0.40), RsiDipBuy(2, 65, 200), MacdTrend()]),
        Ensemble([TrendVol(100, 20, 0.40), TrendVol(200, 20, 0.40), RsiDipBuy(2, 65, 200)]),
        diversified_trend,
        balanced,
    ]
