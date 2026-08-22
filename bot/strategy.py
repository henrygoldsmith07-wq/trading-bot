"""Trading strategies.

Two interfaces coexist:
- Signal strategies (SmaCrossover): signal(candles) -> "buy"/"sell"/"hold"
- Weight strategies: weight_at(candles, i) -> target exposure in [0, 1] for
  day i, decided at the close of day i-1. Long-only.

Rolling indicators are computed from cached prefix-sum arrays over the candle
list, so a full backtest of one candidate costs O(n) rather than O(n*window).
The cache is keyed to the identity and endpoints of the candle list; the
engine passes the same unmutated list for every day of a run.
"""
from __future__ import annotations

import math
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
    if gains == 0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


class _Series:
    """Prefix sums over closes for O(1) rolling mean / RSI / volatility."""

    __slots__ = ("n", "close", "ps", "gains", "losses", "pr", "pr2")

    def __init__(self, candles: list[dict]):
        n = len(candles)
        close = [float(c["close"]) for c in candles]
        ps = [0.0] * (n + 1)
        gains = [0.0] * (n + 1)
        losses = [0.0] * (n + 1)
        pr = [0.0] * (n + 1)
        pr2 = [0.0] * (n + 1)
        for j in range(n):
            ps[j + 1] = ps[j] + close[j]
            if j:
                ch = close[j] - close[j - 1]
                gains[j + 1] = gains[j] + (ch if ch > 0 else 0.0)
                losses[j + 1] = losses[j] + (-ch if ch < 0 else 0.0)
                r = math.log(close[j] / close[j - 1])
                pr[j + 1] = pr[j] + r
                pr2[j + 1] = pr2[j] + r * r
        self.n = n
        self.close = close
        self.ps = ps
        self.gains = gains
        self.losses = losses
        self.pr = pr
        self.pr2 = pr2

    def mean(self, end: int, period: int) -> float:
        """Mean close of candles[end-period:end]."""
        return (self.ps[end] - self.ps[end - period]) / period

    def rsi(self, end: int, period: int) -> float:
        g = self.gains[end] - self.gains[end - period]
        l = self.losses[end] - self.losses[end - period]
        if l <= 0:
            return 100.0
        if g <= 0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + g / l)

    def ann_vol(self, end: int, window: int, periods_per_year: int = 365) -> float:
        m = window
        s = self.pr[end] - self.pr[end - m]
        s2 = self.pr2[end] - self.pr2[end - m]
        var = (s2 - s * s / m) / (m - 1)
        if var <= 0:
            return 0.0
        return math.sqrt(max(var, 0.0) * periods_per_year)


def _cache_key(candles: list[dict]):
    first, last = candles[0], candles[-1]
    return (
        id(candles),
        len(candles),
        first.get("open_time"),
        last.get("open_time"),
        first["close"],
        last["close"],
    )


class WeightStrategy:
    """Base class for weight-based strategies."""

    def weight_at(self, candles: list[dict], i: int) -> float:
        raise NotImplementedError

    def weight(self, candles: list[dict]) -> float:
        """Target exposure given the full candle list (live/one-shot use)."""
        return self.weight_at(candles, len(candles))

    def _series(self, candles: list[dict]) -> _Series:
        key = _cache_key(candles)
        if getattr(self, "_skey", None) != key:
            self._skey = key
            self._s = _Series(candles)
        return self._s


class BuyHold(WeightStrategy):
    """Always fully invested — the benchmark any strategy must beat."""

    def weight_at(self, candles, i):
        return 1.0

    def __repr__(self):
        return "BuyHold"


class SmaCrossover(WeightStrategy):
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

    def weight_at(self, candles, i):
        if i < self.slow:
            return 0.0
        s = self._series(candles)
        return 1.0 if s.mean(i, self.fast) > s.mean(i, self.slow) else 0.0

    def __repr__(self):
        return f"SmaCross({self.fast},{self.slow})"


class TrendVol(WeightStrategy):
    """Long while price is above its SMA, sized to a target annualized volatility.

    The classic time-series-momentum building block: direction from a trend
    filter, position size from inverse realized volatility (capped at 1x so
    the bot stays unlevered).
    """

    def __init__(self, lookback: int = 100, vol_window: int = 20, target_vol: float = 0.40):
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight_at(self, candles, i):
        if i < self.lookback + self.vol_window + 1:
            return 0.0
        s = self._series(candles)
        if s.close[i - 1] <= s.mean(i, self.lookback):
            return 0.0
        rv = s.ann_vol(i, self.vol_window)
        if rv <= 0:
            return 1.0
        return min(1.0, self.target_vol / rv)

    def __repr__(self):
        return f"TrendVol({self.lookback},{self.target_vol:.2f})"


class RsiDipBuy(WeightStrategy):
    """Hold while RSI is below its exit level and the long-term trend is up.

    A stateless proxy for the classic "buy the dip in an uptrend" rule: the
    position is on whenever short-term momentum has not yet recovered past
    `exit_above` and price sits above its long SMA.
    """

    def __init__(self, period: int = 2, exit_above: float = 65.0, trend_filter: int = 200):
        self.period = period
        self.exit_above = exit_above
        self.trend_filter = trend_filter

    def weight_at(self, candles, i):
        need = max(self.trend_filter, self.period + 1)
        if i < need:
            return 0.0
        s = self._series(candles)
        if s.close[i - 1] <= s.mean(i, self.trend_filter):
            return 0.0
        return 1.0 if s.rsi(i, self.period) < self.exit_above else 0.0

    def __repr__(self):
        return f"RsiDipBuy({self.period},{self.exit_above:.0f},{self.trend_filter})"


class MacdTrend(WeightStrategy):
    """Long while the MACD histogram is positive."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def _arrays(self, candles):
        key = _cache_key(candles)
        if getattr(self, "_mkey", None) != key:
            k_f = 2.0 / (self.fast + 1.0)
            k_s = 2.0 / (self.slow + 1.0)
            k_g = 2.0 / (self.signal + 1.0)
            close = [float(c["close"]) for c in candles]
            ema_f = [close[0]]
            ema_s = [close[0]]
            for v in close[1:]:
                ema_f.append(v * k_f + ema_f[-1] * (1.0 - k_f))
                ema_s.append(v * k_s + ema_s[-1] * (1.0 - k_s))
            macd = [f - s for f, s in zip(ema_f, ema_s)]
            sig = [macd[0]]
            for v in macd[1:]:
                sig.append(v * k_g + sig[-1] * (1.0 - k_g))
            self._mkey = key
            self._m = (macd, sig)
        return self._m

    def weight_at(self, candles, i):
        if i < self.slow + self.signal + 5:
            return 0.0
        macd, sig = self._arrays(candles)
        return 1.0 if macd[i - 1] > sig[i - 1] else 0.0

    def __repr__(self):
        return f"MacdTrend({self.fast},{self.slow},{self.signal})"


class Ensemble(WeightStrategy):
    """Average of member strategies' weights — diversifies regime bets."""

    def __init__(self, members: list):
        self.members = members

    def weight_at(self, candles, i):
        total = 0.0
        for m in self.members:
            total += m.weight_at(candles, i)
        w = total / len(self.members)
        return max(0.0, min(1.0, w))

    def __repr__(self):
        return "Ensemble(" + ",".join(repr(m) for m in self.members) + ")"


def build_candidates() -> list:
    """Systematic candidate pool for walk-forward selection (~70 strategies).

    A grid over trend lookbacks, volatility targets, RSI dip-buy settings,
    MACD parameter sets, and blends of the above.
    """
    candidates = [BuyHold()]
    candidates += [
        SmaCrossover(f, s)
        for f, s in [(10, 50), (20, 100), (50, 150)]
    ]
    for lookback in (25, 50, 75, 100, 125, 150, 200):
        for target in (0.20, 0.25, 0.30, 0.40, 0.50):
            candidates.append(TrendVol(lookback, 20, target))
    for period in (2, 3, 4):
        for exit_above in (55, 60, 65, 70):
            for trend_filter in (150, 200):
                candidates.append(RsiDipBuy(period, exit_above, trend_filter))
    for f, s, g in [(12, 26, 9), (8, 21, 5), (16, 32, 9)]:
        candidates.append(MacdTrend(f, s, g))
    candidates += [
        Ensemble([TrendVol(50, 20, 0.30), TrendVol(100, 20, 0.30), TrendVol(200, 20, 0.30)]),
        Ensemble([TrendVol(50, 20, 0.40), TrendVol(100, 20, 0.25), RsiDipBuy(2, 65, 200)]),
        Ensemble([TrendVol(100, 20, 0.40), RsiDipBuy(2, 65, 200), MacdTrend()]),
        Ensemble([TrendVol(100, 20, 0.40), TrendVol(200, 20, 0.40), RsiDipBuy(2, 65, 200)]),
        Ensemble([TrendVol(50, 20, 0.25), RsiDipBuy(3, 60, 150), MacdTrend(8, 21, 5)]),
        Ensemble([TrendVol(75, 20, 0.30), TrendVol(150, 20, 0.30)]),
        Ensemble([RsiDipBuy(2, 65, 200), RsiDipBuy(3, 60, 150), MacdTrend()]),
        Ensemble([TrendVol(25, 20, 0.40), TrendVol(50, 20, 0.40), TrendVol(100, 20, 0.40)]),
    ]
    return candidates


# Backwards-compatible alias
def default_candidates() -> list:
    return build_candidates()


_STRATEGY_TYPES = {
    "BuyHold": BuyHold,
    "SmaCrossover": SmaCrossover,
    "TrendVol": TrendVol,
    "RsiDipBuy": RsiDipBuy,
    "MacdTrend": MacdTrend,
}


def strategy_to_spec(strategy) -> dict:
    """Serializable spec for freezing a strategy exactly as-is."""
    name = type(strategy).__name__
    if name not in _STRATEGY_TYPES:
        raise ValueError(f"strategy {name!r} cannot be frozen")
    params = {
        k: v
        for k, v in vars(strategy).items()
        if not k.startswith("_") and isinstance(v, (int, float, str, bool))
    }
    return {"type": name, "params": params}


def strategy_from_spec(spec: dict):
    """Rebuild a frozen strategy. Raises on anything unknown — the forward
    runner never falls back to re-selection."""
    t = _STRATEGY_TYPES.get(spec.get("type"))
    if t is None:
        raise ValueError(f"unknown frozen strategy {spec!r}")
    return t(**spec.get("params", {}))
