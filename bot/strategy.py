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
from collections import deque
from enum import Enum
from statistics import mean
from typing import Any


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


def _rolling_extreme(values: list[float], window: int, is_max: bool) -> list[float]:
    """Rolling max (or min) over a trailing window, O(n) total.

    Monotonic-deque algorithm: element i of the result is the extreme of
    values[i - window + 1 : i + 1], which is `nan` until the window fills.
    Rolling extremes are NOT decomposable from prefix sums, so they cannot
    reuse the O(1) trick the mean/vol paths use — this keeps them O(n)
    instead of O(n * window), which matters when a walk-forward evaluates
    dozens of breakout candidates over thousands of bars.
    """
    n = len(values)
    out = [float("nan")] * n
    if window <= 0 or n == 0:
        return out
    dq: deque[int] = deque()
    for i, v in enumerate(values):
        while dq and (values[dq[-1]] <= v if is_max else values[dq[-1]] >= v):
            dq.pop()
        dq.append(i)
        while dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            out[i] = values[dq[0]]
    return out


class _Series:
    """Prefix sums over closes for O(1) rolling mean / RSI / volatility.

    Also carries a price-dispersion prefix sum (rolling stdev of price, for
    z-score rules) and a memoised rolling-extreme table (for channel
    breakouts), both built lazily so strategies that never ask for them pay
    nothing.
    """

    __slots__ = ("n", "close", "ps", "ps2", "gains", "losses", "pr", "pr2", "_ext")

    def __init__(self, candles: list[dict]):
        n = len(candles)
        close = [float(c["close"]) for c in candles]
        ps = [0.0] * (n + 1)
        ps2 = [0.0] * (n + 1)
        gains = [0.0] * (n + 1)
        losses = [0.0] * (n + 1)
        pr = [0.0] * (n + 1)
        pr2 = [0.0] * (n + 1)
        for j in range(n):
            ps[j + 1] = ps[j] + close[j]
            ps2[j + 1] = ps2[j] + close[j] * close[j]
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
        self.ps2 = ps2
        self.gains = gains
        self.losses = losses
        self.pr = pr
        self.pr2 = pr2
        self._ext: dict[tuple[int, bool], list[float]] = {}

    def mean(self, end: int, period: int) -> float:
        """Mean close of candles[end-period:end]."""
        return (self.ps[end] - self.ps[end - period]) / period

    def stdev(self, end: int, period: int) -> float:
        """Sample stdev of closes[end-period:end] (same units as price)."""
        m = period
        if m < 2 or end - m < 0:
            return 0.0
        s = self.ps[end] - self.ps[end - m]
        s2 = self.ps2[end] - self.ps2[end - m]
        var = (s2 - s * s / m) / (m - 1)
        return math.sqrt(max(var, 0.0))

    def rsi(self, end: int, period: int) -> float:
        g = self.gains[end] - self.gains[end - period]
        losses = self.losses[end] - self.losses[end - period]
        if losses <= 0:
            return 100.0
        if g <= 0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + g / losses)

    def ann_vol(self, end: int, window: int, periods_per_year: int = 365) -> float:
        m = window
        s = self.pr[end] - self.pr[end - m]
        s2 = self.pr2[end] - self.pr2[end - m]
        var = (s2 - s * s / m) / (m - 1)
        if var <= 0:
            return 0.0
        return math.sqrt(max(var, 0.0) * periods_per_year)

    def _extreme(self, window: int, is_max: bool) -> list[float]:
        key = (window, is_max)
        arr = self._ext.get(key)
        if arr is None:
            arr = _rolling_extreme(self.close, window, is_max)
            self._ext[key] = arr
        return arr

    def roll_max(self, end: int, window: int) -> float:
        """Highest close in candles[end-window:end]."""
        if end < 1 or end - window < 0:
            return float("nan")
        return self._extreme(window, True)[end - 1]

    def roll_min(self, end: int, window: int) -> float:
        """Lowest close in candles[end-window:end]."""
        if end < 1 or end - window < 0:
            return float("nan")
        return self._extreme(window, False)[end - 1]


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

        if None in (fast_now, slow_now, fast_prev, slow_prev):
            return Signal.HOLD
        assert fast_now is not None and slow_now is not None
        assert fast_prev is not None and slow_prev is not None

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
            macd = [f - s for f, s in zip(ema_f, ema_s, strict=True)]
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


class TSMom(WeightStrategy):
    """Time-series momentum: long while the cumulative return over the last
    `horizon` days is positive, sized by inverse realized volatility."""

    def __init__(self, horizon: int = 63, vol_window: int = 20, target_vol: float = 0.35):
        self.horizon = horizon
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight_at(self, candles, i):
        if i < self.horizon + self.vol_window + 1:
            return 0.0
        s = self._series(candles)
        if s.pr[i] - s.pr[i - self.horizon] <= 0:
            return 0.0
        rv = s.ann_vol(i, self.vol_window)
        if rv <= 0:
            return 1.0
        return min(1.0, self.target_vol / rv)

    def __repr__(self):
        return f"TSMom({self.horizon},{self.target_vol:.2f})"


class DualMomentum(WeightStrategy):
    """Multi-horizon time-series momentum: weight scales with the fraction of
    horizons whose trailing return is positive, then by inverse volatility."""

    def __init__(self, horizons=(63, 126, 252), vol_window: int = 20, target_vol: float = 0.30):
        self.horizons = tuple(horizons)
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight_at(self, candles, i):
        need = max(self.horizons) + self.vol_window + 1
        if i < need:
            return 0.0
        s = self._series(candles)
        up = sum(1 for h in self.horizons if s.pr[i] - s.pr[i - h] > 0)
        base = up / len(self.horizons)
        if base == 0:
            return 0.0
        rv = s.ann_vol(i, self.vol_window)
        scale = min(1.0, self.target_vol / rv) if rv > 0 else 1.0
        return base * scale

    def __repr__(self):
        hs = "/".join(str(h) for h in self.horizons)
        return f"DualMom({hs},{self.target_vol:.2f})"


class ChannelBreakout(WeightStrategy):
    """Long while price sits in the top of its own Donchian channel.

    The signal is RANGE POSITION, not a level or a return:

        pos = (close - channel_low) / (channel_high - channel_low)

    which makes it the only scale-invariant family in the pool: multiply
    every price by a constant and the signal is unchanged. The SMA
    crossover compares a level to a mean (units of price) and the momentum
    families compare returns, so both carry the asset's price scale into
    the decision. Range position does not, which is what lets one rule
    travel across BTC at $60k and a penny-alt at $0.003 without
    re-parameterising.

    Hysteresis: on above `entry_frac`, off below `exit_frac`, and inside
    the band the state is recovered deterministically at the band
    midpoint. `weight_at` must stay a pure function of (candles, i) —
    the forward runner re-derives a single day's weight from history, so
    the strategy cannot carry mutable state — and the midpoint is the
    same convention a vectorised stochastic-oscillator backtest uses.
    """

    def __init__(
        self,
        channel: int = 50,
        entry_frac: float = 0.80,
        exit_frac: float = 0.40,
        vol_window: int = 20,
        target_vol: float = 0.30,
    ):
        self.channel = channel
        self.entry_frac = entry_frac
        self.exit_frac = exit_frac
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight_at(self, candles, i):
        if i < self.channel + self.vol_window + 1:
            return 0.0
        s = self._series(candles)
        lo = s.roll_min(i, self.channel)
        hi = s.roll_max(i, self.channel)
        # nan propagates from an unfilled window; treat "no channel" as flat
        if not (hi == hi and lo == lo) or hi <= lo:
            return 0.0
        px = s.close[i - 1]
        pos = (px - lo) / (hi - lo)
        if pos >= self.entry_frac:
            on = True
        elif pos <= self.exit_frac:
            on = False
        else:
            on = pos >= 0.5 * (self.entry_frac + self.exit_frac)
        if not on:
            return 0.0
        rv = s.ann_vol(i, self.vol_window)
        if rv <= 0:
            return 1.0
        return min(1.0, self.target_vol / rv)

    def __repr__(self):
        return f"ChanBreak({self.channel},{self.entry_frac:.2f},{self.exit_frac:.2f},{self.target_vol:.2f})"


class MeanReversionZ(WeightStrategy):
    """Long while price is dislocated BELOW its own trailing mean.

    The z-score (close - SMA) / stdev(close) is dimensionless in a
    different way from ChannelBreakout: it measures displacement in units
    of the asset's own recent dispersion rather than in units of its
    trading range. That makes it the contrapuntal family to the trend
    rules — it is the only candidate that is structurally designed to
    buy weakness, where TrendVol/TSMom/DualMomentum all buy strength and
    RsiDipBuy only buys shallow pullbacks inside an uptrend.

    Deliberately gated behind a long-term trend filter: unfiltered mean
    reversion in a downtrend is the classic way to lose money slowly.
    """

    def __init__(
        self,
        window: int = 30,
        entry_z: float = -0.75,
        exit_z: float = 0.50,
        trend_filter: int = 200,
        vol_window: int = 20,
        target_vol: float = 0.30,
    ):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.trend_filter = trend_filter
        self.vol_window = vol_window
        self.target_vol = target_vol

    def weight_at(self, candles, i):
        if i < max(self.window, self.trend_filter) + self.vol_window + 1:
            return 0.0
        s = self._series(candles)
        if s.close[i - 1] <= s.mean(i, self.trend_filter):
            return 0.0
        sd = s.stdev(i, self.window)
        if sd <= 0:
            return 0.0
        z = (s.close[i - 1] - s.mean(i, self.window)) / sd
        if z <= self.entry_z:
            on = True
        elif z >= self.exit_z:
            on = False
        else:
            on = z <= 0.5 * (self.entry_z + self.exit_z)
        if not on:
            return 0.0
        rv = s.ann_vol(i, self.vol_window)
        if rv <= 0:
            return 1.0
        return min(1.0, self.target_vol / rv)

    def __repr__(self):
        return f"MeanRevZ({self.window},{self.entry_z:.2f},{self.exit_z:.2f},{self.target_vol:.2f})"


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


class Blend(WeightStrategy):
    """Explicitly-weighted convex combination of member strategies.

    `Ensemble` is the equal-weight special case. Carrying an explicit
    weight vector is what lets a selection rule *shrink*: a pick can be
    held at (1 - alpha) alongside a fixed a-priori prior at alpha, rather
    than forcing the all-or-nothing choice between "trust the search" and
    "ignore the search" that makes the deflated Sharpe collapse.

    Weights are normalised to sum to 1 so a Blend is always unlevered.
    """

    def __init__(self, members: list, weights: list[float] | None = None):
        if not members:
            raise ValueError("Blend needs at least one member")
        self.members = list(members)
        if weights is None:
            n = len(self.members)
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != len(members):
                raise ValueError("Blend weights must match members")
            if any(w < 0 for w in weights):
                raise ValueError("Blend weights must be non-negative")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Blend weights must sum to a positive number")
            self.weights = [w / total for w in weights]

    def weight_at(self, candles, i):
        total = 0.0
        for w, m in zip(self.weights, self.members):
            total += w * m.weight_at(candles, i)
        return max(0.0, min(1.0, total))

    def __repr__(self):
        parts = ",".join(f"{w:.4g}*{m!r}" for w, m in zip(self.weights, self.members))
        return f"Blend({parts})"


def risk_ensemble() -> Ensemble:
    """The a-priori, no-selection strategy: a fixed blend of trend, momentum,
    and dip-buying across horizons. Using it directly makes the multiple-
    testing trial count 1 instead of 74, which is exactly what the deflated
    Sharpe ratio punishes."""
    return Ensemble(
        [
            TrendVol(50, 20, 0.30),
            TrendVol(100, 20, 0.30),
            TSMom(63, 20, 0.30),
            DualMomentum((63, 126, 252), 20, 0.30),
            RsiDipBuy(2, 65, 200),
        ]
    )


def _extended_candidates() -> list:
    """The two families the base pool is missing.

    `build_candidates()` is ~85 strategies but correlation clustering
    (bot/clustering.py) collapses them to roughly 22 *families*: the
    TrendVol grid alone is 35 near-duplicates of one idea. Adding another
    lookback to that grid buys nothing. These two are structurally new —
    order statistics and standardized dispersion, respectively — so they
    add genuine regime diversity rather than another near-duplicate.

    Opt-in: extending the base pool changes `candidate_pool_version`, which
    is sealed into every freeze, so a frozen experiment keeps the pool it
    was frozen with.
    """
    out: list = []
    # Channel breakout: scale-invariant range position.
    for channel in (30, 50, 100):
        for entry_frac, exit_frac in ((0.80, 0.40), (0.70, 0.30), (0.90, 0.50)):
            for target in (0.30, 0.40):
                out.append(ChannelBreakout(channel, entry_frac, exit_frac, 20, target))
    # Mean reversion in z-space: the only family designed to buy weakness.
    for window in (20, 30, 50):
        for entry_z, exit_z in ((-0.50, 0.50), (-1.00, 0.75)):
            for target in (0.30, 0.40):
                out.append(MeanReversionZ(window, entry_z, exit_z, 200, 20, target))
    return out


def build_candidates(extended: bool = False) -> list:
    """Systematic candidate pool for walk-forward selection (~85 strategies).

    A grid over trend lookbacks, volatility targets, RSI dip-buy settings,
    MACD parameter sets, momentum horizons, and blends of the above.

    `extended=True` appends the ChannelBreakout and MeanReversionZ families
    (see `_extended_candidates`). The default pool is unchanged, so the
    sealed `candidate_pool_version` — and therefore every existing freeze —
    is bit-identical.
    """
    candidates: list = [BuyHold()]
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
    for horizon, target in [(63, 0.30), (63, 0.45), (126, 0.30), (126, 0.45), (189, 0.35)]:
        candidates.append(TSMom(horizon, 20, target))
    candidates += [
        DualMomentum((63, 126, 252), 20, 0.30),
        DualMomentum((42, 84, 168), 20, 0.35),
        DualMomentum((63, 126), 20, 0.30),
    ]
    candidates += [
        Ensemble([TrendVol(50, 20, 0.30), TrendVol(100, 20, 0.30), TrendVol(200, 20, 0.30)]),
        Ensemble([TrendVol(50, 20, 0.40), TrendVol(100, 20, 0.25), RsiDipBuy(2, 65, 200)]),
        Ensemble([TrendVol(100, 20, 0.40), RsiDipBuy(2, 65, 200), MacdTrend()]),
        Ensemble([TrendVol(100, 20, 0.40), TrendVol(200, 20, 0.40), RsiDipBuy(2, 65, 200)]),
        Ensemble([TrendVol(50, 20, 0.25), RsiDipBuy(3, 60, 150), MacdTrend(8, 21, 5)]),
        Ensemble([TrendVol(75, 20, 0.30), TrendVol(150, 20, 0.30)]),
        Ensemble([RsiDipBuy(2, 65, 200), RsiDipBuy(3, 60, 150), MacdTrend()]),
        Ensemble([TrendVol(25, 20, 0.40), TrendVol(50, 20, 0.40), TrendVol(100, 20, 0.40)]),
        Ensemble([TrendVol(50, 20, 0.30), TSMom(63, 20, 0.30), RsiDipBuy(2, 65, 200)]),
        Ensemble([TSMom(63, 20, 0.35), TSMom(126, 20, 0.35), DualMomentum((63, 126, 252), 20, 0.30)]),
        risk_ensemble(),
    ]
    if extended:
        candidates += _extended_candidates()
        # blends across the new families, so a fold can hold breakout and
        # dip-buying at once instead of choosing between them
        candidates += [
            Ensemble([TrendVol(100, 20, 0.30), ChannelBreakout(50, 0.80, 0.40, 20, 0.30)]),
            Ensemble([TrendVol(100, 20, 0.30), MeanReversionZ(30, -0.75, 0.50, 200, 20, 0.30)]),
            Ensemble([ChannelBreakout(50, 0.80, 0.40, 20, 0.30), MeanReversionZ(30, -0.75, 0.50, 200, 20, 0.30)]),
            Ensemble([
                TrendVol(100, 20, 0.30),
                ChannelBreakout(50, 0.80, 0.40, 20, 0.30),
                MeanReversionZ(30, -0.75, 0.50, 200, 20, 0.30),
            ]),
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
    "TSMom": TSMom,
    "DualMomentum": DualMomentum,
    "ChannelBreakout": ChannelBreakout,
    "MeanReversionZ": MeanReversionZ,
    "Ensemble": Ensemble,
    "Blend": Blend,
}

# Strategies that hold other strategies and therefore need recursive
# serialisation rather than the flat scalar/sequence walk below.
_CONTAINER_TYPES = frozenset({"Ensemble", "Blend"})


def strategy_to_spec(strategy) -> dict:
    """Serializable spec for freezing a strategy exactly as-is."""
    name = type(strategy).__name__
    if name not in _STRATEGY_TYPES:
        raise ValueError(f"strategy {name!r} cannot be frozen")
    if name in _CONTAINER_TYPES:
        # Annotated explicitly: without it mypy joins `str` and `list[dict]`
        # into a value type too narrow to accept the weight vector below.
        spec: dict[str, Any] = {"type": name, "members": [strategy_to_spec(m) for m in strategy.members]}
        if name == "Blend":
            # the weight vector is the whole point of a Blend; dropping it
            # would silently rebuild an equal-weight Ensemble
            spec["weights"] = [float(w) for w in strategy.weights]
        return spec
    params: dict = {}
    for k, v in vars(strategy).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool)):
            params[k] = v
        elif isinstance(v, (tuple, list)):
            # sequences (e.g. DualMomentum horizons) must survive the freeze:
            # dropping them would silently resurrect defaults on rebuild
            params[k] = list(v)
    return {"type": name, "params": params}


def strategy_from_spec(spec: dict):
    """Rebuild a frozen strategy. Raises on anything unknown — the forward
    runner never falls back to re-selection."""
    type_name = spec.get("type")
    t = _STRATEGY_TYPES.get(type_name) if isinstance(type_name, str) else None
    if t is None:
        raise ValueError(f"unknown frozen strategy {spec!r}")
    if type_name in _CONTAINER_TYPES:
        members = [strategy_from_spec(m) for m in spec.get("members", [])]
        if type_name == "Blend":
            return Blend(members, spec.get("weights"))
        return Ensemble(members)
    params: dict = dict(spec.get("params", {}))
    # sequences were frozen as lists; restore the tuple form constructors expect
    for k, v in params.items():
        if isinstance(v, list):
            params[k] = tuple(v)
    return t(**params)
