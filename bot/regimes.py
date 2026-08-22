"""Market-regime analysis: label out-of-sample days as bull / bear / sideways
using the BTC proxy's trailing return, segment consecutive labels into
periods, and report strategy vs benchmark behavior per regime — including
high-volatility stress windows.
"""
from __future__ import annotations

from bisect import bisect_left

from .metrics import cagr, max_drawdown

DAY_MS = 86_400_000


def label_regimes(btc_candles: list[dict], timeline: list[int], lookback_days: int = 180, bull: float = 0.25, bear: float = -0.25) -> dict[int, str]:
    """Regime label per timeline day, from BTC's trailing `lookback_days` return.

    bull: trailing return >= +bull; bear: <= bear; otherwise sideways.
    Days without enough history are 'sideways'.
    """
    times = [c["open_time"] for c in btc_candles]
    closes = [c["close"] for c in btc_candles]
    labels = {}
    for t in timeline:
        i = bisect_left(times, t)
        j = bisect_left(times, t - lookback_days * DAY_MS)
        if i <= 0 or j <= 0 or i - j < 10:
            labels[t] = "sideways"
            continue
        tr = closes[i - 1] / closes[j - 1] - 1.0
        if tr >= bull:
            labels[t] = "bull"
        elif tr <= bear:
            labels[t] = "bear"
        else:
            labels[t] = "sideways"
    return labels


def segment(labels: dict[int, str], timeline: list[int], min_days: int = 30) -> list[dict]:
    """Group consecutive same-label days into segments (min length in days)."""
    segments = []
    cur_label, start, last = None, None, None
    for t in timeline:
        lbl = labels.get(t, "sideways")
        if lbl != cur_label:
            if cur_label is not None and (last - start) / DAY_MS + 1 >= min_days:
                segments.append({"label": cur_label, "start": start, "end": last})
            cur_label, start = lbl, t
        last = t
    if cur_label is not None and (last - start) / DAY_MS + 1 >= min_days:
        segments.append({"label": cur_label, "start": start, "end": last})
    return segments


def stress_mask(btc_candles: list[dict], timeline: list[int], window: int = 30, drop: float = -0.20, vol_pct: float = 90.0) -> dict[int, bool]:
    """Stress day = trailing 30d BTC return below `drop` OR trailing 30d
    annualized vol above its `vol_pct` percentile."""
    import math
    from bisect import bisect_left as bl

    times = [c["open_time"] for c in btc_candles]
    closes = [c["close"] for c in btc_candles]
    vols = {}
    trailing = {}
    for t in timeline:
        i = bl(times, t)
        j = bl(times, t - window * DAY_MS)
        if i < window + 1 or i - j < window:
            trailing[t] = None
            vols[t] = None
            continue
        trailing[t] = closes[i - 1] / closes[i - 1 - window] - 1.0
        rets = [math.log(closes[k] / closes[k - 1]) for k in range(i - window, i)]
        m = sum(rets) / window
        var = sum((x - m) ** 2 for x in rets) / (window - 1)
        vols[t] = math.sqrt(max(var, 0.0) * 365)
    observed = sorted(v for v in vols.values() if v is not None)
    if not observed:
        return {t: False for t in timeline}
    threshold = observed[int(len(observed) * vol_pct / 100.0) - 1]
    return {
        t: (trailing[t] is not None and trailing[t] <= drop) or (vols[t] is not None and vols[t] >= threshold)
        for t in timeline
    }


def segment_metrics(returns_by_day: dict[int, float], timeline: list[int], start: int, end: int) -> dict:
    """CAGR/max drawdown of a return stream restricted to [start, end]."""
    rets = [returns_by_day[t] for t in timeline if start <= t <= end]
    days = len(rets)
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1.0 + r))
    return {
        "days": days,
        "cagr": cagr(equity, max(days, 1)),
        "max_drawdown": max_drawdown(equity),
        "final": equity[-1],
    }
