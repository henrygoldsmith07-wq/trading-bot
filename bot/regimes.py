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
    segments: list[dict] = []
    cur_label: str | None = None
    start: int | None = None
    last: int | None = None
    for t in timeline:
        lbl = labels.get(t, "sideways")
        if lbl != cur_label:
            if cur_label is not None and start is not None and last is not None and (last - start) / DAY_MS + 1 >= min_days:
                segments.append({"label": cur_label, "start": start, "end": last})
            cur_label, start = lbl, t
        last = t
    if cur_label is not None and start is not None and last is not None and (last - start) / DAY_MS + 1 >= min_days:
        segments.append({"label": cur_label, "start": start, "end": last})
    return segments


def stress_mask(btc_candles: list[dict], timeline: list[int], window: int = 30, drop: float = -0.20, vol_pct: float = 90.0) -> dict[int, bool]:
    """Stress day = trailing 30d BTC return below `drop` OR trailing 30d
    annualized vol above its `vol_pct` percentile."""
    import math
    from bisect import bisect_left as bl

    times = [c["open_time"] for c in btc_candles]
    closes = [c["close"] for c in btc_candles]
    vols: dict[int, float | None] = {}
    trailing: dict[int, float | None] = {}
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
    result: dict[int, bool] = {}
    for t in timeline:
        tr = trailing[t]
        vol = vols[t]
        result[t] = (tr is not None and tr <= drop) or (vol is not None and vol >= threshold)
    return result


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


def regime_conditioned_performance(
    returns_by_day: dict[int, float],
    timeline: list[int],
    labels: dict[int, str],
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
) -> dict[str, dict]:
    """Performance of a return stream CONDITIONED on each regime label.

    Answers 'where does the edge live?' — e.g. trend rules that only earn in
    bull segments have a regime-dependent edge, not an unconditional one.
    Per label: days, CAGR (annualized within-regime), volatility, excess
    Sharpe, hit rate, and average exposure proxy (= mean |return| share).
    """
    from .metrics import sharpe as _sharpe
    from .metrics import volatility as _vol

    by_label: dict[str, list[float]] = {}
    for t in timeline:
        lbl = labels.get(t, "sideways")
        if t in returns_by_day:
            by_label.setdefault(lbl, []).append(returns_by_day[t])
    out = {}
    for lbl in sorted(by_label):
        rets = by_label[lbl]
        days = len(rets)
        equity = [1.0]
        for r in rets:
            equity.append(equity[-1] * (1.0 + r))
        out[lbl] = {
            "days": days,
            "share_of_window": days / max(len(timeline), 1),
            "cagr": cagr(equity, days),
            "vol": _vol(rets, periods_per_year),
            "sharpe": _sharpe(rets, periods_per_year, risk_free_annual),
            "hit_rate": sum(1 for r in rets if r > 0) / max(days, 1),
            "max_drawdown": max_drawdown(equity),
        }
    return out
