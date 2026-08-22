"""Daily-bar backtest engine.

The engine is deliberately simple and honest:
- A weight function sees candles up to and including day i-1's close and
  decides the position held during day i (no lookahead).
- Changing position costs `fee` times the turnover.
- Weights are clamped to [0, 1]: long-only, unlevered.

weight_fn has signature (candles, i) -> float so strategies can use cached
prefix-sum arrays over the shared candle list instead of re-slicing windows.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .metrics import summarize

DAY_MS = 86_400_000


def candle_date(candle: dict):
    return datetime.fromtimestamp(candle["open_time"] / 1000, tz=timezone.utc).date()


def run_strategy(
    candles: list[dict],
    weight_fn,
    fee: float = 0.001,
    periods_per_year: int = 365,
    start_index: int = 1,
) -> dict:
    """Backtest `weight_fn` over `candles`.

    weight_fn(candles, i) -> target exposure for day i, using data through
    day i-1 only. Returns equity curve, daily returns, and summary stats.
    """
    n = len(candles)
    if n < 2:
        raise ValueError("need at least 2 candles")
    equity = [1.0]
    returns = []
    weights = []
    prev_w = 0.0
    closes = [c["close"] for c in candles]
    for i in range(start_index, n):
        w = min(1.0, max(0.0, weight_fn(candles, i)))
        r = w * (closes[i] / closes[i - 1] - 1.0) - fee * abs(w - prev_w)
        equity.append(equity[-1] * (1.0 + r))
        returns.append(r)
        weights.append(w)
        prev_w = w

    days = (candles[n - 1]["open_time"] - candles[start_index - 1]["open_time"]) / DAY_MS
    stats = summarize(equity, returns, days, periods_per_year)
    stats["exposure"] = sum(weights) / len(weights) if weights else 0.0
    stats["turnover"] = sum(abs(weights[i] - (weights[i - 1] if i else 0.0)) for i in range(len(weights)))
    stats["equity"] = equity
    stats["returns"] = returns
    stats["weights"] = weights
    stats["return_days"] = [(candles[i]["open_time"], returns[k]) for k, i in enumerate(range(start_index, n))]
    return stats
