"""Daily-bar backtest engine.

The engine is deliberately simple and honest:
- A weight function sees candles up to and including day t's close and decides
  the position held during day t+1 (no lookahead).
- Changing position costs `fee` times the turnover.
- Weights are clamped to [0, 1]: long-only, unlevered.
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

    weight_fn(candles[:i]) -> float in [0, 1] is the target exposure for day i.
    Returns equity curve, daily returns, and summary stats.
    """
    n = len(candles)
    if n < 2:
        raise ValueError("need at least 2 candles")
    equity = [1.0]
    returns = []
    weights = []
    prev_w = 0.0
    for i in range(start_index, n):
        w = min(1.0, max(0.0, weight_fn(candles[:i])))
        asset_ret = candles[i]["close"] / candles[i - 1]["close"] - 1.0
        cost = fee * abs(w - prev_w)
        r = w * asset_ret - cost
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
    return stats


def buy_hold_weight(candles: list[dict]) -> float:
    return 1.0
