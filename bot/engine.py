"""Daily-bar backtest engine with realistic frictions.

Honesty and realism features:
- No lookahead: the weight for day i is computed from candles strictly
  before day i (shifted further by `latency_days`).
- Execution models:
  * "close"     — rebalance at day i's close (optimistic baseline).
  * "next_open" — signal at close of day i-1, execute at day i's open: the
    overnight gap accrues to yesterday's position, the intraday move to the
    new position. This is the realistic close-to-next-open convention.
- Costs: `fee` plus `spread_bps` and `slippage_bps`, all charged per unit of
  turnover (weight change).
- Cash: the uninvested fraction accrues `risk_free_annual` (T-bill-style
  return on idle cash), compounded per period.
- Weights are clamped to [0, 1]: long-only, unlevered.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .metrics import summarize

DAY_MS = 86_400_000


def candle_date(candle: dict):
    return datetime.fromtimestamp(candle["open_time"] / 1000, tz=timezone.utc).date()


def _open_price(candle: dict, fallback: float) -> float:
    o = candle.get("open")
    if o is None or o <= 0:
        return fallback
    return o


def run_strategy(
    candles: list[dict],
    weight_fn,
    fee: float = 0.001,
    periods_per_year: int = 365,
    start_index: int = 1,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    latency_days: int = 0,
    execution: str = "close",
    risk_free_annual: float = 0.0,
) -> dict:
    """Backtest `weight_fn` over `candles`.

    weight_fn(candles, i) -> target exposure for day i, using data through
    day i-1 only. `latency_days` further delays when that signal takes effect.
    """
    if execution not in ("close", "next_open"):
        raise ValueError("execution must be 'close' or 'next_open'")
    n = len(candles)
    if n < 2:
        raise ValueError("need at least 2 candles")

    cost_rate = fee + (spread_bps + slippage_bps) / 10_000.0
    rf_daily = risk_free_annual / periods_per_year
    equity = [1.0]
    returns = []
    weights = []
    prev_w = 0.0
    closes = [c["close"] for c in candles]
    for i in range(start_index, n):
        sig_i = i - latency_days
        if sig_i >= 1:
            w = min(1.0, max(0.0, weight_fn(candles, sig_i)))
        else:
            w = 0.0
        cost = cost_rate * abs(w - prev_w)
        if execution == "next_open":
            o = _open_price(candles[i], closes[i - 1])
            overnight = prev_w * (o / closes[i - 1] - 1.0)
            intraday = w * (closes[i] / o - 1.0)
            r = overnight + (1.0 - prev_w) * rf_daily + intraday - cost
        else:
            r = w * (closes[i] / closes[i - 1] - 1.0) + (1.0 - w) * rf_daily - cost
        equity.append(equity[-1] * (1.0 + r))
        returns.append(r)
        weights.append(w)
        prev_w = w

    days = (candles[n - 1]["open_time"] - candles[start_index - 1]["open_time"]) / DAY_MS
    stats = summarize(equity, returns, days, periods_per_year, risk_free_annual)
    stats["exposure"] = sum(weights) / len(weights) if weights else 0.0
    stats["turnover"] = sum(abs(weights[i] - (weights[i - 1] if i else 0.0)) for i in range(len(weights)))
    stats["equity"] = equity
    stats["returns"] = returns
    stats["weights"] = weights
    stats["return_days"] = [(candles[i]["open_time"], returns[k]) for k, i in enumerate(range(start_index, n))]
    return stats
