"""Portfolio performance metrics.

All functions take simple sequences so they work for any asset/strategy:
- equity: list of portfolio values (equity[0] is the starting value)
- returns: list of simple period returns (e.g. daily)
- days: calendar days elapsed (CAGR uses actual calendar time so assets with
  different trading calendars, e.g. crypto 365d vs equities 252d, compare fairly)
"""
from __future__ import annotations

import math
from statistics import mean, stdev


def cagr(equity: list[float], days: float) -> float:
    if days <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (365.0 / days) - 1.0


def volatility(returns: list[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    return stdev(returns) * math.sqrt(periods_per_year)


def sharpe(returns: list[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    sd = stdev(returns)
    if sd == 0:
        return 0.0
    return mean(returns) / sd * math.sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    """Most negative peak-to-trough decline, as a fraction (e.g. -0.5 = -50%)."""
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return mdd


def summarize(equity: list[float], returns: list[float], days: float, periods_per_year: int) -> dict:
    return {
        "final": equity[-1],
        "cagr": cagr(equity, days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
    }
