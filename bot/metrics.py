"""Portfolio performance metrics.

All functions take simple sequences so they work for any asset/strategy:
- equity: list of portfolio values (equity[0] is the starting value)
- returns: list of simple period returns (e.g. daily)
- days: calendar days elapsed (CAGR uses actual calendar time so assets with
  different trading calendars, e.g. crypto 365d vs equities 252d, compare fairly)

Sharpe ratios are excess-return: (mean return - risk-free accrual) / stdev.
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


def sharpe(returns: list[float], periods_per_year: int, risk_free_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    sd = stdev(returns)
    if sd == 0:
        return 0.0
    excess = mean(returns) - risk_free_annual / periods_per_year
    return excess / sd * math.sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    """Most negative peak-to-trough decline, as a fraction (e.g. -0.5 = -50%)."""
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return mdd


def downside_deviation(returns: list[float], periods_per_year: int, risk_free_annual: float = 0.0) -> float:
    """Annualized downside deviation: RMS of returns below the cash rate."""
    if not returns:
        return 0.0
    rf_daily = risk_free_annual / periods_per_year
    shortfall = [min(r - rf_daily, 0.0) for r in returns]
    return math.sqrt(sum(x * x for x in shortfall) / len(shortfall)) * math.sqrt(periods_per_year)


def sortino(returns: list[float], periods_per_year: int, risk_free_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    dd = downside_deviation(returns, periods_per_year, risk_free_annual)
    if dd == 0:
        return 0.0
    excess_annual = (mean(returns) - risk_free_annual / periods_per_year) * periods_per_year
    return excess_annual / dd


def calmar(cagr_value: float, max_drawdown_value: float) -> float:
    if max_drawdown_value >= 0:
        return 0.0
    return cagr_value / abs(max_drawdown_value)


def var_hist(returns: list[float], alpha: float = 0.95) -> float:
    """Historical Value-at-Risk (descriptive only, not a risk limit)."""
    if not returns:
        return 0.0
    ordered = sorted(returns)
    idx = max(0, min(len(ordered) - 1, int((1 - alpha) * len(ordered))))
    return ordered[idx]


def expected_shortfall(returns: list[float], alpha: float = 0.95) -> float:
    """Average loss in the worst (1-alpha) fraction of days."""
    if not returns:
        return 0.0
    ordered = sorted(returns)
    k = max(1, int((1 - alpha) * len(ordered)))
    return mean(ordered[:k])


def skewness(returns: list[float]) -> float:
    if len(returns) < 3:
        return 0.0
    m = mean(returns)
    m2 = sum((x - m) ** 2 for x in returns) / len(returns)
    if m2 == 0:
        return 0.0
    m3 = sum((x - m) ** 3 for x in returns) / len(returns)
    return m3 / m2 ** 1.5


def kurtosis(returns: list[float]) -> float:
    """Raw kurtosis (normal = 3), not excess."""
    if len(returns) < 4:
        return 3.0
    m = mean(returns)
    m2 = sum((x - m) ** 2 for x in returns) / len(returns)
    if m2 == 0:
        return 3.0
    m4 = sum((x - m) ** 4 for x in returns) / len(returns)
    return m4 / m2 ** 2


def trade_stats(weights: list[float], returns: list[float], entry_epsilon: float = 1e-9) -> dict:
    """Per-trade accounting from a weight/return stream.

    A 'trade' is any bar where exposure changed (|Δw| > epsilon). Hit rate,
    average win/loss and profit factor are computed over those event bars;
    they describe the strategy's day-level edge around its own turnover, and
    are reported alongside — never instead of — Sharpe/drawdown."""
    if len(weights) != len(returns):
        raise ValueError("weights and returns must align")
    trades = []
    prev = 0.0
    for w, r in zip(weights, returns):
        if abs(w - prev) > entry_epsilon:
            trades.append(r)
        prev = w
    n = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n_trades": n,
        "hit_rate": round(len(wins) / n, 4) if n else None,
        "avg_win": round(sum(wins) / len(wins), 6) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
    }


def summarize(equity: list[float], returns: list[float], days: float, periods_per_year: int, risk_free_annual: float = 0.0) -> dict:
    return {
        "final": equity[-1],
        "cagr": cagr(equity, days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "max_drawdown": max_drawdown(equity),
    }
