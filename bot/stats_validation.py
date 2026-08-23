"""Statistical validation of the out-of-sample track record.

Implements, with stdlib only:
- Probabilistic Sharpe Ratio (PSR, Bailey & Lopez de Prado)
- Deflated Sharpe Ratio (DSR) using the actual trial count and the observed
  variance of candidate Sharpes from walk-forward selection
- Stationary block bootstrap (Politis-Romano) confidence intervals for
  CAGR / Sharpe / max drawdown, and a drawdown distribution
- White's Reality Check (block-bootstrap max-across-strategies test) over the
  full candidate universe
- Monte Carlo trade-order (return-shuffle) test for path dependence
- Start/end-date sensitivity
- Parameter-stability scoring for the sensitivity grid
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist, mean, stdev

from .metrics import cagr, kurtosis, max_drawdown, sharpe, skewness

EULER_GAMMA = 0.5772156649015329
_ND = NormalDist()


def psr(returns: list[float], periods_per_year: int = 365, sr_benchmark_annual: float = 0.0) -> float:
    """Probabilistic Sharpe: probability the true Sharpe exceeds `sr_benchmark`."""
    n = len(returns)
    if n < 4:
        return 0.5
    sr = sharpe(returns, periods_per_year) / math.sqrt(periods_per_year)  # per-period
    sr_star = sr_benchmark_annual / math.sqrt(periods_per_year)
    g3 = skewness(returns)
    g2 = kurtosis(returns)
    denom = 1.0 - g3 * sr + (g2 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.5
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return _ND.cdf(z)


def expected_max_sharpe_annual(trial_sharpes_annual: list[float], n_trials: int) -> float:
    """Expected maximum Sharpe under the null across `n_trials` independent
    trials (Lopez de Prado's deflation benchmark), in annualized units."""
    if n_trials < 2 or len(trial_sharpes_annual) < 2:
        return 0.0  # no cross-trial dispersion to estimate a max from
    var = stdev(trial_sharpes_annual) ** 2
    if var <= 0:
        return 0.0
    z1 = _ND.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _ND.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def dsr(returns: list[float], trial_sharpes_annual: list[float], n_trials: int, periods_per_year: int = 365) -> float:
    """Deflated Sharpe Ratio: PSR against the multiple-testing-adjusted benchmark."""
    benchmark = expected_max_sharpe_annual(trial_sharpes_annual, n_trials)
    return psr(returns, periods_per_year, sr_benchmark_annual=benchmark)


def stationary_bootstrap_indices(n: int, block: int, rng: random.Random) -> list[int]:
    """Politis-Romano stationary bootstrap: geometric block lengths with mean `block`."""
    out: list[int] = []
    i = rng.randrange(n)
    while len(out) < n:
        out.append(i)
        if rng.random() < 1.0 / block:
            i = rng.randrange(n)
        else:
            i = (i + 1) % n
    return out


def _equity(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def bootstrap_metrics(returns: list[float], n_boot: int = 1000, block: int = 20, seed: int = 42, periods_per_year: int = 365) -> dict:
    """Stationary-bootstrap distributions of CAGR / Sharpe / max drawdown."""
    rng = random.Random(seed)
    cagrs, sharpes, mdds = [], [], []
    n = len(returns)
    days = max(n, 1)
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng)
        res = [returns[i] for i in idx]
        eq = _equity(res)
        cagrs.append(cagr(eq, days))
        sharpes.append(sharpe(res, periods_per_year))
        mdds.append(max_drawdown(eq))
    cagrs.sort()
    sharpes.sort()
    mdds.sort()

    def q(xs, p):
        return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]

    return {
        "cagr_ci": (q(cagrs, 0.05), q(cagrs, 0.95)),
        "sharpe_ci": (q(sharpes, 0.05), q(sharpes, 0.95)),
        "mdd_median": q(mdds, 0.5),
        "mdd_p95": q(mdds, 0.05),  # deep tail: worse than 95% of paths
        "mdd_worst": mdds[0],
    }


def shuffle_test(returns: list[float], n_boot: int = 1000, seed: int = 42, periods_per_year: int = 365) -> dict:
    """Monte Carlo trade-order resampling.

    Compounded CAGR is nearly order-invariant for iid returns, so the
    path-sensitive statistic is max drawdown: we locate the actual drawdown
    in the distribution of shuffled-path drawdowns. A high percentile means
    few random orderings would have been as shallow — the return sequence
    itself (which trend-following trades on) carries risk information.
    """
    rng = random.Random(seed)
    actual_mdd = max_drawdown(_equity(returns))
    mdds = []
    for _ in range(n_boot):
        res = returns[:]
        rng.shuffle(res)
        mdds.append(max_drawdown(_equity(res)))
    mdds.sort()
    protected = sum(1 for x in mdds if x <= actual_mdd) / len(mdds)
    return {
        "actual_mdd": actual_mdd,
        "dd_percentile": protected,
        "shuffled_mdd_median": mdds[len(mdds) // 2],
        "shuffled_mdd_worst": mdds[0],
    }


def reality_check(candidate_returns: list[list[float]], n_boot: int = 200, block: int = 20, seed: int = 42, periods_per_year: int = 365) -> dict:
    """White's Reality Check across strategies.

    Each candidate's daily OOS returns (aligned length) are recentered at
    their own mean (the 'no skill anywhere' null), block-bootstrap-resampled
    with a COMMON index set (preserving cross-correlation), and the max
    Sharpe across candidates forms the null distribution. The p-value is the
    fraction of bootstrap maxima reaching the observed best Sharpe.
    """
    n_cands = len(candidate_returns)
    n = min(len(r) for r in candidate_returns)
    rows = [r[:n] for r in candidate_returns]
    obs = [sharpe(r, periods_per_year) for r in rows]
    best_obs = max(obs)
    centered = [[x - mean(r) for x in r] for r in rows]
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng)
        best_b = max(
            sharpe([row[i] for i in idx], periods_per_year) for row in centered
        )
        if best_b >= best_obs:
            exceed += 1
    return {"best_sharpe": best_obs, "p_value": exceed / n_boot, "n_candidates": n_cands, "n_boot": n_boot}


def start_end_sensitivity(returns: list[float], trims_days=(0, 90, 180), periods_per_year: int = 365) -> list[dict]:
    """Recompute headline metrics with the window's start and/or end trimmed."""
    out = []
    for start in trims_days:
        for end in trims_days:
            r = returns[start : len(returns) - end if end else None]
            if len(r) < 60:
                continue
            eq = _equity(r)
            out.append(
                {
                    "trim_start": start,
                    "trim_end": end,
                    "cagr": cagr(eq, len(r)),
                    "sharpe": sharpe(r, periods_per_year),
                }
            )
    return out


def parameter_stability(grid: dict, metric: str = "sharpe") -> dict:
    """Stability of a {(lookback, target): metrics} grid: dispersion plus
    mean absolute delta between grid-adjacent cells (neighborhood roughness)."""
    values = sorted(m[metric] for m in grid.values())
    lookbacks = sorted({k[0] for k in grid})
    targets = sorted({k[1] for k in grid})
    deltas = []
    for i, lb in enumerate(lookbacks):
        for j, tv in enumerate(targets):
            if (lb, tv) not in grid:
                continue
            if i + 1 < len(lookbacks) and (lookbacks[i + 1], tv) in grid:
                deltas.append(abs(grid[(lb, tv)][metric] - grid[(lookbacks[i + 1], tv)][metric]))
            if j + 1 < len(targets) and (lb, targets[j + 1]) in grid:
                deltas.append(abs(grid[(lb, tv)][metric] - grid[(lb, targets[j + 1])][metric]))
    mid = values[len(values) // 2]
    return {
        "min": values[0],
        "median": mid,
        "max": values[-1],
        "share_above_half_max": sum(1 for v in values if v >= 0.5 * values[-1]) / len(values),
        "mean_neighbor_delta": mean(deltas) if deltas else 0.0,
    }
