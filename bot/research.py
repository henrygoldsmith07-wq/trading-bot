"""Research-methodology extensions (stdlib only).

Everything here operates on plain daily-return lists so it works for a single
asset, a selected walk-forward stream, or a whole portfolio:

- Hansen's SPA test: the studentized sibling of White's Reality Check.
- Expanded bootstrap battery: stationary (Politis-Romano), circular block,
  and moving-block resampling, reported side by side.
- Drawdown confidence intervals: bootstrap distribution of max drawdown
  plus time-under-water statistics.
- Probability of underperformance: paired bootstrap probability that the
  strategy trails a benchmark on CAGR and Sharpe over the same window.
- Bayesian Sharpe estimate: posterior over annualized Sharpe under a Normal
  model with the Jeffreys prior - a credible interval instead of a point.
- Sequence-risk testing: rolling-window outcome distribution (how much
  entry timing matters) versus shuffled orderings.
- Forward Monte Carlo: block-bootstrap future paths preserving serial
  correlation, giving P(loss over horizon) and terminal-wealth bands.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist, mean, stdev

from .metrics import cagr, max_drawdown, sharpe
from .stats_validation import finite_sample_p_value, stationary_bootstrap_indices

_ND = NormalDist()


def _quantile(sorted_xs: list[float], p: float) -> float:
    return sorted_xs[min(len(sorted_xs) - 1, max(0, int(p * len(sorted_xs))))]


def _equity_from(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def circular_block_bootstrap_indices(n: int, block: int, rng: random.Random) -> list[int]:
    """Circular block bootstrap: fixed-length blocks drawn wrap-around."""
    out: list[int] = []
    while len(out) < n:
        start = rng.randrange(n)
        out.extend((start + k) % n for k in range(block))
    return out[:n]


def moving_block_bootstrap_indices(n: int, block: int, rng: random.Random) -> list[int]:
    """Moving-block bootstrap: contiguous blocks from non-wrapping starts."""
    out: list[int] = []
    while len(out) < n:
        start = rng.randrange(max(1, n - block + 1))
        out.extend(min(start + k, n - 1) for k in range(block))
    return out[:n]


_BOOTSTRAPS = {
    "stationary": stationary_bootstrap_indices,
    "circular": circular_block_bootstrap_indices,
    "moving": moving_block_bootstrap_indices,
}

# Keep each bootstrap stream independent without relying on Python's hash().
# String hashes are deliberately randomized between interpreter processes, so
# using hash(name) here would make nominally seeded research irreproducible.
_BOOTSTRAP_SEED_OFFSETS = {
    "stationary": 0,
    "circular": 1,
    "moving": 2,
}


def spa_test(
    candidate_returns: list[list[float]],
    n_boot: int = 200,
    block: int = 20,
    seed: int = 42,
    periods_per_year: int = 365,
) -> dict:
    """Hansen's Superior Predictive Ability test (studentized Reality Check).

    Same null as White's RC ('no candidate beats zero after fees'), but each
    candidate's mean excess return is studentized by its own volatility, so
    one wild strategy no longer dominates the test statistic. Recentered
    stationary-bootstrap p-value for the max studentized statistic, with the
    observed statistic included via the finite-sample add-one correction.
    """
    rows = [r[:] for r in candidate_returns]
    n = min(len(r) for r in rows)
    rows = [r[:n] for r in rows]
    sds = [stdev(r) for r in rows]
    usable = [i for i, sd in enumerate(sds) if sd > 0]
    if not usable:
        return {"best_stat": 0.0, "p_value": 1.0, "n_candidates": len(rows), "n_boot": n_boot}
    obs_stats = [math.sqrt(n) * mean(rows[i]) / sds[i] for i in usable]
    best_obs = max(obs_stats)
    centered = [[x - mean(rows[i]) for x in rows[i]] for i in usable]
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng)
        best_b = max(
            math.sqrt(n) * (sum(centered[k][i] for i in idx) / n) / sds[usable[k]]
            for k in range(len(usable))
        )
        if best_b >= best_obs:
            exceed += 1
    return {
        "best_stat": best_obs,
        "p_value": finite_sample_p_value(exceed, n_boot),
        "n_candidates": len(usable),
        "n_boot": n_boot,
    }


def portfolio_dsr(
    port_returns: list[float],
    trial_sharpes_annual: list[float],
    n_trials: int,
    periods_per_year: int = 365,
) -> float:
    """Deflated Sharpe applied at the PORTFOLIO level.

    Portfolio streams have their own multiple-testing exposure: every
    combiner/overlay variant that was looked at is a trial. Pass the
    annualized Sharpes of the portfolio variants considered (e.g. equal
    weight, inv-vol, tilt, crisis, each overlay ablation) and the count.
    """
    from .stats_validation import dsr

    return dsr(port_returns, trial_sharpes_annual, n_trials, periods_per_year)


def expanded_bootstrap(
    returns: list[float],
    n_boot: int = 1000,
    block: int = 20,
    seed: int = 42,
    periods_per_year: int = 365,
) -> dict:
    """CAGR / Sharpe / max-drawdown CIs under three block-bootstrap schemes.

    Agreement across schemes is itself evidence the interval is not an
    artifact of one resampling assumption; wide disagreement flags strong
    serial dependence or short samples.
    """
    n = len(returns)
    out = {}
    for name, sampler in _BOOTSTRAPS.items():
        rng = random.Random(seed + _BOOTSTRAP_SEED_OFFSETS[name])
        cagrs, sharpes, mdds, tuw = [], [], [], []
        for _ in range(n_boot):
            idx = sampler(n, block, rng)
            res = [returns[i] for i in idx]
            eq = _equity_from(res)
            cagrs.append(cagr(eq, max(n, 1)))
            sharpes.append(sharpe(res, periods_per_year))
            mdds.append(max_drawdown(eq))
            tuw.append(_time_under_water(eq))
        cagrs.sort()
        sharpes.sort()
        mdds.sort()
        tuw.sort()
        out[name] = {
            "cagr_ci": (_quantile(cagrs, 0.05), _quantile(cagrs, 0.95)),
            "sharpe_ci": (_quantile(sharpes, 0.05), _quantile(sharpes, 0.95)),
            "mdd_ci": (_quantile(mdds, 0.05), _quantile(mdds, 0.95)),
            "time_under_water_median": _quantile(tuw, 0.5),
            "time_under_water_p95": _quantile(tuw, 0.95),
        }
    return out


def _time_under_water(equity: list[float]) -> float:
    """Fraction of bars spent below the running peak."""
    peak = equity[0]
    under = 0
    for v in equity[1:]:
        peak = max(peak, v)
        if v < peak:
            under += 1
    return under / max(len(equity) - 1, 1)


def drawdown_confidence_intervals(
    returns: list[float],
    n_boot: int = 1000,
    block: int = 20,
    seed: int = 42,
) -> dict:
    """Confidence bands for future drawdown behavior: max-drawdown percentiles
    plus time-under-water distribution over bootstrap paths."""
    rng = random.Random(seed)
    n = len(returns)
    mdds, tuw = [], []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng)
        eq = _equity_from([returns[i] for i in idx])
        mdds.append(max_drawdown(eq))
        tuw.append(_time_under_water(eq))
    mdds.sort()
    tuw.sort()
    return {
        "actual_mdd": max_drawdown(_equity_from(returns)),
        "mdd_90_ci": (_quantile(mdds, 0.05), _quantile(mdds, 0.95)),
        "mdd_median": _quantile(mdds, 0.5),
        "mdd_worst": mdds[0],
        "time_under_water_median": _quantile(tuw, 0.5),
        "time_under_water_p95": _quantile(tuw, 0.95),
    }


def probability_of_underperformance(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    days: list[int] | None = None,
    n_boot: int = 1000,
    block: int = 20,
    seed: int = 42,
    periods_per_year: int = 365,
) -> dict:
    """Paired-bootstrap probability that the strategy trails the benchmark.

    Both series are resampled with a COMMON index set on aligned days, so the
    strategy's diversification/correlation against the benchmark is kept.
    Reported for CAGR and excess Sharpe separately, plus the analytic
    normal-approximation probability for the Sharpe difference.
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("strategy and benchmark returns must be pre-aligned to the same days")
    n = len(strategy_returns)
    if n < 30:
        raise ValueError("need at least 30 aligned days")
    obs_cagr_s = cagr(_equity_from(strategy_returns), n)
    obs_cagr_b = cagr(_equity_from(benchmark_returns), n)
    obs_sh_s = sharpe(strategy_returns, periods_per_year)
    obs_sh_b = sharpe(benchmark_returns, periods_per_year)
    rng = random.Random(seed)
    worse_cagr = worse_sharpe = 0
    diffs = []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng)
        rs = [strategy_returns[i] for i in idx]
        rb = [benchmark_returns[i] for i in idx]
        d_cagr = cagr(_equity_from(rs), n) - cagr(_equity_from(rb), n)
        d_sh = sharpe(rs, periods_per_year) - sharpe(rb, periods_per_year)
        diffs.append(d_sh)
        if d_cagr < 0:
            worse_cagr += 1
        if d_sh < 0:
            worse_sharpe += 1
    diffs.sort()
    sd_diff = stdev(diffs) if len(diffs) > 1 else 0.0
    z = (obs_sh_s - obs_sh_b) / sd_diff if sd_diff > 0 else 0.0
    return {
        "p_underperform_cagr": worse_cagr / n_boot,
        "p_underperform_sharpe": worse_sharpe / n_boot,
        "observed_cagr_gap": obs_cagr_s - obs_cagr_b,
        "observed_sharpe_gap": obs_sh_s - obs_sh_b,
        "sharpe_gap_ci": (_quantile(diffs, 0.05), _quantile(diffs, 0.95)),
        "analytic_p_underperform_sharpe": _ND.cdf(z),
    }


def bayesian_sharpe(
    returns: list[float],
    periods_per_year: int = 365,
    draws: int = 10_000,
    seed: int = 42,
    benchmark_annual_sharpe: float = 0.0,
) -> dict:
    """Posterior over annualized Sharpe, Normal model with Jeffreys prior.

    sigma^2 ~ scaled-inv-chi2(nu=n-1, s^2); mu | sigma^2 ~ N(xbar, sigma^2/n).
    Returns credible intervals and P(Sharpe > benchmark) as posterior
    probabilities rather than frequentist accept/reject.
    """
    import random as _random

    n = len(returns)
    if n < 3:
        raise ValueError("need at least 3 returns")
    xbar = mean(returns)
    s2 = stdev(returns) ** 2
    rng = _random.Random(seed)
    samples = []
    for _ in range(draws):
        g = rng.gammavariate((n - 1) / 2.0, 2.0)  # chi-square(n-1)/... via gamma
        sigma2 = (n - 1) * s2 / g if g > 0 else float("inf")
        mu = rng.gauss(xbar, math.sqrt(sigma2 / n)) if sigma2 > 0 and math.isfinite(sigma2) else xbar
        sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0
        samples.append(mu / sigma * math.sqrt(periods_per_year) if sigma > 0 else 0.0)
    samples.sort()
    above = sum(1 for s in samples if s > benchmark_annual_sharpe)
    return {
        "posterior_mean": mean(samples),
        "ci_90": (_quantile(samples, 0.05), _quantile(samples, 0.95)),
        "median": _quantile(samples, 0.5),
        "prob_above_benchmark": above / draws,
        "n": n,
        "draws": draws,
    }


def sequence_risk(
    returns: list[float],
    horizon_days: int = 252,
    n_shuffles: int = 200,
    seed: int = 42,
) -> dict:
    """How much does ORDER matter? Rolling h-day outcomes over the observed
    sequence vs the same returns in shuffled order.

    If shuffled sequences lose money as often as the real one did, timing was
    luck of entry dates within an iid process; if the real sequence is safer
    (or riskier) than shuffled ones, the ordering itself carries information.
    """
    if len(returns) < horizon_days:
        raise ValueError("series must be at least one horizon long")
    rng = random.Random(seed)
    windows = [
        _window_return(returns[i : i + horizon_days])
        for i in range(0, max(len(returns) - horizon_days + 1, 1))
    ]
    windows = [w for w in windows if w is not None]
    if not windows:
        raise ValueError("series shorter than one horizon")

    def stats(xs):
        xs_sorted = sorted(xs)
        losses = sum(1 for x in xs if x < 0) / len(xs)
        return {
            "p_loss": losses,
            "worst": xs_sorted[0],
            "p05": _quantile(xs_sorted, 0.05),
            "median": _quantile(xs_sorted, 0.5),
            "best": xs_sorted[-1],
        }

    shuffle_loss_probs: list[float] = []
    for _ in range(n_shuffles):
        res = returns[:]
        rng.shuffle(res)
        sh_windows: list[float] = []
        for i in range(0, max(len(res) - horizon_days + 1, 1)):
            wr = _window_return(res[i : i + horizon_days])
            if wr is not None:
                sh_windows.append(wr)
        if sh_windows:
            shuffle_loss_probs.append(sum(1 for w in sh_windows if w < 0) / len(sh_windows))
    observed = stats(windows)
    return {
        **{f"observed_{k}": v for k, v in observed.items()},
        "shuffle_mean_p_loss": mean(shuffle_loss_probs) if shuffle_loss_probs else observed["p_loss"],
        "sequence_risk_gap": observed["p_loss"] - (mean(shuffle_loss_probs) if shuffle_loss_probs else observed["p_loss"]),
        "horizon_days": horizon_days,
        "n_windows": len(windows),
    }


def _window_return(chunk: list[float]) -> float | None:
    if not chunk:
        return None
    eq = 1.0
    for r in chunk:
        eq *= 1.0 + r
    return eq - 1.0


def mc_future_paths(
    returns: list[float],
    horizon_days: int = 252,
    n_paths: int = 5000,
    block: int = 20,
    seed: int = 42,
) -> dict:
    """Forward Monte Carlo that PRESERVES serial correlation.

    Future paths are drawn as stationary-block resamples of history (not iid
    shuffles): volatility clustering and trend persistence survive, so
    drawdown-risk estimates stay honest for trend-following streams.
    """
    rng = random.Random(seed)
    n = len(returns)
    terminals = []
    worst_dds = []
    for _ in range(n_paths):
        idx = stationary_bootstrap_indices(horizon_days, block, rng)
        path = [returns[i % n] for i in idx]
        eq = _equity_from(path)
        terminals.append(eq[-1])
        worst_dds.append(max_drawdown(eq))
    terminals.sort()
    worst_dds.sort()
    return {
        "horizon_days": horizon_days,
        "p_loss": sum(1 for t in terminals if t < 1.0) / n_paths,
        "terminal_p05": _quantile(terminals, 0.05),
        "terminal_median": _quantile(terminals, 0.5),
        "terminal_p95": _quantile(terminals, 0.95),
        "path_mdd_median": _quantile(worst_dds, 0.5),
        "path_mdd_p95": _quantile(worst_dds, 0.05),
        "path_mdd_worst": worst_dds[0],
    }
