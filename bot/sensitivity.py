"""Backtesting-quality sensitivity analysis.

Answers four robustness questions about the walk-forward result:
1. Parameter sensitivity — does performance collapse when the winning
   strategy's parameters move, or is it a stable plateau?
2. Rolling parameter sensitivity — is performance consistent across
   sub-periods, or earned in one lucky window?
3. Transaction-cost sensitivity — how much friction can the bot absorb
   (fees + spread + slippage combined) before it stops beating the benchmark?
4. Latency and execution realism — what does acting one or two days late,
   or executing at the next open instead of the close, cost?
"""
from __future__ import annotations

from .engine import run_strategy
from .metrics import cagr, max_drawdown, sharpe
from .strategy import TrendVol
from .walkforward import absolute_folds, walk_forward_at


def _summarize_returns(returns: list[float], periods_per_year: int = 365) -> dict:
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = max(len(returns), 1)
    return {
        "cagr": cagr(equity, days),
        "sharpe": sharpe(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "final": equity[-1],
    }


def parameter_grid(candles: list[dict], folds: list[tuple[int, int]], lookbacks=(25, 50, 75, 100, 150, 200), targets=(0.20, 0.30, 0.40), **engine_kwargs) -> dict:
    """Walk-forward OOS metrics for each TrendVol parameter combination."""
    grid = {}
    for lb in lookbacks:
        for tv in targets:
            wf = walk_forward_at(candles, folds, candidates=[TrendVol(lb, 20, tv)], **engine_kwargs)
            days = sorted(wf["daily"])
            grid[(lb, tv)] = _summarize_returns([wf["daily"][t] for t in days])
    return grid


def rolling_blocks(candles: list[dict], folds: list[tuple[int, int]], block_days: int = 730, **engine_kwargs) -> list[dict]:
    """Split the OOS timeline into consecutive blocks and report per-block stats.

    Uses the walk-forward winner per fold throughout, so within-block results
    remain strictly out-of-sample.
    """
    from .strategy import build_candidates

    wf = walk_forward_at(candles, folds, candidates=build_candidates(), **engine_kwargs)
    days = sorted(wf["daily"])
    blocks = []
    start = 0
    while start < len(days):
        end = min(start + block_days, len(days))
        rets = [wf["daily"][t] for t in days[start:end]]
        blocks.append(
            {
                "start": days[start],
                "end": days[end - 1],
                "metrics": _summarize_returns(rets),
            }
        )
        start = end
    return blocks


def cost_sweep(candles: list[dict], folds: list[tuple[int, int]], total_bps=(5, 10, 20, 50, 100), **engine_kwargs) -> dict:
    """Walk-forward OOS at combined friction levels (fee + spread + slippage)."""
    out = {}
    for bps in total_bps:
        kw = dict(engine_kwargs)
        kw.update(fee=bps / 10_000.0, spread_bps=0.0, slippage_bps=0.0)
        wf = walk_forward_at(candles, folds, **kw)
        days = sorted(wf["daily"])
        out[bps] = _summarize_returns([wf["daily"][t] for t in days])
    return out


def latency_sweep(candles: list[dict], folds: list[tuple[int, int]], latencies=(0, 1, 2), **engine_kwargs) -> dict:
    out = {}
    for lat in latencies:
        kw = dict(engine_kwargs)
        kw.update(latency_days=lat)
        wf = walk_forward_at(candles, folds, **kw)
        days = sorted(wf["daily"])
        out[lat] = _summarize_returns([wf["daily"][t] for t in days])
    return out


def execution_comparison(candles: list[dict], folds: list[tuple[int, int]], **engine_kwargs) -> dict:
    out = {}
    for mode in ("close", "next_open"):
        kw = dict(engine_kwargs)
        kw.update(execution=mode)
        wf = walk_forward_at(candles, folds, **kw)
        days = sorted(wf["daily"])
        out[mode] = _summarize_returns([wf["daily"][t] for t in days])
    return out


def full_sensitivity(candles: list[dict], train_days: int = 1095, test_days: int = 365, **engine_kwargs) -> dict:
    folds = absolute_folds(candles, train_days, test_days)
    return {
        "folds": folds,
        "parameters": parameter_grid(candles, folds, **engine_kwargs),
        "rolling": rolling_blocks(candles, folds, **engine_kwargs),
        "costs": cost_sweep(candles, folds, **engine_kwargs),
        "latency": latency_sweep(candles, folds, **engine_kwargs),
        "execution": execution_comparison(candles, folds, **engine_kwargs),
    }
