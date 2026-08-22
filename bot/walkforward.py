"""Walk-forward validation: the honest way to pick a strategy.

For each fold:
  1. On the training window, backtest every candidate and pick the one with
     the best risk-adjusted return (Sharpe, after fees).
  2. Apply that choice to the *following* test window it has never seen.
  3. Stitch all test windows together into one out-of-sample track record.

Fold boundaries are expressed in absolute open_time milliseconds so several
assets can be walked forward on the same calendar and combined into a
portfolio without date misalignment.
"""
from __future__ import annotations

from bisect import bisect_left

from .engine import DAY_MS, run_strategy
from .metrics import cagr, max_drawdown, sharpe, volatility
from .strategy import build_candidates


def _fold_boundaries(candles: list[dict], train_days: int, test_days: int) -> list[tuple[int, int, int]]:
    """Expanding-window folds: train [0, train_end), test [train_end, test_end).

    Each fold's training window grows by one test period, and every test
    window is used exactly once — the most recent data is never skipped.
    """
    times = [c["open_time"] for c in candles]
    n = len(candles)
    folds = []
    epoch = times[0]
    while True:
        train_end = bisect_left(times, epoch + train_days * DAY_MS)
        if train_end >= n:
            break
        test_end = bisect_left(times, times[train_end] + test_days * DAY_MS)
        if test_end >= n:
            break
        folds.append((0, train_end, test_end))
        epoch += test_days * DAY_MS  # next fold's training window grows by one test period
    return folds


def absolute_folds(candles: list[dict], train_days: int, test_days: int) -> list[tuple[int, int]]:
    """Fold boundaries as (train_end_time, test_end_time) in epoch ms."""
    return [(candles[te]["open_time"], candles[ts]["open_time"]) for _, te, ts in _fold_boundaries(candles, train_days, test_days)]


def walk_forward_at(
    candles: list[dict],
    abs_folds: list[tuple[int, int]],
    candidates: list | None = None,
    fee: float = 0.001,
    periods_per_year: int = 365,
) -> dict:
    """Walk-forward using absolute fold boundaries.

    Returns per-day out-of-sample returns keyed by the day's open_time, the
    per-fold strategy picks, and summary statistics.
    """
    candidates = candidates if candidates is not None else build_candidates()
    times = [c["open_time"] for c in candles]
    n = len(candles)
    if not abs_folds:
        raise ValueError("no folds supplied")

    daily: dict[int, float] = {}
    picks = []
    for train_end_time, test_end_time in abs_folds:
        train_end = bisect_left(times, train_end_time)
        test_end = bisect_left(times, test_end_time)
        if train_end >= n or train_end < 2 or test_end <= train_end:
            continue
        train_slice = candles[:train_end]
        test_slice = candles[train_end:test_end + 1]

        best = None
        best_sharpe = float("-inf")
        for cand in candidates:
            try:
                tr = run_strategy(train_slice, cand.weight_at, fee=fee, periods_per_year=periods_per_year)
            except (ValueError, ZeroDivisionError):
                continue
            s = tr["sharpe"]
            if s > best_sharpe:
                best_sharpe = s
                best = cand
        if best is None:
            continue
        picks.append({"strategy": repr(best), "train_sharpe": best_sharpe})

        te = run_strategy(test_slice, best.weight_at, fee=fee, periods_per_year=periods_per_year)
        for t, r in te["return_days"]:
            daily[t] = r

    if not daily:
        raise ValueError("no out-of-sample days produced (insufficient history)")
    days_sorted = sorted(daily)
    returns = [daily[t] for t in days_sorted]
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    span_days = (days_sorted[-1] - days_sorted[0]) / DAY_MS + 1
    result = {
        "equity": equity[-1],
        "cagr": cagr(equity, span_days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "folds": picks,
        "n_folds": len(picks),
        "daily": daily,
        "first_day": days_sorted[0],
        "last_day": days_sorted[-1],
    }
    return result


def walk_forward(
    candles: list[dict],
    candidates: list | None = None,
    train_days: int = 1095,
    test_days: int = 365,
    fee: float = 0.001,
    periods_per_year: int = 365,
) -> dict:
    folds = _fold_boundaries(candles, train_days, test_days)
    if not folds:
        raise ValueError("not enough data for one walk-forward fold")
    return walk_forward_at(
        candles,
        [(candles[te]["open_time"], candles[ts]["open_time"]) for _, te, ts in folds],
        candidates=candidates,
        fee=fee,
        periods_per_year=periods_per_year,
    )
