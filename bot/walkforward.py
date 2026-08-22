"""Walk-forward validation: the honest way to pick a strategy.

For each fold:
  1. On the training window, backtest every candidate and pick the one with
     the best risk-adjusted return (Sharpe, after fees).
  2. Apply that choice to the *following* test window it has never seen.
  3. Stitch all test windows together into one out-of-sample track record.

The stitched out-of-sample result is what gets compared against the S&P 500 —
parameters are never evaluated on the data that scores them.
"""
from __future__ import annotations

from bisect import bisect_left

from .engine import DAY_MS, run_strategy
from .metrics import cagr, max_drawdown, sharpe, volatility
from .strategy import default_candidates


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


def walk_forward(
    candles: list[dict],
    candidates: list | None = None,
    train_days: int = 1095,
    test_days: int = 365,
    fee: float = 0.001,
    periods_per_year: int = 365,
) -> dict:
    candidates = candidates if candidates is not None else default_candidates()
    folds = _fold_boundaries(candles, train_days, test_days)
    if not folds:
        raise ValueError("not enough data for one walk-forward fold")

    oos_returns: list[float] = []
    picks = []
    oos_start_idx = folds[0][1]
    oos_end_idx = folds[-1][2]
    for start, train_end, test_end in folds:
        train_slice = candles[start:train_end]
        test_slice = candles[train_end:test_end + 1]

        best = None
        best_sharpe = float("-inf")
        for cand in candidates:
            try:
                tr = run_strategy(train_slice, cand.weight, fee=fee, periods_per_year=periods_per_year)
            except (ValueError, ZeroDivisionError):
                continue
            s = tr["sharpe"]
            if s > best_sharpe:
                best_sharpe = s
                best = cand
        picks.append({"strategy": repr(best), "train_sharpe": best_sharpe})

        te = run_strategy(test_slice, best.weight, fee=fee, periods_per_year=periods_per_year)
        oos_returns.extend(te["returns"])  # fold test windows are contiguous, non-overlapping

    equity = [1.0]
    for r in oos_returns:
        equity.append(equity[-1] * (1.0 + r))
    days = (candles[oos_end_idx]["open_time"] - candles[oos_start_idx]["open_time"]) / DAY_MS
    result = {
        "equity": equity[-1],
        "cagr": cagr(equity, days),
        "vol": volatility(oos_returns, periods_per_year),
        "sharpe": sharpe(oos_returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "folds": picks,
        "n_folds": len(folds),
    }
    return result
