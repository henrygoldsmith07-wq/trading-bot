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
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    latency_days: int = 0,
    execution: str = "close",
    risk_free_annual: float = 0.0,
    embargo_days: int = 0,
    selection_fn=None,
) -> dict:
    """Walk-forward using absolute fold boundaries.

    `embargo_days` trims the tail of each training window used for strategy
    selection, so no information within `embargo_days` of the test window can
    influence the pick. `selection_fn(candidates, train_slice, engine_kwargs)`
    may override the default pick-by-train-Sharpe rule (used for nested
    walk-forward). Trial Sharpes of every candidate are recorded for
    multiple-testing corrections (DSR / Reality Check).

    Returns per-day out-of-sample returns keyed by the day's open_time, the
    per-fold strategy picks, and summary statistics.
    """
    candidates = candidates if candidates is not None else build_candidates()
    times = [c["open_time"] for c in candles]
    n = len(candles)
    if not abs_folds:
        raise ValueError("no folds supplied")

    engine_kwargs = dict(
        fee=fee,
        periods_per_year=periods_per_year,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        latency_days=latency_days,
        execution=execution,
        risk_free_annual=risk_free_annual,
    )

    def _run(slice_, cand):
        return run_strategy(slice_, cand.weight_at, **engine_kwargs)

    daily: dict[int, float] = {}
    picks = []
    trial_sharpes: list[float] = []
    exposures = []
    turnovers = []
    day_weights = []
    for train_end_time, test_end_time in abs_folds:
        train_end = bisect_left(times, train_end_time)
        test_end = bisect_left(times, test_end_time)
        if train_end >= n or train_end < 2 or test_end <= train_end:
            continue
        selection_slice = candles[: max(2, train_end - embargo_days)]
        test_slice = candles[train_end:test_end + 1]

        if selection_fn is not None:
            best = selection_fn(candidates, selection_slice, engine_kwargs)
            best_sharpe = float("nan")
        else:
            best = None
            best_sharpe = float("-inf")
            for cand in candidates:
                try:
                    tr = _run(selection_slice, cand)
                except (ValueError, ZeroDivisionError):
                    continue
                s = tr["sharpe"]
                trial_sharpes.append(s)
                if s > best_sharpe:
                    best_sharpe = s
                    best = cand
        if best is None:
            continue
        picks.append({"strategy": repr(best), "train_sharpe": best_sharpe})

        te = _run(test_slice, best)
        for t, r in te["return_days"]:
            daily[t] = r
        exposures.append(te["exposure"])
        turnovers.append(te["turnover"])
        day_weights.append(len(te["returns"]))

    if not daily:
        raise ValueError("no out-of-sample days produced (insufficient history)")
    days_sorted = sorted(daily)
    returns = [daily[t] for t in days_sorted]
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    span_days = (days_sorted[-1] - days_sorted[0]) / DAY_MS + 1
    total_days = sum(day_weights) or 1
    result = {
        "equity": equity[-1],
        "cagr": cagr(equity, span_days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "max_drawdown": max_drawdown(equity),
        "folds": picks,
        "n_folds": len(picks),
        "daily": daily,
        "first_day": days_sorted[0],
        "last_day": days_sorted[-1],
        "trial_sharpes": trial_sharpes,
        "exposure": sum(e * w for e, w in zip(exposures, day_weights)) / total_days,
        "turnover": sum(t * w for t, w in zip(turnovers, day_weights)) / total_days,
    }
    return result


def _purged_inner_folds(candles: list[dict], train_days: int, test_days: int, purge_days: int) -> list[tuple[int, int, int]]:
    """Inner (selection) folds with a purge gap of `purge_days` between the
    end of training and the start of testing, so indicators computed on
    training data cannot reach into the evaluation window."""
    times = [c["open_time"] for c in candles]
    n = len(candles)
    folds = []
    epoch = times[0]
    while True:
        train_end = bisect_left(times, epoch + train_days * DAY_MS)
        if train_end >= n:
            break
        test_start = bisect_left(times, times[train_end] + purge_days * DAY_MS)
        if test_start >= n:
            break
        test_end = bisect_left(times, times[test_start] + test_days * DAY_MS)
        if test_end - test_start < 30:
            break
        folds.append((0, train_end, test_start, test_end))
        epoch += test_days * DAY_MS
    return folds


def nested_selection_fn(inner_train_days: int = 365, inner_test_days: int = 182, purge_days: int = 220, embargo_days: int = 30):
    """Build a selection function that picks candidates by *inner* walk-forward
    performance on the training window (nested walk-forward selection)."""

    def select(candidates, train_slice, engine_kwargs):
        inner_folds = _purged_inner_folds(train_slice, inner_train_days, inner_test_days, purge_days)
        best = None
        best_score = float("-inf")
        for cand in candidates:
            scores = []
            for _, train_end, test_start, test_end in inner_folds:
                sel = train_slice[: max(2, train_end - embargo_days)]
                test = train_slice[test_start:test_end + 1]
                if len(test) < 30:
                    continue
                try:
                    tr = run_strategy(test, cand.weight_at, **engine_kwargs)
                except (ValueError, ZeroDivisionError):
                    continue
                scores.append(tr["sharpe"])
            if not scores:
                continue
            score = sum(scores) / len(scores)
            if score > best_score:
                best_score = score
                best = cand
        return best

    return select


def fixed_candidate_streams(candles: list[dict], abs_folds: list[tuple[int, int]], candidates: list, **engine_kwargs) -> dict[str, dict[int, float]]:
    """OOS daily return streams for every candidate trading ALL folds with no
    selection — the raw material for White's Reality Check."""
    times = [c["open_time"] for c in candles]
    n = len(candles)
    streams: dict[str, dict[int, float]] = {}
    for train_end_time, test_end_time in abs_folds:
        train_end = bisect_left(times, train_end_time)
        test_end = bisect_left(times, test_end_time)
        if train_end >= n or test_end <= train_end:
            continue
        test_slice = candles[train_end:test_end + 1]
        for cand in candidates:
            try:
                te = run_strategy(test_slice, cand.weight_at, **engine_kwargs)
            except (ValueError, ZeroDivisionError):
                continue
            stream = streams.setdefault(repr(cand), {})
            for t, r in te["return_days"]:
                stream[t] = r
    return streams


def combine_portfolio(asset_dailies: dict[str, dict[int, float]], timeline: list[int], n_assets: int) -> list[float]:
    """Equal-weight portfolio returns on the shared timeline.

    The denominator is the *selected* asset count, not the count with data on
    a given day: an asset that is late-listed, stale, or delisted simply sits
    in cash for its missing days. This is a survivorship-bias control — a
    vanished asset can not silently hand its capital to the survivors.
    """
    dailies = list(asset_dailies.values())
    out = []
    for t in timeline:
        total = 0.0
        for daily in dailies:
            total += daily.get(t, 0.0)
        out.append(total / n_assets)
    return out


def combine_portfolio_invvol(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    window: int = 20,
    max_multiple_of_equal: float = 2.0,
) -> list[float]:
    """Inverse-volatility-weighted portfolio returns.

    Same survivorship control as the equal-weight combiner: the day's total
    exposure is (assets with data) / (selected assets) — a missing asset's
    sleeve sits in cash rather than being redistributed. Within the assets
    that do trade, weights are proportional to 1/trailing-vol (computed from
    strictly past returns), capped at `max_multiple_of_equal` x the equal
    weight so a single low-vol asset (e.g. a bond ETF) cannot dominate.
    """
    import math

    syms = list(asset_dailies)
    hist: dict[str, list[float]] = {s: [] for s in syms}
    out = []
    for t in timeline:
        present = [s for s in syms if t in asset_dailies[s]]
        if not present:
            out.append(0.0)
            continue
        if any(len(hist[s]) >= window for s in present):
            raw = {}
            for s in present:
                h = hist[s][-window:]
                if len(h) < 2:
                    raw[s] = 1.0  # no usable history yet: neutral weight
                    continue
                m = sum(h) / len(h)
                var = sum((x - m) ** 2 for x in h) / (len(h) - 1)
                vol = math.sqrt(max(var, 0.0) * 365)
                raw[s] = 1.0 / max(vol, 1e-6)
            cap = max_multiple_of_equal / len(present)
            total_raw = sum(raw.values())
            weights = {s: min(cap, raw[s] / total_raw) for s in present}
            # renormalize once after capping (capped assets give back the excess)
            free = [s for s in present if weights[s] < cap]
            slack = 1.0 - sum(weights.values())
            free_total = sum(raw[s] for s in free)
            if free and free_total > 0 and slack > 0:
                for s in free:
                    weights[s] += slack * raw[s] / free_total
        else:
            eq = 1.0 / len(present)
            weights = {s: eq for s in present}
        gross = sum(weights[s] * asset_dailies[s][t] for s in present)
        out.append(gross * len(present) / n_assets)
        for s in present:
            hist[s].append(asset_dailies[s][t])
    return out


def walk_forward(
    candles: list[dict],
    candidates: list | None = None,
    train_days: int = 1095,
    test_days: int = 365,
    fee: float = 0.001,
    periods_per_year: int = 365,
    **kwargs,
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
        **kwargs,
    )
