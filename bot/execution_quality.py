"""Execution quality: three different things that used to be one vague "slippage".

WHY THIS EXISTS
    The old single slippage number merged three unrelated quantities:
    modelled cost (what the cost model PREDICTS), the observed simulated
    execution gap (what the paper runner actually measured between decision
    and execution), and data-quality error (outages, skipped fills, stale
    feeds, lifecycle failures). Merging them made the figure uninterpretable
    and made a feed outage look like a trading cost.

    This module keeps them apart structurally: `separation` carries the three
    blocks and no code path sums them. Predicted cost is NEVER guessed — when
    no cost model was supplied it is None and the report says so.

Paper trading only. These are simulated fills; nothing here touches an
exchange or implies real execution.
"""
from __future__ import annotations

from typing import Any

from .lifecycle import audit_log_lifecycle

# A day counts as a turnover event when its one-way book turnover exceeds this.
# Small numbers are drift inside the rebalance band, not trades.
TURNOVER_EVENT_EPSILON = 0.01

SLIPPAGE_BUCKETS = ("0-1bps", "1-5bps", "5-10bps", "10-25bps", "25bps+", "unknown")
_BUCKET_LIMITS = ((1.0, "0-1bps"), (5.0, "1-5bps"), (10.0, "5-10bps"), (25.0, "10-25bps"))


def _bucket(value: float | None) -> str:
    """Bucket by magnitude with left-closed intervals: [0,1) [1,5) [5,10) [10,25) [25,inf).

    A boundary value belongs to the HIGHER bucket (1.0 is '1-5bps'), so the
    test can pin each edge without ambiguity.
    """
    if value is None:
        return "unknown"
    magnitude = abs(value)
    for limit, label in _BUCKET_LIMITS:
        if magnitude < limit:
            return label
    return "25bps+"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def diagnostics(
    forward_entries: list[dict[str, Any]],
    *,
    execution_model: str,
    cost_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turnover, measured execution gap, and the data-quality errors, separated.

    `forward_entries` are the same rows the forward tape records: per-day
    portfolio state with per-asset weights and observed slippage. `cost_model`
    is the predicted-cost block when one was computed; its numbers pass
    through verbatim and are never re-derived here.
    """
    notes: list[str] = []

    ordered = sorted(forward_entries, key=lambda e: str(e.get("date") or ""))
    total_turnover = 0.0
    per_day_turnover: list[float] = []
    turnover_events = 0
    gap_values: list[float] = []
    bucket_counts: dict[str, int] = dict.fromkeys(SLIPPAGE_BUCKETS, 0)
    skipped_fills = 0
    stale_data_events = 0
    orders: list[dict[str, Any]] = []
    delayed_signals = 0

    prev_weights: dict[str, float] = {}
    for entry in ordered:
        assets = entry.get("assets") or {}
        weights: dict[str, float] = {}
        for sym, block in assets.items():
            if not isinstance(block, dict):
                continue
            w = block.get("weight")
            if isinstance(w, (int, float)) and not isinstance(w, bool):
                weights[sym] = float(w)
            slip = block.get("slippage_bps")
            if isinstance(slip, (int, float)) and not isinstance(slip, bool):
                gap_values.append(float(slip))
                bucket_counts[_bucket(float(slip))] += 1
            elif slip is None:
                bucket_counts["unknown"] += 1

        day_turnover = sum(
            abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            for s in set(weights) | set(prev_weights)
        ) / 2.0
        per_day_turnover.append(day_turnover)
        total_turnover += day_turnover
        if day_turnover > TURNOVER_EVENT_EPSILON:
            turnover_events += 1
        prev_weights = weights

        skipped_fills += len(entry.get("missed_fills") or [])
        stale_data_events += len(entry.get("stale_data") or [])
        orders.extend(o for o in (entry.get("orders") or []) if isinstance(o, dict))

    # `audit_log_lifecycle` consumes forward-log ENTRIES (it reads each
    # entry's date as the session and its `orders` list), so pass the rows
    # themselves — flattened order dicts would silently audit zero orders.
    lifecycle = audit_log_lifecycle(ordered) if orders else {"n_orders_flagged": 0}
    raw_count = lifecycle.get("n_orders_flagged", 0)
    lifecycle_count = int(raw_count) if isinstance(raw_count, (int, float)) else 0

    # Delayed signals are only knowable when orders record both decision and
    # effect times. Without them the count is unknown-as-zero, stated in notes
    # rather than guessed.
    if orders and all("decision_at" in o and "executed_at" in o for o in orders):
        from .lifecycle import parse_lifecycle_ts

        for o in orders:
            d = parse_lifecycle_ts(o.get("decision_at"))
            x = parse_lifecycle_ts(o.get("executed_at"))
            if d is not None and x is not None and x > d:
                delayed_signals += 1
    elif orders:
        notes.append("order rows lack decision/execution timestamps; delayed_signals reported as 0, not measured")

    observed = {
        "mean_abs_bps": (sum(abs(v) for v in gap_values) / len(gap_values)) if gap_values else None,
        "p50_bps": _percentile(gap_values, 0.5),
        "p90_bps": _percentile(gap_values, 0.9),
        "max_bps": max(gap_values, key=abs) if gap_values else None,
        "observations": len(gap_values),
    }
    predicted = None
    if cost_model is not None:
        predicted = cost_model.get("predicted_cost_bps")

    n_days = len(ordered)
    return {
        "execution_model": execution_model,
        "turnover": total_turnover,
        "turnover_per_day": (total_turnover / n_days) if n_days else 0.0,
        "turnover_events": turnover_events,
        "predicted_cost_bps": predicted,
        "observed_execution_gap": observed,
        "slippage_distribution": [{"bucket": b, "count": bucket_counts[b]} for b in SLIPPAGE_BUCKETS],
        "skipped_fills": skipped_fills,
        "delayed_signals": delayed_signals,
        "stale_data_events": stale_data_events,
        "lifecycle_validation_failures": lifecycle_count,
        # THE SEPARATION. Three blocks, no sum. A feed outage is not a
        # trading cost and a predicted cost is not an observation.
        "separation": {
            "modelled_cost": {
                "predicted_cost_bps": predicted,
                "source": "cost model (prediction)" if predicted is not None else "no cost model supplied",
            },
            "observed_gap": dict(observed),
            "data_quality_error": {
                "outage_days": sum(1 for e in ordered if e.get("outages")),
                "skipped_fills": skipped_fills,
                "stale_data_events": stale_data_events,
                "lifecycle_validation_failures": lifecycle_count,
            },
        },
        "notes": notes,
    }
