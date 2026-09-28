"""Order-lifecycle timestamp provenance.

Every forward order carries four timestamps describing how it came to exist:

    signal_generated_ts  when the strategy produced the raw target
    intent_ts            the session the runner INTENDED to execute in
    submitted_ts         when the order was actually submitted
    fill_ts              when the paper fill was recorded

LEGACY CORRUPTION (real, committed, NOT to be silently rewritten)
------------------------------------------------------------------
Committed rows in `forward_log.jsonl` contain `intent_ts` values ~2.7 years in
the FUTURE (e.g. "2029-06-02" for a 2026-09-07 session). The cause was a unit
error: the intent time added a millisecond-scale constant to a seconds-scale
timestamp, so 86_400_000 extra *seconds* were added instead of 86_400. The
current runner computes `open_time/1000 + 86_400.0` (seconds), which is
correct.

Those rows are append-only evidence and are preserved byte-for-byte. What we
do instead is DETECT them, WARN loudly, and EXCLUDE them from the statistics
they would contaminate (latency, slippage, cost) — corrupted lifecycle
metadata is not evidence of anything, and must never silently enter return
arithmetic or fudge a measured slippage distribution.

INVARIANT enforced here:  signal <= intent <= submitted <= fill
and every timestamp must be timezone-aware. A value that fails is reported,
never repaired in place.
"""
from __future__ import annotations

from datetime import UTC, datetime

# The lifecycle fields, in their required order.
LIFECYCLE_FIELDS = ("signal_generated_ts", "intent_ts", "submitted_ts", "fill_ts")

# Reasons a lifecycle record is rejected. These feed exclusion counters.
REASON_UNPARSEABLE = "unparseable_lifecycle_ts"
REASON_NAIVE_TZ = "naive_timestamp"
REASON_OUT_OF_ORDER = "lifecycle_out_of_order"
REASON_FUTURE = "intent_in_future"

# Sanity bound: an intent timestamp may not sit more than this far from the
# session it is meant to belong to. A 1-day tolerance absorbs a UTC/local
# boundary; the observed legacy corruption is ~1000 days, so this separates
# "implausible" from "impossible" without special-casing the old bug. A
# deliberately DELAYED scheduled run is still bounded by its own session.
MAX_INTENT_SKEW_DAYS = 1.0


def _looks_iso(value: str) -> bool:
    """True when a string parses as an ISO timestamp (possibly without a zone)."""
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def parse_lifecycle_ts(value) -> datetime | None:
    """Parse one lifecycle timestamp; None when absent or unusable.

    Naive (tz-less) strings are rejected deliberately: a timestamp without a
    timezone cannot be ordered against one that has a zone, and silently
    assuming UTC is exactly how ms/second confusions survive.
    """
    if not isinstance(value, str) or not value:
        return None
    if not _looks_iso(value):
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else None


def lifecycle_problems(order: dict, session_date: str | None = None) -> list[str]:
    """Return the provenance problems for one order record.

    Empty list == trustworthy lifecycle metadata. Non-empty == the order's
    timestamps must not be used for latency/slippage/cost statistics.
    """
    problems: list[str] = []
    parsed: dict[str, datetime] = {}
    for field in LIFECYCLE_FIELDS:
        raw = order.get(field)
        if raw is None:
            # Absent is not a failure: older schemas never wrote every field.
            continue
        dt = parse_lifecycle_ts(raw)
        if dt is None:
            problems.append(REASON_NAIVE_TZ if _looks_iso(str(raw)) else REASON_UNPARSEABLE)
            continue
        parsed[field] = dt.astimezone(UTC)

    # Ordering: signal <= intent <= submitted <= fill. Compare only fields
    # that are actually present, so a partially-stamped legacy row is judged
    # on what it claims rather than padded with invented times.
    ordered = [f for f in LIFECYCLE_FIELDS if f in parsed]
    for earlier, later in zip(ordered, ordered[1:]):
        if parsed[earlier] > parsed[later]:
            problems.append(REASON_OUT_OF_ORDER)
            break

    # The intended session must BE the session being accounted for. An intent
    # timestamp centuries away is the signature of a unit error, not of a
    # delayed run (a delayed run stays within its own scheduled session).
    if session_date is not None and "intent_ts" in parsed:
        try:
            session = datetime.fromisoformat(session_date).replace(tzinfo=UTC)
        except ValueError:
            session = None
        if session is not None:
            skew_days = abs((parsed["intent_ts"] - session).total_seconds()) / 86_400.0
            if skew_days > MAX_INTENT_SKEW_DAYS:
                problems.append(REASON_FUTURE)
    return problems


def order_is_trustworthy(order: dict, session_date: str | None = None) -> bool:
    """Whether this order's lifecycle metadata may inform statistics."""
    return not lifecycle_problems(order, session_date)


def audit_log_lifecycle(entries: list[dict]) -> dict:
    """Audit every order in a forward log for lifecycle provenance.

    Returns counts per problem reason plus affected (date, symbol) pairs, so
    the forward report can state exactly how much of its latency/slippage
    evidence was set aside. Nothing here mutates the log.
    """
    flagged: dict[str, list[str]] = {}
    n_orders = 0
    n_flagged = 0
    for entry in entries:
        session_date = entry.get("date")
        for order in entry.get("orders", []) or []:
            n_orders += 1
            problems = lifecycle_problems(order, session_date)
            if not problems:
                continue
            n_flagged += 1
            label = f"{session_date}:{order.get('symbol', '?')}"
            for reason in problems:
                flagged.setdefault(reason, []).append(label)
    return {
        "n_orders": n_orders,
        "n_orders_flagged": n_flagged,
        "n_orders_trusted": n_orders - n_flagged,
        "flagged_by_reason": {k: len(v) for k, v in sorted(flagged.items())},
        "flagged_examples": {k: sorted(set(v))[:5] for k, v in sorted(flagged.items())},
        "any_flagged": n_flagged > 0,
    }
