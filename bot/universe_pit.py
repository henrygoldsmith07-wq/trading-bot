"""Point-in-time universe: date -> eligible assets, using only past data.

Survivorship bias, concretely: today's top-volume pairs are partly today's
WINNERS. Backtesting a portfolio built from 2026's winners over 2020..2026
silently imports knowledge of which alts survive.

This module builds eligibility from each symbol's OWN history:

    eligible(symbol, day) =
        listed before day                (first candle <= day - min_history)
        + minimum history at that date   (>= min_history_days of bars)
        + trailing dollar volume         (mean daily quote volume >= floor)
        + still trading at that date     (last bar >= day)
        + static filters                 (stablecoins/leverage handled upstream)

No future information crosses the boundary: a 2023 listing only becomes
eligible in 2023, however large it is today.

RESIDUAL SURVIVORSHIP BIAS — STATED PRECISELY, NOT OVERSTATED
------------------------------------------------------------
The membership RULES above are point-in-time, but the INITIAL SYMBOL SET they
are applied to is not. The candidate list is seeded from TODAY's Binance
ticker ranking (bot/universe.py), so the reconstructed universe is:

    "of the assets that still exist and are liquid today, which of those were
     eligible on date d?"

It is NOT:

    "of the assets that existed on date d, which were eligible on date d?"

The difference is exactly the assets that were delisted, re-listed, or purged
from the API before we ever fetched them: they can never appear, because we
never knew they existed. This reconstruction therefore REMOVES the
membership-timing bias it can see (late listings, liquidity decay, deaths)
and CANNOT remove the bias from assets absent from the seed list. Any claim
that it "eliminates survivorship bias" would be false; it reduces a
measurable part of it and leaves an unquantifiable remainder. Historical
numbers derived this way remain mildly optimistic.

The forward snapshot log below is the actual cure, and it is the only complete
one: every day the scheduled runner records today's universe, so membership
is observed rather than reconstructed. Those dated snapshots are the seed for
a genuinely historical universe (see `eligibility_from_snapshots`), and future
backtests should consume them directly instead of rebuilding membership from
today's survivors. Do not fabricate historical constituents to fill the gap —
a synthetic delisted asset would be a fabricated data point dressed as
evidence.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

DAY_MS = 86_400_000

# Append-only, dated snapshot of the tradeable universe. Forward runs append
# to it daily; it is the only membership evidence that is genuinely
# point-in-time rather than reconstructed from today's survivors.
UNIVERSE_LOG = Path("universe_log.jsonl")


# ---------------------------------------------------------------------------
# listing dates (first candle per symbol) — cached on disk
# ---------------------------------------------------------------------------

_LISTING_CACHE = Path(".cache/listing_map.json")


def fetch_listing_dates(symbols: list[str], fetch_first_open=None) -> dict[str, int]:
    """{symbol: first_candle_open_ms}. `fetch_first_open` injects the network
    call in tests; default hits Binance klines with startTime=0&limit=1."""
    if _LISTING_CACHE.exists():
        try:
            cached = json.loads(_LISTING_CACHE.read_text())
            missing = [s for s in symbols if s not in cached]
            if not missing:
                return {s: cached[s] for s in symbols}
        except (json.JSONDecodeError, OSError):
            pass
    if fetch_first_open is None:
        from .data import _get_any, klines_urls

        def fetch_first_open(sym: str) -> int | None:
            # mirrors, not one host: a 451 here would otherwise leave every
            # symbol undated and quietly drop assets from the eligible set
            urls = klines_urls(f"?symbol={sym}&interval=1d&startTime=0&limit=1")
            try:
                raw = _get_any(urls)
                rows = json.loads(raw)
                return int(rows[0][0]) if rows else None
            except Exception:
                return None

    out: dict[str, int] = {}
    if _LISTING_CACHE.exists():
        try:
            out.update(json.loads(_LISTING_CACHE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    for s in symbols:
        first = fetch_first_open(s)
        if first is not None:
            out[s] = first
    _LISTING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _LISTING_CACHE.write_text(json.dumps(out))
    return {s: out[s] for s in symbols if s in out}


# ---------------------------------------------------------------------------
# point-in-time eligibility — pure function of pre-fetched histories
# ---------------------------------------------------------------------------

def mean_daily_quote_volume(candles: list[dict], end_ms: int, window: int) -> float | None:
    """Mean daily quote volume over the `window` bars ending strictly BEFORE
    `end_ms`. None when the feed carries no quote volumes."""
    prior: list[float] = []
    for c in candles:
        if c["open_time"] >= end_ms:
            break  # histories are sorted; everything from here is not "past"
        v = c.get("quote_volume")
        if v is not None:
            prior.append(float(v))
    prior = prior[-window:]
    if not prior:
        return None
    return sum(prior) / len(prior)


def eligible_on(
    candles: list[dict],
    day_ms: int,
    *,
    min_history_days: int = 90,
    min_mean_daily_quote_volume: float = 5_000_000.0,
    volume_window_days: int = 30,
) -> bool:
    """Was this asset an eligible candidate ON `day_ms`, using only data
    from before that day?"""
    if not candles:
        return False
    # listed long enough ago to have the required history
    if candles[0]["open_time"] > day_ms - min_history_days * DAY_MS:
        return False
    # still alive at that date (last print not far in the past relative to day)
    if candles[-1]["open_time"] < day_ms - 2 * DAY_MS:
        return False
    vol = mean_daily_quote_volume(candles, day_ms, volume_window_days)
    if vol is None:
        return True  # no volume field (e.g. Yahoo): cannot judge -> do not block
    return vol >= min_mean_daily_quote_volume


def eligibility_from_snapshots(
    log_path: str | Path = UNIVERSE_LOG,
    timeline: list[int] | None = None,
) -> dict[int, set[str]]:
    """Dated universe snapshots -> {day_ms: set(symbols actually holdable then)}.

    This is the OBSERVED-membership path, free of the survivorship limitation
    that `point_in_time_universe` cannot escape: a snapshot for date d was
    written on date d, so the symbol set cannot have been chosen with knowledge
    of later returns. Where a day has no snapshot, it is OMITTED from the
    result rather than interpolated — an invented membership for a gap day
    would be fabricated evidence, and the caller must fail closed on the
    missing day instead.
    """
    snaps = load_snapshots(log_path)
    out: dict[int, set[str]] = {}
    for date_str, symbols in snaps.items():
        try:
            day = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
        except ValueError:
            continue  # unparseable date in a legacy line: not usable as evidence
        out[int(day.timestamp() * 1000)] = set(symbols)
    if timeline is not None:
        out = {t: s for t, s in out.items() if t in set(timeline)}
    return out


def assert_membership_consistent(
    contributors: set[str],
    eligible: set[str] | None,
    day: int,
) -> None:
    """Fail closed if a symbol contributes return/allocation while ineligible.

    This is the invariant that makes the PIT universe a real MASK rather than
    a denominator that merely rescales exposure. A denominator-only control
    lets an ineligible asset's return stream flow into the portfolio with its
    contribution scaled down; the number still "looks" point-in-time while the
    ineligible asset is in fact being paid for. Contributors must be a subset
    of the eligible set on every single day.
    """
    if eligible is None:
        return  # no PIT control in force for this day; the fixed-denominator path
    if not isinstance(eligible, set):
        raise TypeError("eligible must be a set of symbols")
    extra = set(contributors) - set(eligible)
    if extra:
        raise AssertionError(
            f"PIT membership violated at {day}: symbols {sorted(extra)} contribute "
            f"but are not eligible; contributors must be a subset of {sorted(eligible)}"
        )


def point_in_time_universe(
    histories: dict[str, list[dict]],
    timeline: list[int],
    *,
    min_history_days: int = 90,
    min_mean_daily_quote_volume: float = 5_000_000.0,
    volume_window_days: int = 30,
) -> dict[int, set[str]]:
    """{day_ms: set(eligible symbols)} for every timeline day.

    This is the date -> assets map: computed from each history's own past,
    so membership changes as listings/liquidity/deaths actually happened.
    """
    out: dict[int, set[str]] = {}
    for t in timeline:
        elig = {
            s for s, candles in histories.items()
            if eligible_on(
                candles, t,
                min_history_days=min_history_days,
                min_mean_daily_quote_volume=min_mean_daily_quote_volume,
                volume_window_days=volume_window_days,
            )
        }
        out[t] = elig
    return out


# ---------------------------------------------------------------------------
# forward snapshots: start building the genuinely point-in-time dataset NOW
# ---------------------------------------------------------------------------

def record_snapshot(
    ranked: list[tuple[str, float]],
    log_path: str | Path = UNIVERSE_LOG,
    now: datetime | None = None,
    source: str = "binance-ticker24h",
) -> dict:
    """Append today's ranked universe (today's data recorded TODAY is
    point-in-time by construction). Same-date entries are idempotent."""
    now = now or datetime.now(UTC)
    today = now.date().isoformat()
    prior = load_snapshots(log_path)
    if today in prior:
        return {"status": "already_logged", "date": today, "symbols": prior[today]}
    entry = {
        "date": today,
        "generated_at": now.isoformat(),
        "source": source,
        "universe": [{"symbol": s, "quote_volume_usd": round(v, 2)} for s, v in ranked],
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
    prior[today] = [s for s, _ in ranked]
    return {"status": "logged", "date": today, "n": len(ranked)}


def load_snapshots(log_path: str | Path = UNIVERSE_LOG) -> dict[str, list[str]]:
    p = Path(log_path)
    if not p.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            out[e["date"]] = [u["symbol"] for u in e["universe"]]
        except (json.JSONDecodeError, KeyError):
            continue
    return out
