"""Point-in-time universe: date -> eligible assets, using only past data.

Survivorship bias, concretely: today's top-volume pairs are partly today's
WINNERS. Backtesting a portfolio built from 2026's winners over 2020..2026
silently imports knowledge of which alts survive. The fixed denominator
stops dead assets donating capital to survivors; it does NOT stop the
universe selection itself from peeking.

This module builds eligibility from each symbol's OWN history:

    eligible(symbol, day) =
        listed before day                (first candle <= day - min_history)
        + minimum history at that date   (>= min_history_days of bars)
        + trailing dollar volume         (mean daily quote volume >= floor)
        + still trading at that date     (last bar >= day)
        + static filters                 (stablecoins/leverage handled upstream)

No future information crosses the boundary: a 2023 listing only becomes
eligible in 2023, however large it is today.

Honest residual limitation: symbols DELISTED AND PURGED from Binance's API
before we ever fetched them are invisible. The forward snapshot log below is
the cure — every day the scheduled runner records today's universe, building
a genuinely point-in-time dataset for next year's backtests.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

DAY_MS = 86_400_000


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

UNIVERSE_LOG = Path("universe_log.jsonl")


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
