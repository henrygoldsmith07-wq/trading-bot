"""Market data fetching using free public endpoints (no API key required)."""
from __future__ import annotations

import json
import math
import time
import urllib.request

# Public mirrors, tried in order. One hard-coded host is a single point of
# failure FOR THE EVIDENCE ITSELF: when api.binance.com answers 451 to a whole
# datacentre range (GitHub Actions runners included), every crypto sleeve
# records an outage and the day is still counted as a forward trading day —
# ten of thirteen sleeves dark, graded as though the portfolio were observed.
# The mirrors serve the same candles, so falling back changes the transport,
# never the experiment.
#
# Order is measured, not guessed. Probed from a real GitHub Actions runner:
# every *.binance.com host answered 451 and data-api.binance.vision answered
# 200 — so the vision mirror goes FIRST, and the .com hosts remain as
# fallbacks for the networks where they do answer. Putting it last meant every
# fetch paid four dead requests before the one that worked.
BINANCE_KLINES_URLS: tuple[str, ...] = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
)
BINANCE_KLINES = BINANCE_KLINES_URLS[0]
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval=1d"
DAY_MS = 86_400_000


# 4xx codes that are deterministic for a given client and will NOT clear on
# retry. 429 is deliberately absent: rate limits do clear, so backing off is
# worth it. 451 matters most here — it is how Binance refuses whole datacentre
# ranges (GitHub Actions runners included). Retrying a 451 three times just
# burns 45s of the workflow's timeout before failing anyway.
NON_RETRYABLE_HTTP = frozenset({400, 401, 403, 404, 410, 451})


def _get(url: str, timeout: int = 15, attempts: int = 3) -> str:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as exc:
            if exc.code in NON_RETRYABLE_HTTP or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")  # loop raises on final failed attempt


def _get_any(urls: tuple[str, ...] | list[str], timeout: int = 15) -> str:
    """Try each mirror once, in order; re-raise the LAST failure.

    One attempt per host on purpose: the loop means "ask somewhere else", not
    "wait". A 451 is deterministic for the client IP, so retrying the host that
    just refused only burns the workflow's timeout. Re-raising the last error
    (not the first) means the log shows why the final attempt failed.
    """
    last: Exception | None = None
    for url in urls:
        try:
            return _get(url, timeout=timeout, attempts=1)
        except Exception as exc:  # any refusal: move on to the next mirror
            last = exc
    assert last is not None
    raise last


def klines_urls(query: str) -> tuple[str, ...]:
    """Every klines mirror with `query` (which must include the leading '?')."""
    return tuple(url + query for url in BINANCE_KLINES_URLS)


def _parse_klines(raw: str) -> list[dict]:
    candles = []
    for k in json.loads(raw):
        candle = {
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        # k[7] is quote-asset volume (true USD turnover) — required for
        # point-in-time liquidity/eligibility decisions; tolerate absent field
        if len(k) > 7:
            try:
                qv = float(k[7])
                if math.isfinite(qv):
                    candle["quote_volume"] = qv
            except (TypeError, ValueError):
                pass
        candles.append(candle)
    return candles


def fetch_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> list[dict]:
    """Fetch the most recent OHLCV candles from Binance's public REST API."""
    query = f"?symbol={symbol}&interval={interval}&limit={limit}"
    return clean_candles(_parse_klines(_get_any(klines_urls(query))))


def fetch_daily_history(symbol: str = "BTCUSDT", since_ms: int | None = None, max_candles: int = 4000) -> list[dict]:
    """Fetch full daily history, paginating through Binance's 1000-candle limit.

    Pages forward from `since_ms` (or the symbol's first candle when omitted —
    Binance's default is to return the *newest* window, which breaks paging).
    """
    out: list[dict] = []
    start = since_ms if since_ms is not None else 0
    while len(out) < max_candles:
        # mirrors are re-tried per page: a host can start refusing mid-history
        query = f"?symbol={symbol}&interval=1d&limit=1000"
        if start is not None:
            query += f"&startTime={start}"
        batch = _parse_klines(_get_any(klines_urls(query)))
        if not batch:
            break
        if out and batch[0]["open_time"] == out[-1]["open_time"]:
            batch = batch[1:]
        if not batch:
            break
        out.extend(batch)
        start = out[-1]["open_time"] + 86_400_000
        if len(batch) < 1000:
            break
    return clean_candles(out[:max_candles])


def recent_window(candles: list[dict], n: int) -> list[dict]:
    """The most recent `n` bars from an ascending history."""
    return candles[-n:] if len(candles) > n else candles


def clean_candles(candles: list[dict]) -> list[dict]:
    """Missing/invalid-data handling: drop non-finite and non-positive closes,
    deduplicate timestamps (keeping the higher-volume print), and sort by time."""
    best: dict[int, dict] = {}
    for c in candles:
        close = c.get("close")
        if close is None or not math.isfinite(close) or close <= 0:
            continue
        t = c["open_time"]
        existing = best.get(t)
        if existing is None or c.get("volume", 0.0) >= existing.get("volume", 0.0):
            best[t] = c
    return [best[t] for t in sorted(best)]


def is_stale(candles: list[dict], now_ms: int | None = None, max_age_days: float = 45.0) -> bool:
    """Delisted-asset detection: history that stopped updating weeks ago."""
    if not candles:
        return True
    now = time.time() * 1000 if now_ms is None else now_ms
    return (now - candles[-1]["open_time"]) > max_age_days * DAY_MS


def fetch_yahoo_daily(symbol: str, range_: str = "10y") -> list[dict]:
    """Daily OHLCV candles for ETFs/equities from Yahoo Finance's chart API."""
    url = YAHOO_CHART.format(symbol=symbol, range_=range_)
    payload = json.loads(_get(url))
    result = payload["chart"]["result"][0]
    ts = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    volumes = quote.get("volume", []) or []
    candles = []
    for k, (t, o, c) in enumerate(zip(ts, quote.get("open", []), quote.get("close", []), strict=False)):
        if c is None or not math.isfinite(c) or c <= 0:
            continue
        candle = {"open_time": int(t) * 1000, "close": float(c)}
        if o is not None and math.isfinite(o) and o > 0:
            candle["open"] = float(o)
        if k < len(volumes) and volumes[k] is not None and math.isfinite(volumes[k]) and volumes[k] > 0:
            candle["volume"] = float(volumes[k])
        candles.append(candle)
    return clean_candles(candles)


def gap_report(candles: list[dict], expected_interval_ms: int = DAY_MS) -> list[dict]:
    """Gaps longer than one expected interval: [(start_ms, end_ms, days)]."""
    out = []
    for prev, cur in zip(candles, candles[1:], strict=False):
        gap_days = (cur["open_time"] - prev["open_time"]) / expected_interval_ms
        if gap_days > 1.5:
            out.append({"start": prev["open_time"], "end": cur["open_time"], "days": gap_days})
    return out


def fill_small_gaps(candles: list[dict], max_gap_days: int = 3) -> list[dict]:
    """Forward-fill closes across gaps of at most `max_gap_days` missing bars.

    Filled bars carry `"filled": True` and zero volume, so indicators see a
    continuous series while anything downstream can distinguish real prints
    from carried ones. Gaps beyond the tolerance are left alone (a long
    outage or delisting must stay visible, never papered over).
    """
    if not candles:
        return []
    out = [dict(candles[0])]
    for c in candles[1:]:
        gap_days = (c["open_time"] - out[-1]["open_time"]) / DAY_MS
        n_missing = int(gap_days) - 1 if gap_days > 1.5 else 0
        if 0 < n_missing <= max_gap_days:
            for k in range(n_missing):
                out.append(
                    {
                        "open_time": c["open_time"] - (n_missing - k) * DAY_MS,
                        "open": out[-1]["close"],
                        "high": out[-1]["close"],
                        "low": out[-1]["close"],
                        "close": out[-1]["close"],
                        "volume": 0.0,
                        "filled": True,
                    }
                )
        out.append(dict(c))
    return out


def extend_returns_to_timeline(
    daily: dict[int, float],
    timeline: list[int],
) -> dict[int, float]:
    """Place an asset's daily returns on the shared portfolio timeline using
    exchange-calendar-aware semantics:

    - Interior gaps (weekends/holidays/outages between the asset's first and
      last return) become 0.0-return *invested* days: an ETF held over a
      weekend does not sit in cash, and its next trading bar carries the
      whole Fri->Mon move, so compounding stays exact under constant weights.
    - Days before the first return (late listing) and after the last
      (delisted/stale) stay ABSENT: the sleeve sits in cash there, which is
      the survivorship-safe convention — dead capital never reflows.
    """
    if not daily:
        return {}
    first_t, last_t = min(daily), max(daily)
    out = {}
    for t in timeline:
        if t in daily:
            out[t] = daily[t]
        elif first_t < t < last_t:
            out[t] = 0.0
    return out


def simulate_delisting(
    candles: list[dict],
    delist_at_ms: int,
    terminal_cost_bps: float = 10.0,
) -> list[dict]:
    """Truncate a history at `delist_at_ms` to simulate a delisting.

    The final print marks the asset down by `terminal_cost_bps` (forced
    liquidation into a bid-less book) and carries a note. Downstream stages
    treat the result exactly like a real death: stale detection fires past
    the cutoff and the portfolio strands the sleeve in cash.
    """
    kept = [c for c in candles if c["open_time"] <= delist_at_ms]
    if not kept:
        return []
    last = dict(kept[-1])
    last["close"] = last["close"] * (1.0 - terminal_cost_bps / 10_000.0)
    if "open" in last:
        last["open"] = last["open"] * (1.0 - terminal_cost_bps / 10_000.0)
    last["note"] = "simulated delisting"
    kept[-1] = last
    return kept
