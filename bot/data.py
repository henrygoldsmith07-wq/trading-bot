"""Market data fetching using free public endpoints (no API key required)."""
from __future__ import annotations

import json
import time
import urllib.request


BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def _get(url: str, timeout: int = 15, attempts: int = 3) -> str:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def _parse_klines(raw: str) -> list[dict]:
    candles = []
    for k in json.loads(raw):
        candles.append(
            {
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return candles


def fetch_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> list[dict]:
    """Fetch the most recent OHLCV candles from Binance's public REST API."""
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    return _parse_klines(_get(url))


def fetch_daily_history(symbol: str = "BTCUSDT", since_ms: int | None = None, max_candles: int = 4000) -> list[dict]:
    """Fetch full daily history, paginating through Binance's 1000-candle limit.

    Pages forward from `since_ms` (or the symbol's first candle when omitted —
    Binance's default is to return the *newest* window, which breaks paging).
    """
    out: list[dict] = []
    start = since_ms if since_ms is not None else 0
    while len(out) < max_candles:
        url = f"{BINANCE_KLINES}?symbol={symbol}&interval=1d&limit=1000"
        if start is not None:
            url += f"&startTime={start}"
        batch = _parse_klines(_get(url))
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
    return out[:max_candles]
