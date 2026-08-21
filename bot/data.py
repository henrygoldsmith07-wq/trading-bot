"""Market data fetching using free public endpoints (no API key required)."""
from __future__ import annotations

import time
import urllib.request


BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> list[dict]:
    """Fetch OHLCV candles from Binance's public REST API."""
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    candles = []
    for k in _parse_json(raw):
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


def _parse_json(raw: str):
    import json

    return json.loads(raw)
