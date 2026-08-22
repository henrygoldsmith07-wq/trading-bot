"""Disk cache for daily candle history.

Walk-forward on a dozen assets refetches nothing between runs; cached files
older than `ttl_hours` are refreshed from the API.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

CACHE_DIR = Path(".cache")


def _cache_path(symbol: str) -> Path:
    safe = "".join(ch for ch in symbol if ch.isalnum())
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{safe}_1d.json"


def load_or_fetch(symbol: str, fetch_fn, ttl_hours: float = 12.0) -> tuple[list[dict], bool]:
    """Return (candles, came_from_cache)."""
    path = _cache_path(symbol)
    now = time.time()
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            if now - blob.get("fetched_at", 0) < ttl_hours * 3600 and blob.get("candles"):
                return blob["candles"], True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    candles = fetch_fn(symbol)
    path.write_text(json.dumps({"fetched_at": now, "candles": candles}))
    return candles, False


def clear() -> None:
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
