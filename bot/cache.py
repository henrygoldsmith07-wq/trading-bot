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
    candles, came_from_cache, _fetched_at = load_or_fetch_meta(symbol, fetch_fn, ttl_hours=ttl_hours)
    return candles, came_from_cache


def load_or_fetch_meta(
    symbol: str,
    fetch_fn,
    ttl_hours: float = 12.0,
    cache_only: bool = False,
) -> tuple[list[dict], bool, float | None]:
    """Return (candles, came_from_cache, downloaded_at_epoch).

    `cache_only=True` never hits the network: a missing/stale/expired cache
    raises FileNotFoundError — used by `reproduce` so frozen datasets cannot
    silently refresh to different bytes."""
    path = _cache_path(symbol)
    now = time.time()
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            if blob.get("candles") and (
                not cache_only and now - blob.get("fetched_at", 0) < ttl_hours * 3600
                or cache_only
            ):
                return blob["candles"], True, blob.get("fetched_at")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if cache_only:
        raise FileNotFoundError(f"frozen source data for {symbol!r} not available in cache ({path})")
    candles = fetch_fn(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": now, "candles": candles}))
    return candles, False, now


def clear() -> None:
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
