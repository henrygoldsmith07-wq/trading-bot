"""Tradeable universe: top Binance USDT spot pairs by quote volume."""
from __future__ import annotations

import json

from .data import _get

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"

# Tokens that quote or track fiat, plus leveraged-token suffixes — not tradeable
# trend-following candidates for this bot.
STABLE_OR_NONVANILLA = {
    "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "AEUR", "USD1",
    "PAXG", "XUSD", "USDE", "BFUSD",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def parse_symbols(ticker_json: str, quote: str = "USDT", n: int = 10) -> list[str]:
    rows = json.loads(ticker_json)
    out = []
    for t in rows:
        symbol = t.get("symbol", "")
        if not symbol.endswith(quote):
            continue
        base = symbol[: -len(quote)]
        if base in STABLE_OR_NONVANILLA or base.endswith(LEVERAGED_SUFFIXES):
            continue
        try:
            vol = float(t.get("quoteVolume", 0) or 0)
        except ValueError:
            continue
        out.append((vol, symbol))
    out.sort(reverse=True)
    return [s for _, s in out[:n]]


def top_symbols(n: int = 10, quote: str = "USDT") -> list[str]:
    return parse_symbols(_get(TICKER_URL), quote=quote, n=n)
