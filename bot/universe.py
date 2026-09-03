"""Tradeable universe: top Binance USDT pairs by quote volume plus ETFs from
other asset classes (equities, gold, bonds) via Yahoo Finance.

Universe selection uses *today's* volume ranking, which embeds survivorship
bias — today's top pairs are partly today's winners. Free public APIs do not
offer point-in-time constituents, so this cannot be fully corrected; the
portfolio combiner's fixed denominator and the printed disclosure are the
controls we apply. Treat absolute numbers as slightly optimistic.
"""
from __future__ import annotations

import json
from typing import TypedDict

from .data import _get

# Binance answers HTTP 451 to some datacentre address ranges — GitHub Actions
# runners among them — so a single hard-coded host makes the scheduled paper
# run fail for reasons that have nothing to do with the strategy. These are the
# public mirrors, tried in order; data-api.binance.vision is the one Binance
# publishes for market data without the geo restriction.
TICKER_URLS: tuple[str, ...] = (
    "https://api.binance.com/api/v3/ticker/24hr",
    "https://api1.binance.com/api/v3/ticker/24hr",
    "https://api2.binance.com/api/v3/ticker/24hr",
    "https://api3.binance.com/api/v3/ticker/24hr",
    "https://data-api.binance.vision/api/v3/ticker/24hr",
)
TICKER_URL = TICKER_URLS[0]

# Tokens that quote or track fiat, plus leveraged-token suffixes — not tradeable
# trend-following candidates for this bot.
STABLE_OR_NONVANILLA = {
    "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "AEUR", "USD1",
    "PAXG", "XUSD", "USDE", "BFUSD",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


class EtfSpec(TypedDict, total=False):
    """Cross-asset-class sleeve traded via Yahoo Finance daily bars."""

    symbol: str
    asset_class: str
    periods_per_year: int
    session: str  # trading-calendar type: "us_equity" | "continuous"


# Non-crypto asset classes the data supports (Yahoo Finance, daily bars).
# `session` drives exchange-calendar-aware forward execution: NYSE assets are
# only traded after their session closes (bot/prospective._session_pending).
ETF_UNIVERSE: list[EtfSpec] = [
    {"symbol": "SPY", "asset_class": "US equity ETF", "periods_per_year": 252, "session": "us_equity"},
    {"symbol": "GLD", "asset_class": "Gold ETF", "periods_per_year": 252, "session": "us_equity"},
    {"symbol": "TLT", "asset_class": "20y Treasury ETF", "periods_per_year": 252, "session": "us_equity"},
]


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


def fetch_ticker_json(urls: tuple[str, ...] = TICKER_URLS, timeout: int = 15) -> str:
    """Fetch 24h ticker rows, falling back across mirror hosts.

    Returns the raw JSON body on the first host that answers. If every host
    fails, re-raises the LAST error rather than the first, so the log shows why
    the final attempt failed instead of a stale reason from the primary.
    """
    last: Exception | None = None
    for url in urls:
        try:
            # attempts=1: a mirror that refuses us will keep refusing, and the
            # point of the loop is to try a DIFFERENT host, not to wait.
            return _get(url, timeout=timeout, attempts=1)
        except Exception as exc:  # any refusal: move on to the next mirror
            last = exc
    assert last is not None  # urls is non-empty, so the loop either returns or sets last
    raise last


def top_symbols(n: int = 10, quote: str = "USDT") -> list[str]:
    return parse_symbols(fetch_ticker_json(), quote=quote, n=n)
