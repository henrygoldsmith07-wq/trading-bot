"""S&P 500 benchmark data from free public sources (no API key).

Primary: FRED's SP500 series (plain CSV, ~10 years of daily closes).
Fallback: Yahoo Finance chart API for ^GSPC.
Returns rows as [{"date": datetime.date, "close": float}] sorted ascending.
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from datetime import UTC, datetime

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=10y&interval=1d"


def _get(url: str, attempts: int = 3) -> str:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (trading-bot research)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")  # loop raises on final failed attempt


def parse_fred_csv(text: str) -> list[dict]:
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        value = rec.get("SP500", ".")
        if value in (".", "", None):
            continue
        rows.append({"date": datetime.strptime(rec["observation_date"], "%Y-%m-%d").date(), "close": float(value)})
    return rows


def parse_yahoo_json(text: str) -> list[dict]:
    payload = json.loads(text)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    closes = result["indicators"]["quote"][0].get("close", [])
    rows = []
    for ts, close in zip(timestamps, closes, strict=False):
        if close is None:
            continue
        rows.append({"date": datetime.fromtimestamp(ts, tz=UTC).date(), "close": float(close)})
    return rows


def fetch_sp500() -> list[dict]:
    try:
        return parse_fred_csv(_get(FRED_URL))
    except Exception:
        return parse_yahoo_json(_get(YAHOO_URL))


def slice_window(rows: list[dict], start_date, end_date) -> list[dict]:
    return [r for r in rows if start_date <= r["date"] <= end_date]


def equity_metrics(rows: list[dict], risk_free_annual: float = 0.0) -> dict:
    """Metrics for a buy-and-hold equity curve over the given daily closes."""
    from .metrics import cagr, calmar, expected_shortfall, max_drawdown, sharpe, sortino, var_hist, volatility

    closes = [r["close"] for r in rows]
    if len(closes) < 3:
        return {"cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "final": 1.0}
    equity = [1.0]
    for i in range(1, len(closes)):
        equity.append(equity[-1] * closes[i] / closes[i - 1])
    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    days = (rows[-1]["date"] - rows[0]["date"]).days
    periods = 252  # S&P trading days per year
    mdd = max_drawdown(equity)
    cagr_v = cagr(equity, days)
    return {
        "cagr": cagr_v,
        "vol": volatility(returns, periods),
        "sharpe": sharpe(returns, periods, risk_free_annual),
        "sortino": sortino(returns, periods, risk_free_annual),
        "calmar": calmar(cagr_v, mdd),
        "max_drawdown": mdd,
        "var95": var_hist(returns, 0.95),
        "es95": expected_shortfall(returns, 0.95),
        "final": equity[-1],
    }
