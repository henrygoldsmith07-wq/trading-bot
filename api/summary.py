"""Vercel serverless endpoint: /api/summary

Serves the authoritative out-of-sample summary computed by the Python bot
package: a fixed-rule (RiskEnsemble) walk-forward on BTC daily data under
realistic frictions, plus the current live weight recommendation.

Zero third-party dependencies — a pure-stdlib ASGI app, which Vercel's
Python runtime detects automatically. Deployed with the repo's classic
zero-config layout (static `public/` + functions in `api/`).
"""
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.data import fetch_daily_history  # noqa: E402
from bot.strategy import risk_ensemble  # noqa: E402
from bot.walkforward import absolute_folds, walk_forward_at  # noqa: E402

ENGINE_KWARGS: dict[str, Any] = dict(
    fee=0.001,
    spread_bps=5.0,
    slippage_bps=5.0,
    execution="next_open",
    risk_free_annual=0.03,
    rebalance_band=0.05,
)


def build_summary(symbol: str = "BTCUSDT") -> dict:
    candles = fetch_daily_history(symbol)
    folds = absolute_folds(candles, train_days=1095, test_days=365)
    wf = walk_forward_at(candles, folds, candidates=[risk_ensemble()], **ENGINE_KWARGS)

    days = sorted(wf["daily"])
    rets = [wf["daily"][d] for d in days]
    curve = [1.0]
    for r in rets:
        curve.append(curve[-1] * (1.0 + r))

    # downsample the curve for the payload (~180 points), always keeping the final value
    step = max(1, len(curve) // 180)
    idxs = list(range(0, len(curve), step))
    if idxs[-1] != len(curve) - 1:
        idxs.append(len(curve) - 1)
    points = [
        {"t": days[min(i, len(days) - 1)], "v": round(curve[i], 5)}
        for i in idxs
    ]

    strategy = risk_ensemble()
    return {
        "symbol": symbol,
        "strategy": "RiskEnsemble (fixed, a-priori)",
        "generated_at": datetime.now(UTC).isoformat(),
        "frictions": ENGINE_KWARGS,
        "oos": {
            "first_day": datetime.fromtimestamp(wf["first_day"] / 1000, tz=UTC).date().isoformat(),
            "last_day": datetime.fromtimestamp(wf["last_day"] / 1000, tz=UTC).date().isoformat(),
            "cagr": round(wf["cagr"], 4),
            "sharpe": round(wf["sharpe"], 3),
            "max_drawdown": round(wf["max_drawdown"], 4),
            "final": round(wf["equity"], 4),
            "exposure": round(wf["exposure"], 3),
            "turnover": round(wf["turnover"], 2),
        },
        "folds": [p["strategy"] for p in wf["folds"]],
        "curve": points,
        "live": {
            "price": candles[-1]["close"],
            "price_date": datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=UTC).date().isoformat(),
            "weight_now": round(strategy.weight(candles), 3),
        },
        "disclaimer": "Paper trading only. Educational software. Not financial advice.",
    }


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    try:
        payload = build_summary()
        status, body = 200, json.dumps(payload).encode()
    except Exception as e:  # surface a clean error to the dashboard
        status, body = 502, json.dumps({"error": str(e)}).encode()
    headers = [
        [b"content-type", b"application/json"],
        [b"cache-control", b"no-store"],
        [b"access-control-allow-origin", b"*"],
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    print(json.dumps(build_summary(), indent=2)[:2000])
