"""Simple backtester: replays candles through a strategy with a paper portfolio."""
from __future__ import annotations

from .strategy import Signal


def backtest(candles: list[dict], strategy, start_cash: float = 10_000.0, fee: float = 0.001) -> dict:
    cash = start_cash
    position = 0.0
    trades = 0

    for i in range(len(candles)):
        window = candles[: i + 1]
        sig = strategy.signal(window)
        price = candles[i]["close"]

        if sig is Signal.BUY and cash > 0:
            position = (cash / price) * (1 - fee)
            cash = 0.0
            trades += 1
        elif sig is Signal.SELL and position > 0:
            cash = position * price * (1 - fee)
            position = 0.0
            trades += 1

    final_value = cash + position * candles[-1]["close"]
    return {
        "start_value": start_cash,
        "final_value": round(final_value, 2),
        "return_pct": round((final_value / start_cash - 1) * 100, 2),
        "trades": trades,
        "buy_and_hold_pct": round((candles[-1]["close"] / candles[0]["close"] - 1) * 100, 2),
    }
