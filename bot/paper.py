"""Paper trading engine: simulated fills, no real orders ever placed."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .data import fetch_candles
from .strategy import Signal


class PaperBroker:
    def __init__(self, start_cash: float = 10_000.0, fee: float = 0.001, state_file: str = "paper_state.json"):
        self.fee = fee
        self.state_file = Path(state_file)
        state = self._load()
        self.cash: float = state.get("cash", start_cash)
        self.position: float = state.get("position", 0.0)

    def _load(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def buy(self, price: float) -> str:
        if self.cash <= 0:
            return "skipped buy: no cash"
        self.position = (self.cash / price) * (1 - self.fee)
        self.cash = 0.0
        self._save(price, "BUY")
        return f"BOUGHT @ {price:.2f}"

    def sell(self, price: float) -> str:
        if self.position <= 0:
            return "skipped sell: no position"
        self.cash = self.position * price * (1 - self.fee)
        self.position = 0.0
        self._save(price, "SELL")
        return f"SOLD @ {price:.2f}"

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def _save(self, price: float, action: str) -> None:
        self.state_file.write_text(
            json.dumps({"cash": self.cash, "position": self.position, "last_action": action, "last_price": price}, indent=2)
        )


def run(symbol: str = "BTCUSDT", interval: str = "1h", poll_seconds: int = 60, fast: int = 20, slow: int = 50, strategy_name: str = "sma"):
    """Live paper-trading loop. Ctrl+C to stop; state persists in paper_state.json."""
    from .strategy import SmaCrossover, TrendVol

    broker = PaperBroker()
    if strategy_name == "trendvol":
        strategy = TrendVol(50, 20, 0.25)  # the walk-forward out-of-sample winner
        label = "TrendVol(50,0.25)"
    else:
        strategy = SmaCrossover(fast, slow)
        label = f"SMA {fast}/{slow}"
    print(f"Paper trading {symbol} ({interval}) | {label} | Ctrl+C to stop")
    try:
        while True:
            candles = fetch_candles(symbol, interval)
            price = candles[-1]["close"]
            w = strategy.weight(candles)
            sig = strategy.signal(candles) if hasattr(strategy, "signal") else None
            if sig is Signal.BUY or (strategy_name == "trendvol" and w > 0 and broker.position == 0):
                msg = broker.buy(price)
            elif sig is Signal.SELL or (strategy_name == "trendvol" and w == 0 and broker.position > 0):
                msg = broker.sell(price)
            else:
                msg = f"hold (target weight {w:.2f})" if strategy_name == "trendvol" else "hold"
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} {price:.2f} -> {msg} | equity {broker.equity(price):.2f}")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print(f"\nStopped. Final equity: {broker.equity(price):.2f}")
