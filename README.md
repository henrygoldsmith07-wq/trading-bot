# Trading Bot (Paper Trading)

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data from Binance's REST API (no API key needed) and trades an SMA-crossover strategy against a simulated portfolio.

> ⚠️ Educational software. Not financial advice. Do not wire real money to anything based on this repo.

## Features

- **SMA crossover strategy** — buys when the fast moving average crosses above the slow one, sells on the reverse
- **Backtesting** — replay historical candles and compare against buy-and-hold
- **Live paper trading** — polls prices and simulates fills with a 0.1% fee; state persists across restarts
- Zero dependencies beyond Python 3.10+ (stdlib only); pytest for tests

## Usage

```bash
# Backtest on the last 500 hourly candles
python -m bot backtest --symbol BTCUSDT --interval 1h

# Run the live paper-trading loop (Ctrl+C to stop)
python -m bot trade --symbol ETHUSDT --interval 15m --poll 60

# Tune strategy periods
python -m bot backtest --fast 10 --slow 30
```

## Project layout

```
bot/
  strategy.py   # SmaCrossover strategy + Signal enum
  data.py       # candle fetching from Binance public API
  backtest.py   # backtesting engine
  paper.py      # paper broker + live loop
tests/          # pytest unit tests
```

## Running tests

```bash
pip install pytest
pytest
```
