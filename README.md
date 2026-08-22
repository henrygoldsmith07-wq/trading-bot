# Trading Bot (Paper Trading)

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, FRED — no API keys) and now includes a serious research pipeline: multiple strategies, proper performance metrics, and **walk-forward out-of-sample validation** against the S&P 500.

> ⚠️ Educational software. Not financial advice. Past out-of-sample performance does not guarantee future results — do not wire real money to anything based on this repo.

## Headline result (real data, out-of-sample)

`python -m bot compare` walks forward over 9 years of BTC daily candles, re-picking the best strategy **every year using only prior data**, then compares the stitched out-of-sample track record against the actual S&P 500 over the same window (2020-08 → 2026-08):

```
                     Bot (OOS)       S&P 500  BTC buy&hold
CAGR                     21.0%         14.9%         36.4%
Volatility               18.4%         16.7%         57.5%
Sharpe                    1.13          0.92          0.83
Max drawdown            -25.0%        -25.4%        -76.6%
Growth of $1              3.13          2.30          6.49
```

The bot beats the S&P 500 on CAGR, Sharpe ratio, and max drawdown simultaneously, with less than half of buy-and-hold BTC's volatility and a third of its drawdown. Every fold independently selected `TrendVol(50, 0.25)` — a trend filter (price vs 50-day SMA) with inverse-volatility position sizing targeting 25% annualized vol, after 0.1%-per-turnover fees.

**Why this is (reasonably) honest:** the strategy is never scored on data used to pick it. Each yearly fold trains on the entire history up to that point, then trades the next 12 months unseen. Fees are charged on turnover. Weights are long-only and capped at 1x (no leverage). Remaining caveats: single asset (BTC), daily-close fills (no intraday slippage), and walk-forward over one historical path is still not a guarantee.

## Features

- **Strategy library** — SMA crossover, trend-following with volatility targeting, RSI dip-buying in uptrends, MACD, and ensembles
- **Metrics engine** — CAGR, annualized volatility, Sharpe, max drawdown, exposure, turnover (calendar-day based so crypto's 365d year and equities' 252d year compare fairly)
- **Backtest engine** — daily bars, no lookahead by construction (weights see only past candles), turnover-priced fees, weights clamped to [0, 1]
- **Walk-forward validation** — expanding-window folds, selection by training Sharpe, stitched out-of-sample track record
- **Benchmark comparison** — S&P 500 daily closes from FRED (Yahoo Finance fallback)
- **Live paper trading** — polls prices and simulates fills; state persists across restarts
- Stdlib only (Python 3.10+); pytest for tests

## Usage

```bash
# Full walk-forward out-of-sample comparison vs the S&P 500
python -m bot compare

# Tune the validation
python -m bot compare --train-days 730 --test-days 365 --fee 0.0015

# Quick hourly SMA backtest (original toy strategy)
python -m bot backtest --symbol BTCUSDT --interval 1h

# Live paper trading — trendvol uses daily bars and hourly polls
python -m bot trade --strategy trendvol
python -m bot trade --strategy sma --interval 15m
```

## Project layout

```
bot/
  strategy.py    # strategy library + candidate pool
  engine.py      # daily-bar backtest engine (no lookahead, fee-aware)
  metrics.py     # CAGR / Sharpe / vol / max drawdown
  walkforward.py # expanding-window walk-forward validation
  benchmark.py   # S&P 500 data (FRED primary, Yahoo fallback)
  data.py        # Binance candles with full-history pagination
  backtest.py    # legacy simple backtester (kept for the toy SMA path)
  paper.py       # paper broker + live loop
tests/           # 31 unit tests
```

## Running tests

```bash
pip install pytest
pytest
```
