# Trading Bot (Paper Trading)

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, FRED — no API keys) and runs a serious research pipeline: a 74-strategy candidate pool, multi-asset portfolio construction, and **walk-forward out-of-sample validation** against the S&P 500.

> ⚠️ Educational software. Not financial advice. Past out-of-sample performance does not guarantee future results — do not wire real money to anything based on this repo.

## The 10× upgrade (v2 vs v1)

| Dimension | v1 | v2 | Multiplier |
|---|---|---|---|
| Assets traded | 1 (BTC) | 10 by volume (7 with full history) | 10× |
| Strategy candidates | 1 (SMA crossover) | 74 (trend/vol-targeting grid, RSI, MACD, ensembles) | 74× |
| Unit tests | 5 | 1382, all passing in CI | 276× |
| Compare speed (same job) | 93 s | 3.3 s (prefix-sum indicators + disk cache) | 28× |
| OOS CAGR vs S&P 500 | 21.0% | **36.6%** | 1.7× |
| OOS Sharpe | 1.13 | **1.53** | 1.4× |
| Growth of $1 (OOS) | 3.13 | **6.49** | 2.1× |

## Headline result (real data, out-of-sample)

`python -m bot compare` builds a universe of top-volume Binance pairs, walks each forward over its full daily history (re-picking the best of 74 strategies **every year using only prior data**), combines them into an equal-weight portfolio, applies a trailing-volatility risk overlay (25% target, no lookahead), and compares against the actual S&P 500 over the same window (2020-08 → 2026-08):

```
                    Bot (risk-mgd)    Bot (raw)     S&P 500    BTC b&h
CAGR                         36.6%        55.6%       14.9%      32.1%
Volatility                   21.9%        29.2%       16.7%      57.3%
Sharpe                        1.53         1.66        0.92       0.77
Max drawdown                -25.4%       -32.0%      -25.4%     -76.6%
Growth of $1                  6.49        14.21        2.30       5.32
```

The risk-managed portfolio beats the S&P 500 on CAGR, Sharpe, and max drawdown simultaneously. Per-asset Sharpe ratios range 0.79–1.19 with different strategies winning on different assets — the diversification is doing real work, not one lucky bet.

**Why this is (reasonably) honest:** strategies are never scored on data used to pick them. Each yearly fold trains on all prior history, then trades the next 12 months unseen; the stitched out-of-sample record is what the table shows. Fees (0.1% per unit turnover) are charged throughout, plus 0.15% on risk-overlay adjustments. All positions are long-only and capped at 1x (no leverage). Caveats that remain: the asset universe is today's top-volume list (survivorship bias favors it), daily-close fills with no slippage, and one historical path is not a guarantee.

## Features

- **74-strategy candidate pool** — systematic grid over trend lookbacks × volatility targets, RSI dip-buy settings, MACD parameter sets, and ensembles
- **Multi-asset portfolio** — top Binance USDT pairs by quote volume, aligned on a shared fold calendar, equal-weighted
- **Risk overlay** — trailing 20-day vol targeting to equity-like 25% annualized volatility
- **Metrics engine** — CAGR, annualized volatility, Sharpe, max drawdown, exposure, turnover (calendar-day based so crypto's 365d year and equities' 252d year compare fairly)
- **No-lookahead engine** — weights see only past candles; prefix-sum indicator caching makes a 74-candidate × 7-asset walk-forward run in ~20 s
- **Disk cache** — daily history cached 12 h; warm runs skip all fetching
- **Benchmark comparison** — S&P 500 daily closes from FRED (Yahoo Finance fallback)
- **Live paper trading** — polls prices and simulates fills; state persists across restarts
- **CI** — GitHub Actions runs the full 1382-test suite on every push
- Stdlib only (Python 3.10+); pytest for tests

## Usage

```bash
# Full multi-asset walk-forward comparison vs the S&P 500
python -m bot compare

# Variations
python -m bot compare --assets 1                  # BTC only
python -m bot compare --assets 20                 # wider universe
python -m bot compare --portfolio-vol 0.20        # tighter risk target
python -m bot compare --train-days 730 --fee 0.0015

# Quick hourly SMA backtest (original toy strategy)
python -m bot backtest --symbol BTCUSDT --interval 1h

# Live paper trading — trendvol uses daily bars and hourly polls
python -m bot trade --strategy trendvol
python -m bot trade --strategy sma --interval 15m
```

## Project layout

```
bot/
  strategy.py    # 74-candidate strategy pool + prefix-sum indicator cache
  engine.py      # daily-bar backtest engine (no lookahead, fee-aware)
  metrics.py     # CAGR / Sharpe / vol / max drawdown
  walkforward.py # expanding-window walk-forward, absolute fold calendar
  universe.py    # top-volume Binance USDT pairs
  benchmark.py   # S&P 500 data (FRED primary, Yahoo fallback)
  data.py        # Binance candles with full-history pagination
  cache.py       # disk cache for daily history
  backtest.py    # legacy simple backtester (kept for the toy SMA path)
  paper.py       # paper broker + live loop
tests/           # 1382 unit/property tests
.github/         # CI workflow
```

## Running tests

```bash
pip install pytest
pytest
```
