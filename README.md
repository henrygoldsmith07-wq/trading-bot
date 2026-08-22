# Trading Bot (Paper Trading)

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, Yahoo Finance, FRED — no API keys) and runs a research-grade validation pipeline: a 74-strategy candidate pool, a multi-asset-class portfolio, walk-forward out-of-sample selection, realistic execution frictions, and regime/stress/sensitivity diagnostics against the S&P 500.

> ⚠️ Educational software. Not financial advice. Past out-of-sample performance does not guarantee future results — do not wire real money to anything based on this repo.

## Headline result (real data, out-of-sample, realistic frictions)

`python -m bot compare` trades a universe of top-volume crypto pairs **plus SPY, GLD, and TLT** (equity/gold/bonds), re-picks the best of 74 strategies per asset **every year using only prior data**, equal-weights the result with a fixed denominator, applies a trailing-volatility risk overlay (25% target, no lookahead), and compares against the actual S&P 500 over the same window (2020-08 → 2026-08). Defaults include **next-open execution, 10bp fee + 5bp spread + 5bp slippage per unit turnover, 3% cash yield on idle capital, excess-of-cash Sharpe everywhere**:

```
                    Bot (risk-mgd)    Bot (raw)     S&P 500    BTC b&h
CAGR                         27.7%        38.2%       14.9%      32.1%
Volatility                   19.5%        24.0%       16.7%      57.3%
Sharpe (excess)               1.20         1.34        0.74       0.72
Max drawdown                -20.0%       -23.6%      -25.4%     -76.6%
Growth of $1                  4.33         6.96        2.30       5.32
```

Beats the S&P 500 on CAGR, excess Sharpe, and max drawdown simultaneously — with the frictions of real trading priced in.

## Backtesting-quality checklist

How the pipeline addresses each dimension:

| Dimension | Implementation |
|---|---|
| More assets | 20 crypto pairs by volume (+3 ETFs); assets with <3y history or stale/delisted data are skipped with reasons |
| More asset classes | Crypto (Binance) + SPY/GLD/TLT (Yahoo), each with its own trading calendar (365 vs 252 periods/yr) |
| Market regimes | `compare` prints a bull/bear/sideways table (BTC-proxy trailing 180d labels) with bot vs S&P per segment |
| Stress periods | 30d-crash / top-decile-vol stress windows with bot vs S&P drawdowns over the stressed span |
| Parameter sensitivity | `sensitivity` sweeps a TrendVol grid: OOS Sharpe by lookback × vol-target |
| Rolling sensitivity | Consecutive 2-year OOS blocks — profitability must hold in every block, not one lucky window |
| Transaction costs | Combined fee+spread+slippage sweep (5/10/20/50/100bp) |
| Spread & slippage | Charged per unit turnover alongside fees in the engine |
| Latency | Signal delay sweep (0/1/2 days) — weight uses only data ≥ latency days old |
| Execution realism | `next_open` mode: overnight gap accrues to yesterday's position, intraday to the new one |
| Stale prices | Assets whose history stopped >45d ago are flagged and excluded |
| Missing data | Non-finite/non-positive closes dropped, timestamps deduped, history sorted on ingest |
| Delisting | Missing days hold cash — a dead asset strands its sleeve, it never redistributes to survivors |
| Survivorship bias | Fixed-denominator portfolio combine + explicit disclosure (point-in-time constituents aren't freely available) |
| Cash returns | Idle cash accrues a configurable risk-free rate (default 3%/yr); all Sharpe ratios are excess-of-cash |
| Benchmark consistency | Same window, calendar-day CAGR, same risk-free rate; the index carries no costs and is labeled as such |

## What the sensitivity analysis says (BTC, honest read)

- **Rolling blocks**: positive OOS Sharpe in every 2-year block (1.16 / 1.38 / 0.76) — not one lucky window.
- **Costs**: Sharpe degrades gracefully from 1.23 at 5bp to 0.84 at 100bp total friction.
- **Latency**: robust to 1-2 day delays (results within noise of zero-latency).
- **Parameters**: edge concentrates at short lookbacks (25-75d) and decays to ~0 at 200d — a real fragility, disclosed rather than hidden.
- **Execution**: close vs next-open are identical on crypto (Binance daily opens equal prior closes — 24h market); gaps matter only for the ETF sleeves.

## Statistical validation (`python -m bot validate`)

Runs the full inferential battery on the walk-forward record (`bot/stats_validation.py`, stdlib only):

- **Nested walk-forward selection** — an inner purged CV inside each training window picks the strategy, separating selection overfitting from trading edge
- **Purged cross-validation + embargo** — inner folds are purged by 220 days (max indicator lookback) and every training window is embargoed 30 days before its test
- **Probabilistic Sharpe (PSR)** and **Deflated Sharpe (DSR)** — deflated using the true trial count (74) and the observed variance of trial Sharpes recorded during selection
- **White's Reality Check** — block-bootstrap max-across-all-74-strategies null test on aligned OOS streams
- **Stationary block bootstrap** — 90% CIs for CAGR/Sharpe plus the max-drawdown distribution (median / p95 / worst)
- **Monte Carlo trade-order resampling** — drawdown of the actual return sequence vs shuffled orderings
- **Start/end-date sensitivity**, **parameter-stability scoring**, and the distribution/tail panel: skew, kurtosis, VaR (descriptive), expected shortfall, Sortino, Calmar, exposure, turnover

**What it honestly finds on BTC (the weakest link, disclosed):** PSR = 0.998, but **DSR = 0.110** — after correcting for 74 trials, the single-asset Sharpe does not clear the conventional 0.95 bar. The Reality Check p-value lands at 0.050, exactly on the boundary. Trimming 180 days off the start of the window cuts CAGR from 26.5% to 7.3%, so the 2020-21 regime drives much of the single-asset result. Nested selection degrades to picking buy-and-hold (inner purged folds with 220-day purges find no selection edge on one asset). The multi-asset portfolio result above is stronger — cross-asset diversification averages partly-independent bets — but the deflation finding stands as the repo's most important caveat: treat all headline numbers as regime-dependent research, not established alpha.

## Usage

```bash
# Multi-asset-class walk-forward comparison vs the S&P 500 (realistic defaults)
python -m bot compare

# Variations
python -m bot compare --assets 40                # wider crypto universe
python -m bot compare --execution close          # optimistic fill model
python -m bot compare --fee 0.002 --slippage-bps 20 --latency-days 1
python -m bot sensitivity                        # all robustness sweeps on BTC
python -m bot validate                           # full statistical battery on BTC
python -m bot validate --symbol ETHUSDT --rc-boots 250

# Original toy path & live paper trading (paper only!)
python -m bot backtest --symbol BTCUSDT --interval 1h
python -m bot trade --strategy trendvol
```

## Project layout

```
bot/
  strategy.py    # 74-candidate strategy pool + prefix-sum indicator cache
  engine.py      # daily-bar engine: next-open execution, spread/slippage/
                 # latency, fee-on-turnover, cash accrual, no lookahead
  metrics.py     # CAGR / excess Sharpe / vol / max drawdown
  walkforward.py # expanding-window walk-forward + survivorship-safe combine
  regime.py     # bull/bear/sideways segmentation + stress windows
  sensitivity.py # parameter / rolling / cost / latency / execution sweeps
  stats_validation.py # PSR, DSR, block bootstrap, Reality Check, shuffle MC
  universe.py    # top-volume crypto + SPY/GLD/TLT cross-class ETFs
  benchmark.py   # S&P 500 data (FRED primary, Yahoo fallback)
  data.py        # Binance + Yahoo fetchers, cleaning, stale/delist detection
  cache.py       # disk cache for daily history
  paper.py       # paper broker + live loop
tests/           # 1425 unit/property tests
.github/         # CI workflow
```

## Running tests

```bash
pip install pytest
pytest
```
