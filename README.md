# Trading Bot (Paper Trading)

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, Yahoo Finance, FRED — no API keys) and runs a research-grade validation pipeline: an 85-strategy candidate pool, multi-asset-class portfolio construction, walk-forward out-of-sample selection, realistic execution frictions, a full statistical-validation battery, prospective (frozen, forward) testing — and a **web dashboard deployable to Vercel in one command**.

> ⚠️ Educational software. Not financial advice. Past out-of-sample performance does not guarantee future results — do not wire real money to anything based on this repo.

## Web dashboard (Vercel)

The repo uses Vercel's zero-config layout — static assets in `public/`, serverless functions in `api/`:

```bash
npm i -g vercel   # once
vercel            # deploy from the repo root; accept defaults
```

- **`/`** — the dashboard: live BTC price and trend monitor (client-side, always works), plus the authoritative out-of-sample summary and equity curve from `/api/summary`. If the Python function is cold or unavailable, the page degrades gracefully to the client-side monitor.
- **`/api/summary`** — pure-stdlib ASGI function that imports the `bot` package and computes the fixed-rule walk-forward live (BTC, realistic frictions, ~4s cold). No third-party Python dependencies.

## Headline result (real data, out-of-sample, realistic frictions)

`python -m bot compare` trades a universe of top-volume crypto pairs **plus SPY, GLD, and TLT** (equity/gold/bonds), re-picks the best of 74 strategies per asset **every year using only prior data**, equal-weights the result with a fixed denominator, applies a trailing-volatility risk overlay (25% target, no lookahead), and compares against the actual S&P 500 over the same window (2020-08 → 2026-08). Defaults include **next-open execution, 10bp fee + 5bp spread + 5bp slippage per unit turnover, 3% cash yield on idle capital, excess-of-cash Sharpe everywhere**:

```
                     Bot inv-vol    Bot equal  Bot raw eq     S&P 500    BTC b&h
CAGR                       19.9%        26.9%       38.0%       14.9%      32.1%
Volatility                 15.5%        19.7%       23.7%       16.7%      57.3%
Sharpe (excess)             1.05         1.15        1.35        0.74       0.72
Max drawdown              -16.5%       -23.1%      -25.8%      -25.4%     -76.6%
Sortino                     1.82         1.79        2.13        1.06       1.07
Calmar                      1.21         1.16        1.47        0.59       0.42
ES 95% (1d)                -1.7%        -2.3%       -2.7%       -2.4%      -6.9%
Growth of $1                2.97         4.17        6.89        2.30       5.32
```

Two algorithms now report side by side: **equal-weight** (higher return) and **inverse-volatility weighted** (each asset's sleeve sized by 1/trailing-vol, capped at 2x equal weight) — the latter runs at 15.5% volatility with a -16.5% max drawdown, roughly two-thirds of the S&P's drawdown while still beating its CAGR. The candidate pool grew to 85 strategies with three new families: time-series momentum (`TSMom`), multi-horizon `DualMomentum`, and `RiskEnsemble`.

On top of the fixed rules, two a-priori portfolio overlays (`bot/portfolio_rules.py`) improve the diversified portfolio without any selection or tuning:

| Fixed rule (risk-managed) | CAGR | Sharpe | maxDD | ES95 | Calmar |
|---|---|---|---|---|---|
| inv-vol | 19.9% | 1.05 | -16.5% | -1.7% | 1.21 |
| inv-vol + XS-momentum tilt | 20.8% | 1.10 | -16.9% | -1.7% | 1.23 |
| inv-vol + tilt + crisis de-risk | **21.0%** | **1.12** | -16.9% | -1.7% | **1.24** |

The **cross-sectional momentum tilt** ranks assets by trailing 90d return and tilts sleeves within a ±50% band (gross exposure preserved); the **crisis de-risk** cuts exposure 40% when average pairwise correlation exceeds 0.6 (diversification breakdown). Both use strictly trailing data.

Two later refinements, measured honestly:

| Variant (risk-managed) | CAGR | Sharpe | maxDD | Verdict |
|---|---|---|---|---|
| inv-vol + tilt + crisis | 21.0% | 1.12 | -16.9% | previous best |
| **+ 5% rebalance band** (trade only when the weight moves >5%) | **22.0%** | **1.23** | -16.9% | **new best — pure cost reduction, DSR 1.000** |
| + drawdown throttle (halve exposure past -10% DD) | 16.5% | 0.94 | **-13.2%** | trades ~5pp CAGR for drawdown; available, not default |
| fully-fixed (RiskEnsemble on every asset, no selection anywhere) | 6.3% | 0.44 | -9.8% | the only *strictly* N=1 pipeline; much weaker — the per-asset selection genuinely adds value across 15 assets |

The honest statistical note: rows 1-2 sit on selection-based per-asset streams (the 85-trial caveat applies and is printed); only the last row is trial-count-1 end to end. The banded variant is the headline configuration.

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

**The fix the statistics point at (`validate` section 8):** running the a-priori `RiskEnsemble` — a fixed blend of trend, momentum, and dip-buying chosen *before* looking at anything, so the trial count is 1 — yields a lower raw Sharpe (0.51) but **DSR = 0.961**: it clears the statistical bar precisely because nothing was searched. The selected stream (Sharpe 1.03, DSR 0.138 across 85 trials) cannot make the same claim. The honest conclusion: the defensible edge is the fixed rule plus diversification, not the per-fold search — and the repo now ships both so you can watch which one the forward test (below) vindicates.

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
  strategy.py    # 85-candidate strategy pool + prefix-sum indicator cache
  engine.py      # daily-bar engine: next-open execution, spread/slippage/
                 # latency, fee-on-turnover, cash accrual, rebalance banding
  metrics.py     # CAGR / excess Sharpe / Sortino / Calmar / VaR / ES
  walkforward.py # expanding-window walk-forward + survivorship-safe combiners
  portfolio_rules.py # XS-momentum tilt, crisis de-risk, drawdown throttle
  regimes.py     # bull/bear/sideways segmentation + stress windows
  sensitivity.py # parameter / rolling / cost / latency / execution sweeps
  stats_validation.py # PSR, DSR, block bootstrap, Reality Check, shuffle MC
  prospective.py # freeze manifest, forward paper-trading log, checkpoints
  universe.py    # top-volume crypto + SPY/GLD/TLT cross-class ETFs
  benchmark.py   # S&P 500 data (FRED primary, Yahoo fallback)
  data.py        # Binance + Yahoo fetchers, cleaning, stale/delist detection
  cache.py       # disk cache for daily history
  paper.py       # paper broker + live loop
api/             # Vercel serverless endpoint (stdlib ASGI)
public/          # static dashboard served by Vercel
tests/           # 1667 unit/property tests
.github/         # CI workflow
```

## Running tests

```bash
pip install pytest
pytest
```
