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

## Research-methodology battery (`python -m bot research`)

A second, deeper battery (`bot/research.py`, `bot/clustering.py`, `bot/ablation.py`) that interrogates the *research process itself*:

- **Hansen's SPA test** — studentized Reality Check; less sensitive to one wild candidate dominating the pool
- **Strategy-family structure** — correlation clustering of all OOS streams, near-duplicate detection (|rho| >= 0.995), effective-trial-count estimate. On BTCUSDT the "85 candidates" collapse to ~22 families with 21 near-duplicate pairs
- **Portfolio-level DSR** — deflation applied to portfolio/overlay variants, not just single-asset picks
- **Strategy-family ablation** — omit one family at a time and re-run selection: negative delta = that family carries edge, positive = noise fit. On BTCUSDT removing TrendVol costs ~0.18 OOS Sharpe; nothing else matters much
- **Overlay ablation** — every fixed rule (inv-vol / tilt / crisis de-risk / vol targeting / throttle) toggled on/off so each headline claim is attributable to a mechanism
- **Bayesian Sharpe** — posterior over annualized Sharpe (Normal model, Jeffreys prior): credible intervals and P(Sharpe > 0)
- **Drawdown confidence intervals** — bootstrap bands for max drawdown plus time-under-water distributions
- **Sequence-risk testing** — rolling-window P(loss) for real entry dates vs shuffled orderings; forward block-bootstrap Monte Carlo preserving volatility clustering
- **Probability of underperformance** — paired-bootstrap probability the bot trails buy&hold on CAGR and Sharpe over the same window
- **Expanded bootstrap battery** — stationary, circular, and moving-block schemes reported side by side

## Execution-realism extensions

- **Cost models** (`bot/costs.py`): flat bps (default), volatility-dependent spread/slippage ((realized vol / reference)^0.5, clamped), square-root market impact `k * daily_vol * sqrt(participation)`, tiered maker/taker fees, and a dollar-volume liquidity filter — engine-integrated behind `CostParams`, off by default so existing results are unchanged
- **Calendar & data handling** (`bot/data.py`): exchange-calendar-aware portfolio alignment (an ETF held over a weekend is invested-flat, not cash; late-listed and delisted spans stay in cash), gap reports, tolerance-bounded forward-fill of small outages (filled bars flagged), delisting simulation with terminal liquidation cost

## Paper-trading reliability (`python -m bot trade`)

Rewritten around a persistent multi-asset paper portfolio:

- **Atomic state persistence** — temp-file + rename writes with a sha256 checksum; a crash mid-write can never corrupt balances
- **Append-only order ledger** — every fill records idempotency key, deltas, post-trade balances, fees, and the decision explanation
- **Crash recovery** — corrupt state files are rebuilt by replaying ledger deltas from start cash
- **Duplicate-order prevention** — decisions carry `(date|symbol|action|target)` keys; re-running a cycle can never double-fill
- **Data-staleness alerts** — symbols with frozen feeds get trading blocked for the cycle and land in the audit trail (also raised by `forward --step`)
- **Decision explanations & daily audit reports** — markdown under `reports/` with positions, fills, alerts, and why every decision was taken (holds included)

## Code identity: a freeze pins the implementation, not just numbers

The original freeze sealed configuration only — `freeze.json` recorded the
commit, but the scheduled runner checked out whatever `main` held *today*, so
editing `strategy.py` after freezing would silently change the experiment
while the manifest kept claiming otherwise. That hole is closed at three
levels:

1. **Algorithm seal.** `freeze.json["config"]["algorithm"]` now captures the
   COMPLETE portfolio construction — selection mode, candidate-pool version,
   universe rule, inverse-vol weighting (window + cap), XS-momentum tilt
   (lookback + max), crisis de-risking (window + threshold + multiplier),
   rebalance band, drawdown throttle state machine, vol-target overlay
   (target/window/fee) — every quantity that can affect a return, validated
   against unknown keys so a typo cannot silently become "default".
2. **Source seal.** `create_freeze` hashes the implementation itself
   (`bot/*.py` + `pyproject.toml`, LF-normalised so Windows trees and Linux
   checkouts of one commit hash identically; algorithm id `sha256-lf-v1` is
   recorded alongside the digest).
3. **Runner refusal.** `load_freeze` verifies all seals by default and
   `run_step` refuses to trade on mismatched code:
   ```
   CODE MISMATCH: the running implementation does not match the freeze.
     expected sha : 6c41…
     running sha  : 9d02…
   ```
4. **Frozen checkout in CI.** `.github/workflows/scheduled-paper.yml` reads
   the pointer from `freeze.json`, checks out `git_commit_at_freeze`
   (detached), runs `python -m bot verify-freeze` as a hard gate, trades one
   forward day on that code, then returns to main to append ONLY the log —
   data flows back; code never changes mid-experiment.

**Backtest/forward parity is tested, not assumed**
(`tests/test_algorithm_freeze.py::TestParity`): feeding identical daily asset
returns day-by-day through `run_step` reproduces `combine_portfolio_rule` +
vol-overlay exactly (|Δ| ≤ 1e-12) across six configurations — full headline,
no-tilt, no-crisis, throttle-on, overlay-off, zero-band. Both paths call the
same `day_allocation` function (`bot/portfolio_rules.py`), so they cannot
drift.

Artifact trail per freeze: config sha256 + algorithm sha256 + code sha256 +
annotated git tag (`freeze/<YYYYMMDD>`) + optional container image digest
(`--image-digest`; build with `docker build -t trading-bot .` and record
`docker images --digests`). Verify any time with
`python -m bot verify-freeze`. Freeze knobs mirror the backtest CLI:
`--band/--no-tilt/--tilt-lookback/--max-tilt/--no-crisis/--corr-window/
--corr-threshold/--derisk/--throttle/--dd-trigger/--dd-exit/
--throttle-factor/--vol-window/--max-multiple-of-equal/--fixed/--no-overlay`.

## Engineering quality

| Gate | Tool | Notes |
|---|---|---|
| Lint | ruff | config in `pyproject.toml` |
| Types | mypy | clean across all source modules |
| Tests + coverage | pytest + pytest-cov | 88% floor on library code |
| Property-style tests | seeded randomized invariants | `tests/test_properties.py` |
| Reproducible snapshots | `bot/snapshot.py` | pin data hashes + seed + metrics; verify drift |
| Environment | `Dockerfile` | quality gate by default; override for paper runs |
| Scheduled paper runs | `.github/workflows/scheduled-paper.yml` | daily forward step + committed log |

## AI commentary (optional, OpenRouter / NVIDIA)

An advisory-only AI layer (`bot/ai.py`) on top of the deterministic pipeline:

```bash
python -m bot ask "why is the deflated Sharpe the honest number?"   # grounded Q&A over live state
python -m bot trade --symbol BTCUSDT --once --ai-note               # adds an 'AI commentary' section
                                                                    # to today's audit report
```

- **Providers**: `nvidia` (`https://integrate.api.nvidia.com/v1`, default) and `openrouter` — both OpenAI-compatible; select with `AI_PROVIDER=openrouter`. Keys read from `NVIDIA_API_KEY` / `OPENROUTER_API_KEY` env vars or a gitignored `.env`; never hardcoded, never logged
- **Model allowlist**: only user-approved free models may be called, enforced before any network request; names resolve against the provider's live catalog (disk-cached 24h per provider)
- **No limits within the allowlist**: output tokens are uncapped by default (`max_tokens=None`), and when a model fails or hits its per-model rate limit the request rotates through *every* remaining allowlisted chat model — embeddings/rerank/TTS/safety endpoints are excluded from the chat rotation automatically
- **Advisory by construction**: model output is clearly labeled commentary appended after execution; no code path feeds it back into weights, signals, or decisions; every entry point degrades gracefully to "no commentary" without a key

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
python -m bot research                           # SPA, clustering/effective trials, ablation, Bayesian Sharpe

# Original toy path & live paper trading (paper only!)
python -m bot backtest --symbol BTCUSDT --interval 1h
python -m bot trade --strategy trendvol                 # persistent multi-asset paper loop
python -m bot trade --symbol BTCUSDT,ETHUSDT --once     # one cycle + daily audit report
```

## Project layout

```
bot/
  strategy.py    # 85-candidate strategy pool + prefix-sum indicator cache
  engine.py      # daily-bar engine: next-open execution, spread/slippage/
                 # latency, fee-on-turnover, cash accrual, rebalance banding,
                 # optional vol-dependent costs & square-root impact
  metrics.py     # CAGR / excess Sharpe / Sortino / Calmar / VaR / ES
  walkforward.py # expanding-window walk-forward + survivorship-safe combiners
  portfolio_rules.py # XS-momentum tilt, crisis de-risk, drawdown throttle
  regimes.py     # bull/bear/sideways segmentation + regime-conditioned stats
  sensitivity.py # parameter / rolling / cost / latency / execution sweeps
  stats_validation.py # PSR, DSR, block bootstrap, Reality Check, shuffle MC
  research.py    # SPA test, expanded bootstraps, drawdown CIs, Bayesian
                 # Sharpe, sequence risk, forward Monte Carlo
  clustering.py  # strategy-family clusters, near-duplicates, effective trials
  ablation.py    # strategy-family and portfolio-overlay ablations
  costs.py       # vol-dependent spread/slippage, market impact, fee tiers,
                 # liquidity filter
  data.py        # Binance + Yahoo fetchers, cleaning, gap handling, calendar
                 # alignment, delisting simulation
  prospective.py # freeze manifest, forward paper-trading log, checkpoints
  identity.py    # source fingerprint (sha256-lf-v1): freezes pin the code,
                 # not just the config; runner refuses on mismatch
  snapshot.py    # reproducible benchmark snapshots (data hashes + metrics)
  universe.py    # top-volume crypto + SPY/GLD/TLT cross-class ETFs
  benchmark.py   # S&P 500 data (FRED primary, Yahoo fallback)
  cache.py       # disk cache for daily history
  paper.py       # persistent multi-asset paper broker: ledger, idempotency,
                 # crash recovery, staleness alerts, audit reports
api/             # Vercel serverless endpoint (stdlib ASGI)
public/          # static dashboard served by Vercel
tests/           # ~1900 unit/property/integration tests
.github/         # CI quality gates + scheduled paper-run workflow
Dockerfile       # containerized environment; default CMD runs the full gate
```

## Running tests

```bash
pip install pytest ruff mypy pytest-cov
ruff check .        # lint
mypy                # type check
pytest --cov=bot    # tests + coverage gate (88% floor on library code)
```
