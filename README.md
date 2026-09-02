# Trading Bot (Paper Trading) — a forward strategy-validation lab

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, Yahoo Finance, FRED — no API keys) and runs a research-grade validation pipeline: an 85-strategy candidate pool, multi-asset-class portfolio construction, walk-forward out-of-sample selection, realistic execution frictions, a full statistical-validation battery, prospective (frozen, forward) testing — and a **web dashboard deployable to Vercel in one command**.

## Evidence taxonomy (do not blur these)

| Label | Meaning | Where it lives |
|---|---|---|
| **BACKTEST** | Full-history research runs used to *search* for configurations. Counted in the research ledger; never evidence of edge on its own. | `validate`, `research`, ledger |
| **OUT-OF-SAMPLE** | Walk-forward folds inside the search window — still part of the selection process. | `compare`, canonical record |
| **FORWARD PAPER** | Days traded by the frozen runner after `freeze.json` was sealed. The only evidence that grows without the researcher touching anything. | `forward_log.jsonl`, dashboard hero |
| **LIVE** | Real money. **Does not exist and must not exist in this repo.** | — |

`python -m bot verdict` grades all five dimensions (historical evidence,
walk-forward robustness, selection-bias risk, cost robustness, prospective
forward evidence) into one auditable verdict with numeric inputs at every
tier — see `bot/verdict.py` for the exact thresholds. A negative forward
result is useful evidence; do not change the strategy to make historical
charts look better.

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

<!-- CANONICAL:BEGIN — generated from runs/canonical-v1/run.json; do not edit by hand -->
Out-of-sample window: 2020-08-16 → 2026-08-14 (6 yearly folds, 13 assets, point-in-time denominators).

```
                     Bot inv-vol    Bot equal  Bot raw eq     S&P 500    BTC b&h
--------------------------------------------------------------------------------
CAGR                       10.8%        25.7%       32.4%       14.9%      32.1%
Volatility                 11.2%        21.3%       24.6%       16.7%      57.3%
Sharpe (excess)             0.70         1.04        1.14        0.74       0.72
Max drawdown              -12.2%       -23.2%      -26.5%      -25.4%     -76.6%
Sortino                     1.25         1.58        1.76        1.06       1.07
Calmar                      0.88         1.11        1.22        0.59       0.42
ES 95% (1d)                -1.2%        -2.5%       -2.9%       -2.4%      -6.9%
Growth of $1                1.85         3.95        5.39        2.30       5.32
```

risk-managed portfolio OOS CAGR BEATS S&P 500 (25.7% vs 14.9%); Sharpe beats (1.04 vs 0.74); max drawdown better (-23.2% vs -25.4%)

**Fixed portfolio rules** (a-priori overlays; all risk-managed to 25% vol):

| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR |
|---|---|---|---|---|---|---|---|
| inv-vol (selected underlying) | 10.8% | 0.70 | -12.2% | -1.2% | 0.88 | 0.997 | 0.997 |
| + tilt + crisis de-risk | 10.2% | 0.71 | -11.6% | -1.2% | 0.87 | 0.995 | 0.995 |
| + drawdown throttle | 9.0% | 0.64 | -11.1% | -1.2% | 0.81 | 0.992 | 0.992 |
| + tilt + crisis, banded 5% rebalance | 10.0% | 0.75 | -9.7% | -1.1% | 1.02 | 0.997 | 0.997 |
| fully-fixed: RiskEnsemble everywhere, banded, all overlays | 5.2% | 0.38 | -7.6% | -0.7% | 0.68 | 0.985 | 0.985 |

*Provenance: reproduced from `canonical-v1/run.json` — commit `bc3102b06b720e5bc20c15e7752b485e698caad9`, code sha `e52484df5b44…`, strategy defs `0ac46902cd73…`, portfolio rules `ef09419e658a…`, universe `a7bd616f574a…`. Verify with `python -m bot reproduce canonical-v1` (frozen cache required).*

<!-- CANONICAL:END -->

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
| Survivorship bias | **Point-in-time eligibility** (`bot/universe_pit.py`): per-day denominators from listing age + trailing 30d dollar volume + alive-at-date — a 2023 listing only joins the 2023 portfolio, however famous it is today. **Forward snapshots** (`python -m bot universe-snapshot`, daily in CI) compound into a genuinely point-in-time dataset. Residual, disclosed: symbols purged from Binance's API before we ever fetched them remain invisible without an external archive |
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

**What it honestly finds on BTC (the weakest link, disclosed):** PSR = 0.998, but **DSR = 0.110** — after correcting for 74 trials, the single-asset Sharpe does not clear the conventional 0.95 bar. The Reality Check result is near the conventional 0.05 boundary and should not be read as decisive. Trimming 180 days off the start of the window cuts CAGR from 26.5% to 7.3%, so the 2020-21 regime drives much of the single-asset result. Nested selection degrades to picking buy-and-hold (inner purged folds with 220-day purges find no selection edge on one asset). The multi-asset portfolio result above is stronger — cross-asset diversification averages partly-independent bets — but the deflation finding stands as the repo's most important caveat: treat all headline numbers as regime-dependent research, not established alpha.

**The fix the statistics point at (`validate` section 8):** running the a-priori `RiskEnsemble` — a fixed blend of trend, momentum, and dip-buying chosen *before* looking at anything, so the trial count is 1 — yields a lower raw Sharpe (0.51) but **DSR = 0.961**: it clears the statistical bar precisely because nothing was searched. The selected stream (Sharpe 1.03, DSR 0.138 across 85 trials) cannot make the same claim. The honest conclusion: the defensible edge is the fixed rule plus diversification, not the per-fold search — and the repo now ships both so you can watch which one the forward test (below) vindicates.

## Robust selection (`bot/selection.py`) — and what it is actually worth

The deflation finding above is really a finding about *how the winner is
chosen*. `argmax` over in-sample Sharpe is the maximally overfit rule: it
picks whichever candidate got lucky in the training window, and it spends
the entire multiple-testing budget doing so. `bot/selection.py` implements
three alternatives and measures them.

**The rules**, each a drop-in replacement for `selection_fn` in
`walk_forward_at`:

- **one-SE** (Breiman, from CART pruning) — instead of taking the single
  best candidate, keep every candidate within one standard error of the
  winner's Sharpe (SE via Lo 2002, with skew/kurtosis correction), then
  choose among *those* by robustness rather than by raw performance.
  Candidates that are statistically indistinguishable are not meaningfully
  ranked, so preferring one on a 0.01 Sharpe edge is fitting noise.
- **minimax tie-break** — among the survivors, take the one whose *worst*
  contiguous sub-window Sharpe is best. This targets the repo's own
  documented failure directly: trimming 180 days cut single-asset CAGR
  from 26.5% to 7.3%, which is what a strategy that was brilliant in one
  regime and mediocre elsewhere looks like.
- **credibility shrinkage** — blend the pick with the a-priori prior at
  `alpha = gap / (gap + SE)`, where `gap` is how far the pick beat the
  prior and `SE` is the noise it had to beat it by. When the search found
  nothing beyond noise, `alpha -> 0` and the bot trades the prior.

**Two new strategy families** (`build_candidates(extended=True)`, 85 → 119
candidates), chosen to be *structurally* new rather than more lookbacks on
the same grid, because clustering showed the 85 collapse to ~22 families:

- `ChannelBreakout` — Donchian range position, `(close - lo) / (hi - lo)`.
  The only scale-invariant family: unchanged if every price is multiplied
  by a constant, where the SMA and momentum families carry the asset's
  price scale into the decision.
- `MeanReversionZ` — z-score dislocation, `(close - SMA) / stdev`, gated
  behind a long-term trend filter. The only family that structurally
  *buys weakness*: TrendVol/TSMom/DualMomentum all buy strength and
  RsiDipBuy only buys shallow pullbacks inside an uptrend.

The default pool is deliberately left at 85 so `candidate_pool_version` and
every existing backtest stay valid; the extended pool is opt-in.

**Measured** (`scripts/measure_selection.py`, 9 assets, walk-forward,
6 folds each, next-open execution, 10bp fee + 5bp spread + 5bp slippage,
3% cash yield, extended pool):

| rule | mean OOS Sharpe | median | mean CAGR | mean maxDD | mean DSR |
|---|---|---|---|---|---|
| `argmax` (current behaviour) | 0.681 | 0.622 | +27.6% | -38.0% | 0.202 |
| `one_se_worst` | 0.691 | 0.711 | +28.0% | -35.6% | 0.200 |
| `one_se_turnover` | 0.350 | 0.332 | +15.1% | -41.5% | 0.083 |
| `robust_shrunk` | 0.605 | 0.690 | +17.7% | **-24.6%** | 0.193 |

**The honest reading**, which is not uniformly flattering:

- `one_se_worst` beats `argmax` by **0.010 mean Sharpe** across 9 assets.
  That is a wash, far inside the noise of a 9-asset sample. It is not
  evidence that the rule adds return.
- `one_se_turnover` is **clearly worse** (-0.33 Sharpe, deeper drawdowns).
  The parsimony tie-break Breiman's original rule would pick does not
  transfer to this problem: here the cheap-to-trade candidates are cheap
  because they are barely invested.
- `robust_shrunk` **gives up** Sharpe (-0.076) and buys a large risk
  reduction: mean max drawdown **-38.0% → -24.6%**. That is the one
  result large enough to take seriously, and it is a risk result, not a
  return result.
- **DSR barely moves for any of them** (0.202 → 0.193-0.200). This is
  expected and is stated in the module docstring: shrinkage reduces the
  *variance* of the pick, not the *trial count*, and DSR is a function of
  the trial count. Any claim that these rules fix the deflation problem
  would be false.

The measurement is per-asset and unpaired. A portfolio-level comparison
with bootstrap confidence intervals is the obvious next step and is not
yet done, so none of the Sharpe differences above should be treated as
statistically established.

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

**Backtest/forward parity is tested, not assumed.** The flagship identity
test (`tests/test_backtest_forward_parity.py`) runs ONE fixed dataset two
ways — `run_strategy(...)` over full history vs freeze→`run_step` day by
day — and requires per-asset, per-day equality of target weights, held
weights, trades (count/size), costs, daily returns (incl. the
overnight/intraday split and cash accrual) plus portfolio equity, under
both a zero band and the 5% band (the banded case genuinely suppresses
trades: 22 vs 40). Tolerance is 1e-12; both paths share
`calculate_transition`, so they cannot drift. If this test fails, fix the
implementation — never loosen it.

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

## Research ledger: counting the TRUE number of experiments

An 85-candidate pool understates the real search. The project also explored
equal vs inverse-vol weighting, XS tilt on/off, crisis thresholds, a drawdown
throttle, band widths, execution models, cost models, universe rules... Every
one of those is a draw from the same multiple-testing lottery.

`research_ledger.jsonl` is the append-only record of every experiment —
hypothesis, full configuration, primary metric, numeric result,
accepted/rejected, git provenance — hash-chained so edits and deletions of
failed ideas are detectable (`python -m bot ledger` verifies and reports).
Backfilled entries (33 seeded from git history) carry `backfilled: true`
and their original commit.

```bash
python -m bot ledger        # counts by category; search total = the honest N
```

Current counts: **29 search experiments** (7 strategy families, 8 portfolio,
12 execution, 2 universe) + 4 methodology tools excluded from the search N.
`validate` now reports DSR twice: once against the 85-candidate pool, once
against the ledger-informed total — the latter is the honest number.
Freezes pin `research_context` (ledger entry count + sha256), so forward-test
corrections are computed against an immutable record of how much was searched.

## Reproducible runs: `run.json` + `python -m bot reproduce`

Every `compare` run now writes `runs/<id>/run.json` sealing the full context:

| Section | Contents |
|---|---|
| environment | python version, git commit, whole-code fingerprint (sha256-lf-v1), **strategy definitions hash**, **portfolio-rules hash**, **universe hash** |
| parameters | the complete invocation namespace (frictions, band, folds, vol target, …) |
| seeds | recorded explicitly; the pipeline is deterministic |
| datasets | per-asset sha256, provider, download timestamp, start/end — plus S&P 500 via cache |
| results | every metric block (equal/inv-vol/tilt/crisis/throttle/banded/fixed, S&P, BTC), picks, verdict |

```bash
python -m bot compare                 # saves runs/<id>/run.json automatically
python -m bot reproduce list          # enumerate saved runs
python -m bot reproduce <run-id>      # re-execute and verify identical metrics
```

`reproduce` REFUSES to run unless: (1) the running code fingerprint matches
the record, (2) each module seal matches, (3) every frozen dataset is still
in `.cache` with an identical sha256 — no silent refreshes. It then
re-executes deterministically and compares all stored metrics at 1e-12.
`PASS` means the number you quoted is the number you can regenerate.

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
  strategy.py    # 85-candidate strategy pool (+ 34 more via extended=True)
                 # and the prefix-sum / monotonic-deque indicator cache
  selection.py   # robustness-aware selection: one-SE rule, minimax
                 # sub-window tie-break, credibility shrinkage to a prior
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
