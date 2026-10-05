# Trading Bot (Paper Trading) — a forward strategy-validation lab

A Python trading bot that trades **on paper only** — no real orders are ever placed. It pulls free public market data (Binance, Yahoo Finance, FRED — no API keys) and runs a research-grade validation pipeline: an 85-strategy candidate pool, multi-asset-class portfolio construction, walk-forward out-of-sample selection, realistic execution frictions, a full statistical-validation battery, prospective (frozen, forward) testing — and a **web dashboard deployable to Vercel in one command**.

## 1. What this project is

A measurement instrument, not a product. Its job is to answer one question honestly: **how much evidence actually supports one specific strategy?** It does that by binding one strategy identity to one experiment manifest, one evidence record and one verdict — and rendering every surface (README, dashboard, CLI, API, generated reports) from that single object, so no surface can accidentally claim a comparator's result.

Nothing here is a live trading system. There are no deposits, no broker connectivity, no API-key trading flows, and no code path that places a real order.

### Evidence taxonomy (do not blur these)

| Label | Meaning | Where it lives |
|---|---|---|
| **BACKTEST** | Full-history research runs used to *search* for configurations. Counted in the research ledger; never evidence of edge on its own. | `validate`, `research`, ledger |
| **OUT-OF-SAMPLE** | Walk-forward folds inside the search window — still part of the selection process. | `compare`, canonical record |
| **FORWARD PAPER (v1 — SUPERSEDED)** | Days traded by the frozen runner after the 2026-09-06 freeze. Those days were produced by execution accounting that summed the overnight and intraday legs instead of evolving wealth exactly, so returns were understated on gapping bars. The tape is preserved unchanged as evidence of *that* implementation; it is **not** prospective validation of the current one. | `forward_log.jsonl`, `freeze.json` |
| **FORWARD PAPER (v2 — current)** | Days traded by a freeze created under the corrected accounting model (`exact-wealth-v2`). The only tape that is prospective evidence of the code now in the repo. Not yet created. | — |
| **LIVE** | Real money. **Does not exist and must not exist in this repo.** | — |

## 2. Current experiment status

The experiment state is derived automatically from the evidence — never assigned, never inferred from whichever file is nearest. One state machine (`bot/experiment_state.py`) produces the word every surface prints:

| State | Meaning |
|---|---|
| `RESEARCH_ONLY` | No canonical evidence record yet, so nothing is being claimed. |
| `CANONICAL_READY` | Canonical research evidence exists but nothing is frozen. |
| `READY_TO_FREEZE` | A freeze exists but its source seal does not verify; it cannot be traded or graded. |
| `FORWARD_PRELIMINARY` | A valid current freeze with fewer than 30 clean forward days: a sanity check, not validation. |
| `FORWARD_MEANINGFUL` | ≥ 30 clean forward days from the CURRENT experiment; prospective evidence is the headline. |
| `SUPERSEDED` | The freeze is intact but its methodology is no longer current; its tape is evidence of what produced it, not of the code now running. |
| `INVALIDATED` | Provenance or the experiment seal is broken; no claim may stand on this evidence. |

A superseded experiment never contributes its forward days to the current experiment's count.

```bash
python -m bot experiment-status   # state, clean forward days, checkpoints, biggest caveat
python -m bot evidence            # the full evidence document as a report (or --json)
```

**Forward checkpoints.** The system measures the frozen experiment; it does not encourage reacting to a short poor run.

| Clean forward days | What it licenses |
|---|---|
| 10 | sanity check only |
| 30 | preliminary evidence |
| 60 | early evaluation |
| 90 | meaningful first review |
| 180 | stronger assessment |
| 365 | full annual regime sample |

## 3. Current evidence

Everything below this line is rendered from the evidence document (`python -m bot evidence`); the block regenerates from it and `--check` fails in CI when the README drifts.

<!-- CANONICAL:BEGIN — generated from the canonical-v2 evidence document; do not edit by hand -->
### At a glance (rendered from the evidence document)

- **What is being tested?** `banded_rm` — inverse-vol weighting + cross-sectional momentum tilt + crisis de-risk + 5% rebalance band + volatility overlay.
- **Current valid freeze?** yes; the frozen methodology is SUPERSEDED by exact-wealth-v2, so its tape is not prospective evidence of the current code
- **Current forward days?** 0 clean day(s) (recorded 0, excluded 11 from other experiments).
- **Beats the benchmark?** no — primary rule [banded_rm] OOS CAGR TRAILS S&P 500 (11.7% vs 14.9%); Sharpe beats (0.74 vs 0.74); max drawdown better (-9.8% vs -25.4%).
- **Biggest caveat?** the frozen experiment is SUPERSEDED by exact-wealth-v2: its tape is evidence of the implementation that produced it, not of the code now running.

**Canonical rule under evaluation: `banded_rm`** — inverse-vol weighting + cross-sectional momentum tilt + crisis de-risk + 5% rebalance band + volatility overlay. This is the strategy the freeze executes and the forward log grades.

**Experiment state: `SUPERSEDED`** — The frozen experiment is intact but its methodology is no longer current; its tape is evidence of the implementation that produced it, not of the code now running.

| Metric | PRIMARY `banded_rm` | S&P 500 |
|---|---|---|
| OOS CAGR | 11.7% | 14.9% |
| Sharpe | 0.74 | 0.74 |
| Max drawdown | -9.8% | -25.4% |

> primary rule [banded_rm] OOS CAGR TRAILS S&P 500 (11.7% vs 14.9%); Sharpe beats (0.74 vs 0.74); max drawdown better (-9.8% vs -25.4%)

Out-of-sample window: 2020-08-16 → 2026-08-14 (6 yearly folds, point-in-time denominators).

**Current forward evidence: 0 clean day(s).** 0 clean forward day(s) is below the 30-day threshold; this is a sanity check, not validation

> The comparator columns below are context only. They are not the frozen strategy and are never the claim.

```
                         PRIMARY      BTC b&h   cmp raw eq    cmp equal    cmp fixed  cmp inv-vol cmp throttle     S&P 500
--------------------------------------------------------------------------------------------------------------------------
CAGR                       11.7%        32.1%        32.9%        25.4%         5.2%        11.9%        10.4%        14.9%
Sharpe (excess)             0.74         0.72         1.18         1.05         0.36         0.77         0.64         0.74
Max drawdown               -9.8%       -76.6%       -26.5%       -23.8%        -8.1%       -12.4%       -11.7%       -25.4%
Volatility                 11.7%        57.3%        24.0%        20.8%         6.4%        11.6%        11.7%        16.7%
Sortino                     1.37         1.07         1.81         1.58         0.51         1.40         1.13         1.06
Calmar                      1.19         0.42         1.24         1.07         0.65         0.96         0.89         0.59
ES 95% (1d)                -1.2%        -6.9%        -2.8%        -2.5%        -0.8%        -1.2%        -1.3%        -2.4%
Growth of $1                1.94         5.32         5.50         3.88         1.36         1.96         1.81         2.30
```

Statistical standing: PSR 0.999 — DSR n/a (effective trials 85 via max_of_recorded_sources).
> DSR withheld: record row predates trial accounting; the stored value corrected for nothing and is withheld

**Fixed portfolio rules** (a-priori overlays; all risk-managed to 25% vol):

| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR | Trials |
|---|---|---|---|---|---|---|---|---|
| inv-vol (selected underlying) | 11.9% | 0.77 | -12.4% | -1.2% | 0.96 | 0.999 | n/a | — |
| + tilt + crisis de-risk | 11.7% | 0.71 | -12.0% | -1.4% | 0.97 | 0.996 | n/a | — |
| + drawdown throttle | 10.4% | 0.64 | -11.7% | -1.3% | 0.89 | 0.995 | n/a | — |
| + tilt + crisis, banded 5% rebalance | 11.7% | 0.74 | -9.8% | -1.2% | 1.19 | 0.999 | n/a | — |
| fully-fixed: RiskEnsemble everywhere, banded, all overlays | 5.2% | 0.36 | -8.1% | -0.8% | 0.65 | 0.976 | n/a | — |

*Provenance: rendered from the `canonical-v2` evidence document (evidence fingerprint `733820238740`, experiment `d4338a04d323`, commit `ab8aea1c95a3d4baaed7029925dbf88f62714cb0`, code sha `70cfa12e0a74…`, strategy defs `4cfdf950c0c0…`, portfolio rules `3d1a4f560ade…`, universe `f550645ddd97…`). Reproduce with `python -m bot reproduce canonical-v2` and verify with `python -m bot verify-evidence`.*

<!-- CANONICAL:END -->

**What counts as a forward day.** Each scheduled day splits by how much of the portfolio was actually observed:

| | meaning |
|---|---|
| `days_full` | every sleeve printed — worth one unit of evidence, and the only number the hero shows |
| `days_partial` | some sleeve was in outage or held `session_pending`; recorded, but not counted |
| `days_dark` | no sleeve printed — zero information, and graded as none |
| `days_closed` | a genuine weekend, regular NYSE holiday, or explicitly closed session (`session_closed`) — disclosed separately, neither evidence nor an outage |

`data_outages` in the payload counts **asset-day events**; `data_outage_days` counts **days**. They differ by up to the number of assets, and grading consumes the second — feeding it the first made the outage ratio meaningless.

The consequence is deliberate: when a data source is unreachable the **evidence count** stops, rather than climbing on days where nothing was measured. Performance accounting is a separate problem: the append-only tape's `port_ret` rows are mark-chained intervals, so the cumulative return/equity curve compounds every recorded interval rather than deleting partial rows and accidentally deleting real crypto price movement. If a real partial/dark interval exists, the return is marked degraded and forward Sharpe is withheld until the tape is clean enough for that inference.

## 4. Main caveats

Read these before quoting any number above.

1. **The forward sample is empty (or tiny).** Right now the strategy is judged on historical walk-forward evidence produced under selection, and selection flatters. Nothing on this page is prospective validation until the current freeze accumulates clean forward days.
2. **The current freeze is superseded.** The 2026-09-06 experiment ran under the additive-legs accounting model. Its 11 recorded days are honest evidence of *that* implementation and are deliberately excluded from the current experiment's count.
3. **Deflation is the central finding.** After correcting for the true number of trials searched, the single-asset Sharpe does not clear the conventional 0.95 bar (DSR = 0.110 on BTC against 74 trials). The selected stream (Sharpe 1.03, DSR 0.138 across 85 trials) cannot claim edge; the un-searched fixed rule reports DSR 0.961 precisely because nothing was searched. See [Statistical validation](#how-the-methodology-works).
4. **Regime dependence is real.** Trimming 180 days off the start of the window cuts single-asset CAGR from 26.5% to 7.3% — one regime carries much of the historical result. Weak folds are published, not hidden.
5. **Survivorship bias is reduced, not removed.** Point-in-time eligibility is a genuine membership mask, but the seed symbol list is still today's Binance ranking: assets purged before the first fetch stay invisible. Disclosed, never reconstructed.
6. **Comparators are not the claim.** Equal-weight and raw-equal-weight variants beat the index in the historical record. They are context only: the claim is the frozen `banded_rm` rule, which trails the S&P 500 on CAGR.

The single recommended reading is the deflated-Sharpe caveat in the statistical-validation section: the search-corrected number is the one that decides whether anything here is real.

## 5. How the methodology works

### Walk-forward pipeline

`python -m bot compare` trades a universe of top-volume crypto pairs **plus SPY, GLD, and TLT** (equity/gold/bonds), re-picks the best of 85 strategies per asset **every year using only prior data**, applies the frozen portfolio rule (inverse-vol weighting + cross-sectional momentum tilt + crisis de-risking + 5% rebalance band + a trailing-volatility overlay, 25% target, no lookahead), and compares against the actual S&P 500 over the same window. Defaults include **next-open execution, 10bp fee + 5bp spread + 5bp slippage per unit turnover, 3% cash yield on idle capital, excess-of-cash Sharpe everywhere**.

### Backtesting-quality checklist

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
| Execution realism | `next_open` mode: overnight gap accrues to yesterday's position, intraday to the new one. The session return is the **exact** evolution of portfolio wealth, not the sum of the two legs (`bot/execution.py`), and turnover is measured on the allocation *drifted* to the execution price |
| Stale prices | Assets whose history stopped >45d ago are flagged and excluded |
| Missing data | Non-finite/non-positive closes dropped, timestamps deduped, history sorted on ingest |
| Delisting | Missing days hold cash — a dead asset strands its sleeve, it never redistributes to survivors |
| Survivorship bias | **Point-in-time eligibility as a real membership mask** (`bot/universe_pit.py`): an asset not eligible on a date is removed from that day's contributors entirely, so it earns no return and carries no weight until it cleared listing age + trailing 30d dollar volume + alive-at-date. Unavailable sleeves hold cash. **Residual, disclosed:** the *seed* symbol list is still today's Binance ranking, so assets purged from the API before we first fetched them stay invisible — this reduces survivorship bias, it does not remove it. **Forward snapshots** (`python -m bot universe-snapshot`, daily in CI) are observed rather than reconstructed, and feed a genuinely historical universe (`eligibility_from_snapshots`) |
| Order-lifecycle provenance | `signal ≤ intent ≤ submitted ≤ fill`, all timezone-aware. The committed v1 tape has a real unit-error bug (`intent_ts` ~2.7 years in the future); it is detected by `bot/lifecycle.py`, excluded from latency/slippage statistics, and left byte-for-byte intact |
| Cash returns | Idle cash accrues a configurable risk-free rate (default 3%/yr); all Sharpe ratios are excess-of-cash |
| Benchmark consistency | Same window, calendar-day CAGR, same risk-free rate; the index carries no costs and is labeled as such |

### Statistical validation (`python -m bot validate`)

Runs the full inferential battery on the walk-forward record (`bot/stats_validation.py`, stdlib only):

- **Nested walk-forward selection** — an inner purged CV inside each training window picks the strategy, separating selection overfitting from trading edge
- **Purged cross-validation + embargo** — inner folds are purged by 220 days (max indicator lookback) and every training window is embargoed 30 days before its test
- **Probabilistic Sharpe (PSR)** and **Deflated Sharpe (DSR)** — deflated using the true trial count (85 for the pool; see the search-accounting block in the evidence document) and the observed variance of trial Sharpes recorded during selection
- **White's Reality Check** — block-bootstrap max-across-all-candidates null test on aligned OOS streams
- **Stationary block bootstrap** — 90% CIs for CAGR/Sharpe plus the max-drawdown distribution (median / p95 / worst)
- **Monte Carlo trade-order resampling** — drawdown of the actual return sequence vs shuffled orderings
- **Start/end-date sensitivity**, **parameter-stability scoring**, and the distribution/tail panel: skew, kurtosis, VaR (descriptive), expected shortfall, Sortino, Calmar, exposure, turnover

**What it honestly finds on BTC (the weakest link, disclosed):** PSR = 0.998, but **DSR = 0.110** — after correcting for 74 trials, the single-asset Sharpe does not clear the conventional 0.95 bar. The Reality Check result is near the conventional 0.05 boundary and should not be read as decisive. Trimming 180 days off the start of the window cuts CAGR from 26.5% to 7.3%, so the 2020-21 regime drives much of the single-asset result. Nested selection degrades to picking buy-and-hold (inner purged folds with 220-day purges find no selection edge on one asset). The multi-asset portfolio result is stronger — cross-asset diversification averages partly-independent bets — but the deflation finding stands as the repo's most important caveat: treat all headline numbers as regime-dependent research, not established alpha.

**The fix the statistics point at (`validate` section 8):** running the a-priori `RiskEnsemble` — a fixed blend of trend, momentum, and dip-buying chosen *before* looking at anything, so the trial count is 1 — yields a lower raw Sharpe (0.51) but **DSR = 0.961**: it clears the statistical bar precisely because nothing was searched. The selected stream (Sharpe 1.03, DSR 0.138 across 85 trials) cannot make the same claim. The honest conclusion: the defensible edge is the fixed rule plus diversification, not the per-fold search — and the repo now ships both so you can watch which one the forward test vindicates.

**Multiple-testing accounting is explicit.** `bot/search_accounting.py` is the single record of what was searched: candidate pool version, strategies generated/evaluated, walk-forward selections, research-ledger experiments, and the **effective trial count with its reasoning**. A rule produced by selection over a pool is never called `N=1` because the final rule happens to be fixed; when the trial count cannot be established, DSR is reported as **unavailable with a reason**, never as a number that corrects for nothing. The canonical record, the statistical pipeline and the evidence document all read this one structure.

**How the search factors compound is disclosed, not hidden.** The applied trial count is the **max** of the recorded sources — the pool (85) currently dominates the fold-by-fold picks and the ledger. But those sources are not competing estimates of one number: a per-fold, per-asset walk-forward selection *re-runs* the pool on each of ~6 folds for each of ~13 sleeves, and a cross-sectional Sharpe chosen from 85 candidates and then re-chosen 78 times is a wider search than any one factor alone. `breadth_disclosure()` therefore publishes the dimensions and a **naive compounded** reading next to the applied `effective_trials_used`. It is a disclosure, not a recomputation — the exact effective count is a judgement about fold/family correlation that the repo's clustering-based effective-trial estimate addresses, and this deliberately moves no published DSR number.

### Robust selection (`bot/selection.py`) — and what it is actually worth

The deflation finding above is really a finding about *how the winner is chosen*. `argmax` over in-sample Sharpe is the maximally overfit rule: it picks whichever candidate got lucky in the training window, and it spends the entire multiple-testing budget doing so. `bot/selection.py` implements three alternatives and measures them.

**The rules**, each a drop-in replacement for `selection_fn` in `walk_forward_at`:

- **one-SE** (Breiman, from CART pruning) — instead of taking the single best candidate, keep every candidate within one standard error of the winner's Sharpe (SE via Lo 2002, with skew/kurtosis correction), then choose among *those* by robustness rather than by raw performance. Candidates that are statistically indistinguishable are not meaningfully ranked, so preferring one on a 0.01 Sharpe edge is fitting noise.
- **minimax tie-break** — among the survivors, take the one whose *worst* contiguous sub-window Sharpe is best. This targets the repo's own documented failure directly: trimming 180 days cut single-asset CAGR from 26.5% to 7.3%, which is what a strategy that was brilliant in one regime and mediocre elsewhere looks like.
- **credibility shrinkage** — blend the pick with the a-priori prior at `alpha = gap / (gap + SE)`, where `gap` is how far the pick beat the prior and `SE` is the noise it had to beat it by. When the search found nothing beyond noise, `alpha -> 0` and the bot trades the prior.

**Two new strategy families** (`build_candidates(extended=True)`, 85 → 119 candidates), chosen to be *structurally* new rather than more lookbacks on the same grid, because clustering showed the 85 collapse to ~22 families:

- `ChannelBreakout` — Donchian range position, `(close - lo) / (hi - lo)`. The only scale-invariant family: unchanged if every price is multiplied by a constant, where the SMA and momentum families carry the asset's price scale into the decision.
- `MeanReversionZ` — z-score dislocation, `(close - SMA) / stdev`, gated behind a long-term trend filter. The only family that structurally *buys weakness*: TrendVol/TSMom/DualMomentum all buy strength and RsiDipBuy only buys shallow pullbacks inside an uptrend.

The default pool is deliberately left at 85 so `candidate_pool_version` and every existing backtest stay valid; the extended pool is opt-in.

**Measured** (`scripts/measure_selection.py`, 9 assets, walk-forward, 6 folds each, next-open execution, 10bp fee + 5bp spread + 5bp slippage, 3% cash yield, extended pool):

| rule | mean OOS Sharpe | median | mean CAGR | mean maxDD | mean DSR |
|---|---|---|---|---|---|
| `argmax` (current behaviour) | 0.681 | 0.622 | +27.6% | -38.0% | 0.202 |
| `one_se_worst` | 0.691 | 0.711 | +28.0% | -35.6% | 0.200 |
| `one_se_turnover` | 0.350 | 0.332 | +15.1% | -41.5% | 0.083 |
| `robust_shrunk` | 0.605 | 0.690 | +17.7% | **-24.6%** | 0.193 |

**The honest reading**, which is not uniformly flattering:

- `one_se_worst` beats `argmax` by **0.010 mean Sharpe** across 9 assets. That is a wash, far inside the noise of a 9-asset sample. It is not evidence that the rule adds return.
- `one_se_turnover` is **clearly worse** (-0.33 Sharpe, deeper drawdowns). The parsimony tie-break Breiman's original rule would pick does not transfer to this problem: here the cheap-to-trade candidates are cheap because they are barely invested.
- `robust_shrunk` **gives up** Sharpe (-0.076) and buys a large risk reduction: mean max drawdown **-38.0% → -24.6%**. That is the one result large enough to take seriously, and it is a risk result, not a return result.
- **DSR barely moves for any of them** (0.202 → 0.193-0.200). This is expected and is stated in the module docstring: shrinkage reduces the *variance* of the pick, not the *trial count*, and DSR is a function of the trial count. Any claim that these rules fix the deflation problem would be false.

The measurement is per-asset and unpaired. A portfolio-level comparison with bootstrap confidence intervals is the obvious next step and is not yet done, so none of the Sharpe differences above should be treated as statistically established.

### Pre-analysis plans: deciding what counts BEFORE seeing the result (`bot/registration.py`)

Every other control in this repository is retrospective. The research ledger records experiments after they ran; the freeze pins *identity*, not a *decision rule*; the deflated Sharpe corrects for the trial count once the winner is known. All three are real, and all three sit downstream of the same unprotected decision: **who decided, in advance, what would count as success, and which arm would be the claim?**

That gap is the largest remaining source of false discovery here, because the freedoms it protects are exercised unconsciously:

- **choosing the flattering metric** once several are available — Sharpe, CAGR, Calmar and ES rank the five portfolio rules differently;
- **choosing the threshold** once the number is known ("30 clean days", and not 60, and not 12);
- **choosing which arm is primary** after seeing which arm wins — this repo grades five portfolio rules and names one of them primary;
- **reading a forward tape repeatedly** until one reading looks good, which is the multiple-comparisons problem applied to *time* rather than to arms;
- **stopping a losing forward run** early and restarting, converting one falsifiable prediction into an unbounded sequence of them.

The DSR cannot see any of this: these are not trials of a strategy, they are trials of the **decision procedure**, and they happen after the trial count has been fixed. `bot/registration.py` closes them by fixing the decision procedure first and refusing to grade against a plan that moved.

A registration declares the hypothesis, **one** primary metric, the direction, the numeric threshold, the minimum evidence required before the plan may be read at all, every competing arm, the expected search breadth, and the stopping rule. It is content-addressed: editing any field changes the seal, and a result may only be graded against a seal that still matches.

```bash
python -m bot register \
  --id banded-rm-validity \
  --title "Does the frozen banded_rm rule clear its registered Sharpe floor?" \
  --hypothesis "banded_rm earns an OOS Sharpe above 0.8 on the frozen universe" \
  --metric oos_sharpe --direction gt --threshold 0.8 \
  --min-evidence 30 --alpha 0.05 \
  --arm "banded_rm:frozen primary rule:primary" \
  --arm "inv_vol:comparator overlay rule" \
  --arm "buy_hold:equal-weight over the same window" \
  --stopping "read once at 30 clean forward days; do not re-read" \
  --window "2026-09-06 -> +30d"

python -m bot registration-status    # list sealed plans; whether each has been read
```

Reading fails closed, and the order of the checks is the argument:

| State | Meaning |
|---|---|
| `UNREGISTERED` | No plan covers this result. Exploratory; never citable as confirmation. |
| `DRIFTED` | The plan's seal no longer matches what the read was set up against. A **new study**, not a result. |
| `ALREADY_READ` | The plan has already been read. A second read is a search over readings. |
| `INCONCLUSIVE` | Fewer observations than the registered minimum. Honest — and **not** a pass. |
| `CONFIRMED` | Cleared the registered criterion on sufficient evidence, surviving family-wise correction. |
| `NOT_CONFIRMED` | Sufficient evidence, criterion not met. |

Two design decisions carry most of the weight:

1. **The minimum-evidence check runs *before* any statistical test.** A significant result computed on three observations is exactly the false discovery this repo exists not to publish, so "the test passed" must never be reachable ahead of "there was enough data to test".
2. **With more than one declared arm, the read applies Holm–Bonferroni.** Declaring five arms and then reading the best-looking one is a search, and this makes the declared arm count actually cost something — a value that clears its threshold but fails correction is `NOT_CONFIRMED`.

A registration cannot manufacture evidence: a well-formed plan read against a losing tape still returns `NOT_CONFIRMED`. Its entire value is making the reader's decision rule inspectable *in advance*, so that a later "well, we said we'd look at Sharpe" is visible drift rather than a quiet one.

#### The registration is enforced, not merely available

A module nothing reads is decoration, so the plan is wired into the same single evidence document every other surface renders from:

- **Evidence document** — a `registration` section is *always* present, including when absent, so "no plan exists" is distinguishable from "this build predates the check". It is part of the `evidence_fingerprint`, so swapping the plan changes the fingerprint (the same isolation guarantee comparators get).
- **A sixth verdict dimension** — `pre_registration` appears in `python -m bot verdict`. It grades Strong (plan read), Moderate (valid plan, not yet read), or Weak (absent, or drifted).
- **The cap, not a refutation.** A claim with no valid plan cannot reach **"validated (provisional)"** — it is held at "promising, not validated", however strong the tape behind it is. Missing registration never produces `INVALIDATED`: a strategy with 200 clean forward days and no plan is still genuinely interesting, and saying otherwise would be its own dishonesty. The floor is `Moderate`, which is what makes the cap bite — `grade_registration` returns exactly `Weak` for an absent plan, so testing against `Weak` would have let the strongest verdict through precisely when there is no plan.
- **`verify-evidence` gains a registration-integrity gate.** An *absent* plan is not a failure — nobody is obliged to have one — but a **drifted** plan (edited after sealing) fails the gate, exactly like an edited freeze or a broken ledger chain. A committed artefact that lies is a failure; a missing one is a state.
- **Honest absence surfaces everywhere.** With no plan, `python -m bot evidence` prints `pre-registration: NONE`, and it becomes the headline caveat. The word is never softened into silence.

**The current registered question.** `registrations/banded-rm-forward-validity-v2.json` fixes, in advance: that under the corrected `exact-wealth-v2` accounting, the frozen `banded_rm` rule earns an out-of-sample Sharpe **≥ 0.70** on **≥ 90 clean forward days**, against two declared comparators (`inv_vol`, `buy_hold_equalweight`), read **once**, with no re-freeze to reset the clock. Its threshold is deliberately set below the 0.74 the historical walk-forward already produced — a threshold set *at* the observed value would be guaranteed to fail on noise, which is the error pre-registration is meant to prevent. It is registered **not yet read**, so it contributes nothing to the current verdict yet.

### Research-methodology battery (`python -m bot research`)

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

**Contribution decomposition** (`bot/attribution.py`) runs a controlled ablation ladder — base selected sleeves, inverse-vol weighting, momentum tilt, crisis de-risking, rebalance band, vol overlay, estimated execution costs — on the **exact same universe and period** for every rung (an `inputs_fingerprint` proves it). The cost line is reported, never re-simulated. It makes it obvious which component actually contributes value instead of letting the headline total hide a stack that subtracts.

**Consistency diagnostics** (`bot/walkforward_diagnostics.py`) answer whether the strategy is consistent rather than merely profitable overall: fold-by-fold return/Sharpe/drawdown, median and worst fold, percentage of profitable folds, cross-asset dispersion, and contribution concentration — including whether most performance comes from one year, one asset or one regime. Weak folds are always published.

**Data-quality and execution-quality scorecards** (`bot/data_quality.py`, `bot/execution_quality.py`) keep three things apart that used to blur into one vague "slippage" number: modelled cost, the observed simulated execution gap, and data-quality error (outages, skipped fills, stale feeds, lifecycle failures). Every problem is counted and visible rather than silently repaired.

### Execution-realism extensions

- **Cost models** (`bot/costs.py`): flat bps (default), volatility-dependent spread/slippage ((realized vol / reference)^0.5, clamped), square-root market impact `k * daily_vol * sqrt(participation)`, tiered maker/taker fees, and a dollar-volume liquidity filter — engine-integrated behind `CostParams`, off by default so existing results are unchanged
- **Calendar & data handling** (`bot/data.py`): exchange-calendar-aware portfolio alignment (an ETF held over a weekend is invested-flat, not cash; late-listed and delisted spans stay in cash), gap reports, tolerance-bounded forward-fill of small outages (filled bars flagged), delisting simulation with terminal liquidation cost

### Paper-trading reliability (`python -m bot trade`)

Rewritten around a persistent multi-asset paper portfolio:

- **Durable atomic state** — state is checksummed, the temp file is flushed + fsynced before rename, and non-finite numbers are refused before JSON persistence
- **Fail-closed append-only order ledger** — every fill records idempotency key, deltas, post-trade balances, fees, and the decision explanation; only an unterminated final crash fragment may be ignored, while interior corruption aborts recovery
- **Ledger-first crash recovery** — every durable fill is replay-validated for cash, position, notional, side, and idempotency continuity; if a crash leaves a valid-but-stale state snapshot behind, the fsynced ledger wins and repairs state
- **Duplicate-order prevention** — decisions carry `(date|symbol|action|target)` keys derived from the cycle timestamp; re-running a cycle cannot double-fill
- **Multi-asset-safe execution** — total portfolio equity is marked with the full price snapshot, missing held-position marks fail closed, every target quantity is sized from one pre-trade equity snapshot, SELLs fund BUYs first, fee pressure scales the whole BUY basket proportionally (not whichever symbol runs last), zero-quantity cash-clamped orders never consume an idempotency key, and over-allocated targets are normalized to the unlevered portfolio capacity (including capacity locked by stale holdings)
- **Failure isolation** — data-source and advisory-AI failures are audited without turning into orders or erasing an already-completed paper cycle
- **Data-staleness alerts** — symbols with frozen feeds get their trading blocked for the cycle and land in the audit trail (also raised by `forward --step`)
- **Decision explanations & atomic daily audit reports** — markdown under `reports/` with positions, fills, alerts, and why every decision was taken (holds included)

### Code identity: a freeze pins the implementation, not just numbers

The original freeze sealed configuration only — `freeze.json` recorded the commit, but the scheduled runner checked out whatever `main` held *today*, so editing `strategy.py` after freezing would silently change the experiment while the manifest kept claiming otherwise. That hole is closed at three levels:

1. **Algorithm seal.** `freeze.json["config"]["algorithm"]` now captures the COMPLETE portfolio construction — selection mode, candidate-pool version, universe rule, inverse-vol weighting (window + cap), XS-momentum tilt (lookback + max), crisis de-risking (window + threshold + multiplier), rebalance band, drawdown throttle state machine, vol-target overlay (target/window/fee) — every quantity that can affect a return, validated against unknown keys so a typo cannot silently become "default".
2. **Source seal.** `create_freeze` hashes the implementation itself (`bot/*.py` + `pyproject.toml`, LF-normalised so Windows trees and Linux checkouts of one commit hash identically; algorithm id `sha256-lf-v1` is recorded alongside the digest).
3. **Runner refusal.** `load_freeze` verifies all seals by default and `run_step` refuses to trade on mismatched code:
   ```
   CODE MISMATCH: the running implementation does not match the freeze.
     expected sha : 6c41…
     running sha  : 9d02…
   ```
4. **Frozen checkout in CI.** `.github/workflows/scheduled-paper.yml` reads the pointer from `freeze.json`, checks out `git_commit_at_freeze` (detached), runs `python -m bot verify-freeze` as a hard gate, trades one forward day on that code, then returns to main to append ONLY the log — data flows back; code never changes mid-experiment.
5. **Session-date scheduling.** New frozen runners execute shortly after midnight UTC and explicitly account the prior fully-closed calendar session (`forward --as-of-date YYYY-MM-DD`). They also record weekend/holiday crypto returns while US-ETF sleeves remain flat and are explicitly marked `session_closed`, preserving a true daily return tape. The workflow keeps the legacy 21:45 UTC weekday schedule for older frozen commits that do not support that argument, so a workflow upgrade cannot silently change an experiment already in flight.

**Experiment manifests are explicit.** `bot/experiment_manifest.py` gives every experiment one frozen identity — experiment id, version, primary rule, source commit, source/config fingerprints, candidate-pool identity, trial count, accounting model, execution model, universe method, benchmark, methodology version — and every forward observation and canonical record names its fingerprint. Combining evidence produced under different manifests raises `MixedExperimentError` and names the fields that disagree; nothing infers experiment identity from unrelated fields.

**Data sources use mirrors, because a refused host is not a market event.** Binance answers `HTTP 451` to whole datacentre ranges, GitHub Actions runner IPs included, and does so per endpoint: `/api/v3/ticker/24hr` and `/api/v3/klines` can refuse independently. Every Binance call therefore walks a mirror list (`api`, `api1`–`api3`, `data-api.binance.vision`), one attempt per host, re-raising the *last* error. `451` is in `NON_RETRYABLE_HTTP` — it is deterministic for a client IP, so retrying the host that just refused only burns the workflow's timeout.

The order is measured on a real runner rather than assumed. Probed there, **every** `*.binance.com` host answered `451`:

| host | `/api/v3/klines` | `/api/v3/ticker/24hr` |
|---|---|---|
| `api.binance.com` | 451 | 451 |
| `api1` / `api2` / `api3.binance.com` | 451 | 451 |
| `data-api.binance.vision` | **200** | **200** |

So `data-api.binance.vision` leads and the `.com` hosts stay as fallbacks for networks where they do answer — listing it last meant every fetch paid four dead requests before the one that worked. Closed candles are byte-identical across all five hosts (checked over 199 daily bars), so this is a transport decision and cannot change the experiment.

Two failure modes this defends against, both of which happened:

- a refused ticker host failed the snapshot step, which **skipped** the append step and discarded the forward day step 4 had already computed — five consecutive runs, one recorded day;
- a refused klines host left ten of thirteen sleeves in outage, so the day was recorded and counted while most of the portfolio went unobserved.

The forward step runs the **frozen** code, so a transport fix only takes effect at the next re-freeze. That is the seal working as designed: it refuses to let the experiment's data acquisition change underneath a forward test already in progress.

**Backtest/forward parity is tested, not assumed.** The flagship identity test (`tests/test_backtest_forward_parity.py`) runs ONE fixed dataset two ways — `run_strategy(...)` over full history vs freeze→`run_step` day by day — and requires per-asset, per-day equality of target weights, held weights, trades (count/size), costs, daily returns (incl. the overnight/intraday split and cash accrual) plus portfolio equity, under both a zero band and the 5% band (the banded case genuinely suppresses trades: 22 vs 40). Tolerance is 1e-12; both paths share `calculate_transition`, so they cannot drift. If this test fails, fix the implementation — never loosen it.

Artifact trail per freeze: config sha256 + algorithm sha256 + code sha256 + annotated git tag (`freeze/<YYYYMMDD>`) + optional container image digest (`--image-digest`; build with `docker build -t trading-bot .` and record `docker images --digests`). Verify any time with `python -m bot verify-freeze`. Freeze knobs mirror the backtest CLI: `--band/--no-tilt/--tilt-lookback/--max-tilt/--no-crisis/--corr-window/--corr-threshold/--derisk/--throttle/--dd-trigger/--dd-exit/--throttle-factor/--vol-window/--max-multiple-of-equal/--fixed/--no-overlay`.

## 6. Reproduction

Every number on this page can be regenerated, and one command checks that it still can:

```bash
python -m bot evidence                          # the evidence document every surface renders
python -m bot experiment-status                 # state machine + forward checkpoints
python -m bot reproduce canonical-v2            # re-execute the canonical run and diff metrics
python -m bot verify-evidence                   # pass/fail over the whole evidence chain
python -m bot compare-experiments v1 v2         # experiment comparison; flags CONFUNDED COMPARISON
```

`verify-evidence` checks git provenance, source fingerprint, canonical run hash, freeze integrity, evidence-chain integrity (which forward rows belong to this experiment), universe-chain integrity, experiment IDs, methodological currency, and README/evidence consistency. It returns a concise pass/fail report with exact failure reasons, and an unestablishable check is a failure — never a pass.

`compare-experiments` enforces common inputs where possible and shows the methodology, accounting-model, universe, strategy, code and candidate-pool differences alongside the performance delta. When more than one important variable changed at once it prints **CONFUNDED COMPARISON** and refuses to attribute the metric difference to any single change.

### Reproducible runs: `run.json`

Every `compare` run writes `runs/<id>/run.json` sealing the full context:

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

`reproduce` REFUSES to run unless: (1) the running code fingerprint matches the record, (2) each module seal matches, (3) every frozen dataset is still in `.cache` with an identical sha256 — no silent refreshes. It then re-executes deterministically and compares all stored metrics at 1e-12. `PASS` means the number you quoted is the number you can regenerate. Two reports generated from identical evidence are byte-identical (enforced by an invariant test).

### Usage

```bash
# Evidence surfaces (all render from one document)
python -m bot evidence
python -m bot evidence --json
python -m bot verdict                    # graded dimensions + exit code follows the primary rule

# Multi-asset-class walk-forward comparison vs the S&P 500 (realistic defaults)
python -m bot compare

# Variations
python -m bot compare --assets 40                # wider crypto universe
python -m bot compare --execution close          # optimistic fill model
python -m bot compare --fee 0.002 --slippage-bps 20 --latency-days 1
python -m bot sensitivity                        # all robustness sweeps on BTC
python -m bot validate                           # full statistical battery on BTC
python -m bot research                           # SPA, clustering/effective trials, ablation, Bayesian Sharpe

# Prospective paper trading (paper only)
python -m bot freeze                             # seal the experiment
python -m bot verify-freeze                      # refuse unless running code matches
python -m bot forward --step                     # one forward paper day

# Original toy path & live paper trading (paper only!)
python -m bot backtest --symbol BTCUSDT --interval 1h
python -m bot trade --strategy trendvol                 # persistent multi-asset paper loop
python -m bot trade --symbol BTCUSDT,ETHUSDT --once     # one cycle + daily audit report
```

## 7. Research details

### What the sensitivity analysis says (BTC, honest read)

- **Rolling blocks**: positive OOS Sharpe in every 2-year block (1.16 / 1.38 / 0.76) — not one lucky window.
- **Costs**: Sharpe degrades gracefully from 1.23 at 5bp to 0.84 at 100bp total friction.
- **Latency**: robust to 1-2 day delays (results within noise of zero-latency).
- **Parameters**: edge concentrates at short lookbacks (25-75d) and decays to ~0 at 200d — a real fragility, disclosed rather than hidden.
- **Execution**: close vs next-open are identical on crypto (Binance daily opens equal prior closes — 24h market); gaps matter only for the ETF sleeves.

### Research ledger: counting the TRUE number of experiments

An 85-candidate pool understates the real search. The project also explored equal vs inverse-vol weighting, XS tilt on/off, crisis thresholds, a drawdown throttle, band widths, execution models, cost models, universe rules... Every one of those is a draw from the same multiple-testing lottery.

`research_ledger.jsonl` is the append-only record of every experiment — hypothesis, full configuration, primary metric, numeric result, accepted/rejected, git provenance — hash-chained so edits and deletions of failed ideas are detectable (`python -m bot ledger` verifies and reports). Reads fail closed on interior JSON corruption; trial-count/DSR/freeze consumers verify the chain before trusting it; appends refuse non-finite results or a tampered history and can repair only the one safe crash shape (an unterminated final record). Backfilled entries (33 seeded from git history) carry `backfilled: true` and their original commit.

```bash
python -m bot ledger        # counts by category; search total = the honest N
```

The ledger total feeds `bot/search_accounting.py`, which is the only authority for the trial count a DSR correction may use. Rejected ideas stay in the file forever — the search was real even when the idea failed. Freezes pin `research_context` (ledger entry count + sha256), so forward-test corrections are computed against an immutable record of how much was searched.

### Recomputation history (the accounting fix)

> **Recomputed under the corrected accounting, on the SAME 20 symbols as the superseded run.** Session returns now evolve portfolio wealth exactly rather than adding simple legs. The comparison is held honest by pinning the universe (`--universe-symbols`): a naive re-run re-ranks today's 24h volume and silently trades a *different* portfolio, which would confound the accounting change with a universe change — an earlier draft of this note made exactly that error and overstated the effect by ~5x.
>
> Isolating the accounting change alone (identical universe, identical window): inv-vol OOS CAGR 10.75% -> 11.90%, Sharpe 0.696 -> 0.766; equal-weight CAGR 25.75% -> 25.38%, Sharpe 1.040 -> 1.047. The S&P 500 row is bit-identical on every metric, confirming the change is isolated to the bot's own P&L. The fix is real but modest — not the 6.6pp headline an unpinned re-run implied.
> `runs/canonical-v1/run.json` is kept as sealed history of the old model. Reproduce with the universe pinned:
> `python -m bot compare --run-id canonical-v2 --universe-symbols BTCUSDT,ETHUSDT,...`

### AI commentary (optional, OpenRouter / NVIDIA)

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

## 8. Architecture

### Web dashboard (Vercel)

The repo uses Vercel's zero-config layout

```bash
npm i -g vercel   # once
vercel            # deploy from the repo root; accept defaults
```

- **`/`** — the dashboard. Its one job is to show whether the frozen paper rule is still holding up, so the hero number is **forward paper days since the freeze plus the current verdict** — never the backtest CAGR. The page carries a permanent red education banner, accepts no deposits and no keys, and has no button that starts trading.
- **`/api/summary`** — pure-stdlib ASGI function whose `evidence` block is the same evidence document the README renders, in descending evidentiary weight: `verdict`, `forward` (evidence produced after the freeze), and `research` (pre-freeze development evidence). No third-party Python dependencies.

Dashboard hierarchy, in order: experiment state, current experiment identity, clean forward-paper days, data reliability, forward return evidence, benchmark comparison, historical walk-forward evidence, research diagnostics. The page gets quieter as the evidence gets worse: seal broken / no freeze → **silent** (integrity record only); fewer than 30 fully-observed forward days or outage-heavy → **quiet** (returns and the equity curve withheld, research folded shut); 30+ clean days → **full**. Bad forward results reduce the number of figures on the page; they never add a strategy or a "try this instead".

### Dashboard rules (enforced by `tests/test_dashboard_copy.py`)

The dashboard is a measurement instrument, not a funnel. Four rules are pinned by tests, because they are exactly the ones that erode:

| Rule | Why it is tested |
|---|---|
| Three evidence labels only — `research` / `out-of-sample` / `forward`. The word "live" appears nowhere in `public/index.html`. | A fourth label would imply the system trades real money. |
| No inputs, buttons or forms; no browser-side exchange calls. | Nothing on the page may let a visitor deposit, paste keys, or start trading. |
| Hero = forward paper days + verdict; the forward day count is the largest type on the page and CAGR never reaches the hero. | The headline must be the evidence quantity, not the most flattering backtest figure. |
| The hero counts days on which **every** sleeve printed, not scheduled days, and only counts them at all if the freeze's accounting model is still current. A superseded experiment's days are shown as **0 current trading days** with the reason stated, so the hero can never advertise a v1 tape as validation of v2 code. | A day with sleeves dark is calendar attendance, not evidence. Counting it would let a broken feed manufacture the number the page exists to report. |
| Exactly one recommended reading: the deflated-Sharpe caveat, not the headline line. | The search-corrected number is the one that decides whether anything here is real. |

### Module layout

```
bot/
  experiment_manifest.py # THE experiment identity: one frozen dataclass every
                         # evidence artifact names; mixed-manifest combinations
                         # are refused with the disagreeing fields listed
  experiment_state.py    # the deterministic state machine (RESEARCH_ONLY …
                         # INVALIDATED) derived from evidence, plus checkpoints
  evidence_model.py      # THE canonical verdict object + evidence.json assembly;
                         # the performance sentence is rendered here and nowhere else
  search_accounting.py   # one record of what was searched + the n-trials reasoning
                         # a DSR correction may use
  registration.py        # PRE-analysis plans: hypothesis, metric, threshold, arms
                         # and stopping rule fixed before results exist; content-
                         # addressed, fail-closed read-out, Holm-Bonferroni across
                         # declared arms
  reporting.py           # evidence assembly + every report renderer (CLI, README,
                         # API consume the same document)
  walkforward_diagnostics.py # fold/asset consistency + contribution concentration
  attribution.py         # controlled ablation ladder on one pinned universe/period
  data_quality.py        # data-quality scorecard (problems counted, never repaired)
  execution_quality.py   # turnover / predicted cost / observed gap kept apart
  canonical_identity.py  # the ONE primary-rule identity and its frozen-config check
  canonical_compare.py   # walk-forward comparison vs the S&P 500
  strategy.py            # 85-candidate strategy pool (+ 34 more via extended=True)
                         # and the prefix-sum / monotonic-deque indicator cache
  selection.py           # robustness-aware selection: one-SE rule, minimax
                         # sub-window tie-break, credibility shrinkage to a prior
  engine.py              # daily-bar engine: next-open execution, spread/slippage/
                         # latency, fee-on-turnover, cash accrual, rebalance banding,
                         # optional vol-dependent costs & square-root impact
  execution.py           # THE single transition calculation, shared by backtest and
                         # forward: exact self-financing wealth accounting
  lifecycle.py           # order-lifecycle timestamp provenance
  experiments.py         # accounting-model versioning: which freeze produced which
                         # tape and whether that model is still current
  metrics.py             # CAGR / excess Sharpe / Sortino / Calmar / VaR / ES
  walkforward.py         # expanding-window walk-forward + survivorship-safe combiners
  portfolio_rules.py     # XS-momentum tilt, crisis de-risk, drawdown throttle
  regimes.py             # bull/bear/sideways segmentation + regime-conditioned stats
  sensitivity.py         # parameter / rolling / cost / latency / execution sweeps
  stats_validation.py    # PSR, DSR, block bootstrap, Reality Check, shuffle MC
  research.py            # SPA test, expanded bootstraps, drawdown CIs, Bayesian Sharpe
  clustering.py          # strategy-family clusters, near-duplicates, effective trials
  ablation.py            # strategy-family and portfolio-overlay ablations
  costs.py               # vol-dependent spread/slippage, market impact, fee tiers
  cost_calibration.py    # predicted-vs-observed cost measurement from the paper tape
  data.py                # Binance + Yahoo fetchers, cleaning, gap handling, calendar
  prospective.py         # freeze manifest, forward paper-trading log, checkpoints
  evidence.py            # forward-evidence purity: classification + quarantine
  forward_evidence.py    # tested evidence transfer from the frozen worktree to main
  identity.py            # source fingerprint (sha256-lf-v1): freezes pin the code
  snapshot.py            # reproducible benchmark snapshots (data hashes + metrics)
  universe.py            # top-volume crypto + SPY/GLD/TLT cross-class ETFs
  universe_pit.py        # point-in-time eligibility + observed membership snapshots
  benchmark.py           # S&P 500 data (FRED primary, Yahoo fallback)
  research_ledger.py     # hash-chained record of every experiment ever searched
  runs.py                # run records + reproduction
  verdict.py             # the five evidence dimensions and their combination
  cache.py               # disk cache for daily history
  paper.py               # persistent multi-asset paper broker
api/             # Vercel serverless endpoint (stdlib ASGI), serves the evidence document
public/          # static dashboard served by Vercel
tests/           # unit/property/integration tests, including invariant tests that pin
                 # the evidence guarantees (comparators can never move the claim, mixed
                 # manifests refuse to combine, byte-identical regeneration, …)
.github/         # CI quality gates + scheduled paper-run workflow
Dockerfile       # containerized environment; default CMD runs the full gate
```

### Engineering quality

| Gate | Tool | Notes |
|---|---|---|
| Lint | ruff | config in `pyproject.toml` |
| Types | mypy | clean across all source modules |
| Tests + coverage | pytest + pytest-cov | 88% floor on library code |
| Property-style tests | seeded randomized invariants | `tests/test_properties.py` |
| Evidence invariants | dedicated test module | comparator isolation, manifest scoping, fail-closed corruption, byte-identical regeneration |
| Reproducible snapshots | `bot/snapshot.py` | pin data hashes + seed + metrics; verify drift |
| Environment | `Dockerfile` | quality gate by default; override for paper runs |
| Scheduled paper runs | `.github/workflows/scheduled-paper.yml` | frozen-code forward step on fully closed sessions + committed log |

### Running tests

```bash
pip install pytest ruff mypy pytest-cov
ruff check .        # lint
mypy                # type check
pytest --cov=bot    # tests + coverage gate (88% floor on library code)
```
