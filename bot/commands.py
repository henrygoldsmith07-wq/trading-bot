"""CLI command implementations.

Each run_* function is one CLI subcommand. Argument parsing and dispatch
are in bot/cli.py; the package entry point is bot/__main__.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from .canonical_compare import compute_compare_results
from .compare_common import (
    as_date,
    fmt_pct,
)


def current_git_commit_safe():
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_compare(args) -> int:
    res = compute_compare_results(args)
    if isinstance(res, dict) and "exit_code" in res:
        return int(res["exit_code"])
    return 1


def run_sensitivity(args) -> int:
    from .cache import load_or_fetch
    from .data import fetch_daily_history
    from .sensitivity import full_sensitivity

    candles = load_or_fetch(args.symbol, lambda s: fetch_daily_history(s))[0]
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
        risk_free_annual=args.risk_free,
    )
    print(f"Sensitivity analysis on {args.symbol} walk-forward (execution={args.execution}, "
          f"fee+spread+slippage = {(args.fee * 1e4 + args.spread_bps + args.slippage_bps):.0f}bp)...")
    res = full_sensitivity(candles, train_days=args.train_days, test_days=args.test_days, **engine_kwargs)

    print("\n1) Parameter sensitivity (TrendVol grid, walk-forward OOS Sharpe / CAGR):")
    print("      " + "".join(f"{f'target {tv:.0%}':>16}" for tv in (0.20, 0.30, 0.40)))
    lookbacks = sorted({lb for lb, _ in res["parameters"]})
    for lb in lookbacks:
        row = f"{f'lb {lb}':>5}"
        for tv in (0.20, 0.30, 0.40):
            m = res["parameters"][(lb, tv)]
            row += f"{m['sharpe']:>8.2f}/{fmt_pct(m['cagr']):>7}"
        print(row)
    sharpes = [m["sharpe"] for m in res["parameters"].values()]
    print(f"  grid Sharpe min/median/max: {min(sharpes):.2f} / {sorted(sharpes)[len(sharpes) // 2]:.2f} / {max(sharpes):.2f}")

    print("\n2) Rolling parameter sensitivity (consecutive 2y blocks, walk-forward winner):")
    for b in res["rolling"]:
        m = b["metrics"]
        print(f"  {as_date(b['start'])}..{as_date(b['end'])}: CAGR {fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}  maxDD {fmt_pct(m['max_drawdown']):>7}")

    print("\n3) Transaction-cost sensitivity (combined fee+spread+slippage):")
    for bps, m in sorted(res["costs"].items()):
        print(f"  {bps:>4}bp: CAGR {fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}  maxDD {fmt_pct(m['max_drawdown']):>7}")

    print("\n4) Latency & execution realism:")
    for lat, m in sorted(res["latency"].items()):
        print(f"  latency {lat}d (next_open): CAGR {fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}")
    for mode, m in res["execution"].items():
        print(f"  execution={mode:10}: CAGR {fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}")
    return 0


def run_validate(args) -> int:
    from .cache import load_or_fetch
    from .data import fetch_daily_history
    from .metrics import calmar, expected_shortfall, kurtosis, skewness, sortino, var_hist
    from .sensitivity import parameter_grid
    from .stats_validation import (
        bootstrap_metrics,
        dsr,
        parameter_stability,
        psr,
        reality_check,
        shuffle_test,
        start_end_sensitivity,
    )
    from .strategy import build_candidates
    from .walkforward import absolute_folds, fixed_candidate_streams, nested_selection_fn, walk_forward_at

    candles = load_or_fetch(args.symbol, lambda s: fetch_daily_history(s))[0]
    folds = absolute_folds(candles, args.train_days, args.test_days)
    if not folds:
        print("Not enough history for one fold")
        return 2
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
        risk_free_annual=args.risk_free,
    )
    candidates = build_candidates()
    print(f"Statistical validation on {args.symbol}: {len(folds)} folds, {len(candidates)} candidates, "
          f"execution={args.execution}, {(args.fee * 1e4 + args.spread_bps + args.slippage_bps):.0f}bp friction")

    print("\n[1/7] Standard walk-forward (selection by train Sharpe, 30d embargo)...")
    wf = walk_forward_at(candles, folds, candidates=candidates, embargo_days=args.embargo_days, **engine_kwargs)
    days = sorted(wf["daily"])
    oos = [wf["daily"][t] for t in days]
    print(f"  OOS: CAGR {fmt_pct(wf['cagr'])}, Sharpe {wf['sharpe']:.2f}, maxDD {fmt_pct(wf['max_drawdown'])}, "
          f"exposure {wf['exposure']:.2f}, turnover {wf['turnover']:.1f}x/yr-ish")

    print("\n[2/7] Nested walk-forward (inner purged CV picks the strategy)...")
    nested = walk_forward_at(
        candles, folds, candidates=candidates, embargo_days=args.embargo_days,
        selection_fn=nested_selection_fn(purge_days=args.purge_days, embargo_days=args.embargo_days),
        **engine_kwargs,
    )
    print(f"  OOS: CAGR {fmt_pct(nested['cagr'])}, Sharpe {nested['sharpe']:.2f}, maxDD {fmt_pct(nested['max_drawdown'])}")
    print(f"  picks: {[p['strategy'] for p in nested['folds']]}")
    print("  (nested vs standard difference is the selection-overfitting estimate)")

    print("\n[3/7] Sharpe ratios: probabilistic & deflated...")
    import statistics as _stats

    p = psr(oos)  # PSR vs zero benchmark
    trials = len(candidates)
    d = dsr(oos, wf["trial_sharpes"], trials)
    print(f"  PSR (vs Sharpe 0): {p:.3f}")
    print(f"  DSR (deflated for {trials} pool trials, trial-Sharpe sd {_stats.stdev(wf['trial_sharpes']):.2f}): {d:.3f}")
    from .research_ledger import recommended_trial_count

    n_ledger = recommended_trial_count("research_ledger.jsonl")
    if n_ledger:
        # The research program searched more than one selection grid: portfolio
        # rules, bands, throttles, execution models... Count them all.
        combined_trials = list(wf["trial_sharpes"])
        from .research_ledger import load_entries, summarize

        led_summary = summarize(load_entries("research_ledger.jsonl"))
        combined_trials += led_summary["trial_sharpes"]
        d_ledger = dsr(oos, combined_trials, trials + n_ledger)
        print(f"  DSR (ledger-informed: {trials} pool + {n_ledger} research-program experiments): {d_ledger:.3f}"
              f"   <- the honest number")
    else:
        print("  (no research ledger found — pool-only correction; seed research_ledger.jsonl for the honest N)")
    print("  PSR/DSR > 0.95 is the usual 'real edge' bar; DSR is the honest number")

    print("\n[4/7] Stationary block bootstrap (20d blocks)...")
    boot = bootstrap_metrics(oos, n_boot=args.boots, seed=args.seed)
    print(f"  CAGR 90% CI: [{fmt_pct(boot['cagr_ci'][0])}, {fmt_pct(boot['cagr_ci'][1])}]")
    print(f"  Sharpe 90% CI: [{boot['sharpe_ci'][0]:.2f}, {boot['sharpe_ci'][1]:.2f}]")
    print(f"  Max-drawdown distribution: median {fmt_pct(boot['mdd_median'])}, p95 {fmt_pct(boot['mdd_p95'])}, worst {fmt_pct(boot['mdd_worst'])}")

    print(f"\n[5/7] White's Reality Check across all {trials} candidates ({args.rc_boots} bootstrap max-tests)...")
    streams = fixed_candidate_streams(candles, folds, candidates, **engine_kwargs)
    common_days = sorted(set.intersection(*(set(s) for s in streams.values()))) if streams else []
    matrix = [[s[t] for t in common_days] for s in streams.values()]
    rc = reality_check(matrix, n_boot=args.rc_boots, seed=args.seed)
    print(f"  best fixed-candidate OOS Sharpe: {rc['best_sharpe']:.2f}")
    print(f"  RC p-value: {rc['p_value']:.3f}  (< 0.05 => best strategy unlikely to be pure selection luck)")

    print("\n[6/7] Path & window robustness...")
    sh = shuffle_test(oos, n_boot=args.boots, seed=args.seed)
    print(f"  trade-order shuffle: actual maxDD {fmt_pct(sh['actual_mdd'])} vs shuffled median {fmt_pct(sh['shuffled_mdd_median'])}; "
          f"protection percentile {sh['dd_percentile']:.2f} (high = return sequencing itself avoids drawdowns)")
    print("  start/end-date sensitivity (CAGR / Sharpe by trimming the window):")
    for row in start_end_sensitivity(oos):
        print(f"    trim start {row['trim_start']:>3}d end {row['trim_end']:>3}d: {fmt_pct(row['cagr']):>7} / {row['sharpe']:.2f}")

    print("\n[7/7] Distribution, tail risk & parameter stability...")
    print(f"  daily returns: skew {skewness(oos):.2f}, kurtosis {kurtosis(oos):.1f} (normal=3)")
    print(f"  VaR 95% (descriptive only): {fmt_pct(var_hist(oos, 0.95))}, VaR 99%: {fmt_pct(var_hist(oos, 0.99))}")
    print(f"  Expected shortfall 95%: {fmt_pct(expected_shortfall(oos, 0.95))}, 97.5%: {fmt_pct(expected_shortfall(oos, 0.975))}")
    print(f"  worst day: {fmt_pct(min(oos))}")
    print(f"  Sortino {sortino(oos, 365):.2f}, Calmar {calmar(wf['cagr'], wf['max_drawdown']):.2f}")
    grid = parameter_grid(candles, folds, **engine_kwargs)
    stab = parameter_stability(grid)
    print(f"  TrendVol grid Sharpe: min {stab['min']:.2f} / median {stab['median']:.2f} / max {stab['max']:.2f}; "
          f"{stab['share_above_half_max']:.0%} of cells >= half-max; mean neighbor delta {stab['mean_neighbor_delta']:.2f}")

    print("\n[8] A-priori fixed rule (no selection, N=1 trials)...")
    from .strategy import risk_ensemble

    fixed = walk_forward_at(candles, folds, candidates=[risk_ensemble()], **engine_kwargs)
    fdays = sorted(fixed["daily"])
    foos = [fixed["daily"][t] for t in fdays]
    fpsr = psr(foos)
    fdsr = dsr(foos, [fixed["sharpe"]], 1)
    print(f"  RiskEnsemble OOS: CAGR {fmt_pct(fixed['cagr'])}, Sharpe {fixed['sharpe']:.2f}, maxDD {fmt_pct(fixed['max_drawdown'])}")
    print(f"  PSR {fpsr:.3f}, DSR {fdsr:.3f} (trial count = 1: nothing was selected, so nothing to deflate)")
    print(f"  vs selected stream: PSR {p:.3f}, DSR {d:.3f} at {trials} trials")
    print("  if the fixed rule's DSR beats the selected stream's, the honest edge is the rule — not the search")
    return 0


def run_research(args) -> int:
    from .ablation import family_ablation
    from .cache import load_or_fetch
    from .clustering import effective_trial_count, near_duplicate_pairs, strategy_clusters
    from .data import fetch_daily_history
    from .research import (
        bayesian_sharpe,
        drawdown_confidence_intervals,
        expanded_bootstrap,
        mc_future_paths,
        probability_of_underperformance,
        sequence_risk,
        spa_test,
    )
    from .strategy import BuyHold, build_candidates
    from .walkforward import absolute_folds, fixed_candidate_streams, walk_forward_at

    candles = load_or_fetch(args.symbol, lambda s: fetch_daily_history(s))[0]
    folds = absolute_folds(candles, args.train_days, args.test_days)
    if not folds:
        print("Not enough history for one fold")
        return 2
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
        risk_free_annual=args.risk_free,
    )
    candidates = build_candidates()
    print(f"Research-methodology battery on {args.symbol}: {len(folds)} folds, {len(candidates)} candidates")

    print("\n[1/7] Walk-forward OOS record (selection by train Sharpe)...")
    wf = walk_forward_at(candles, folds, candidates=candidates, embargo_days=args.embargo_days, **engine_kwargs)
    days = sorted(wf["daily"])
    oos = [wf["daily"][t] for t in days]
    print(f"  OOS: CAGR {fmt_pct(wf['cagr'])}, Sharpe {wf['sharpe']:.2f}, maxDD {fmt_pct(wf['max_drawdown'])}")

    print(f"\n[2/7] Candidate stream structure ({args.rc_boots} SPA bootstraps)...")
    streams = fixed_candidate_streams(candles, folds, candidates, **engine_kwargs)
    common_days = sorted(set.intersection(*(set(s) for s in streams.values()))) if streams else []
    matrix = [[s[t] for t in common_days] for s in streams.values()]
    spa = spa_test(matrix, n_boot=args.rc_boots, seed=args.seed)
    print(f"  SPA p-value: {spa['p_value']:.3f}  (< 0.05 => best candidate unlikely to be selection luck)")

    print("\n[3/7] Strategy families: clusters, duplicates, effective trial count...")
    aligned = {name: [s[t] for t in common_days] for name, s in streams.items()}
    eff = effective_trial_count(aligned)
    print(f"  {eff['n_strategies']} strategies -> {eff['n_clusters']} clusters at corr 0.90; "
          f"effective trials ~{eff['n_effective']:.1f}; avg pairwise corr {eff['avg_pairwise_corr']:.3f}")
    from .research_ledger import recommended_trial_count as _ledger_n

    n_ledger = _ledger_n("research_ledger.jsonl")
    if n_ledger:
        print(f"  research ledger records {n_ledger} additional portfolio/execution/universe "
              f"searches — DSR corrections in 'validate' count these too")
    dups = near_duplicate_pairs(aligned)
    if dups:
        print(f"  near-duplicates (|rho| >= 0.995): {len(dups)} pair(s), e.g.:")
        for d in dups[:5]:
            print(f"    {d['a']}  <->  {d['b']}  (rho={d['correlation']:.4f})")
    else:
        print("  no near-duplicate pairs at |rho| >= 0.995")
    clusters = strategy_clusters(aligned)
    big = sorted(clusters.values(), key=len, reverse=True)[:3]
    print(f"  largest families: {[f'{len(m)} members' for m in big]}")

    print("\n[4/7] Strategy-family ablation (omit one family at a time)...")
    abl = family_ablation(candles, folds, candidates, **engine_kwargs)
    print(f"  {'family':12}{'remaining':>10}{'OOS Sharpe':>12}{'dSharpe':>9}")
    for row in abl[:6]:
        print(f"  {row['omitted_family']:12}{row['n_remaining']:>10}{row['sharpe']:>12.2f}{row.get('sharpe_delta', 0.0):>+9.2f}")
    print("  (negative delta = removing the family HURT; positive = it was noise fit)")

    print("\n[5/7] Bayesian Sharpe (posterior over annualized Sharpe)...")
    bays = bayesian_sharpe(oos, draws=4000, seed=args.seed)
    lo, hi = bays["ci_90"]
    print(f"  posterior mean {bays['posterior_mean']:.2f}, median {bays['median']:.2f}, 90% CI [{lo:.2f}, {hi:.2f}], "
          f"P(Sharpe > 0) = {bays['prob_above_benchmark']:.3f}")

    print("\n[6/7] Drawdown & sequence risk...")
    dd = drawdown_confidence_intervals(oos, n_boot=args.boots // 2, seed=args.seed)
    print(f"  maxDD 90% CI: [{fmt_pct(dd['mdd_90_ci'][0])}, {fmt_pct(dd['mdd_90_ci'][1])}], worst {fmt_pct(dd['mdd_worst'])}; "
          f"time-under-water median {dd['time_under_water_median']:.0%}, p95 {dd['time_under_water_p95']:.0%}")
    seq = sequence_risk(oos, horizon_days=min(252, len(oos) // 2), n_shuffles=100, seed=args.seed)
    print(f"  sequence risk over {seq['horizon_days']}d windows: P(loss) observed {seq['observed_p_loss']:.2f} vs shuffled "
          f"{seq['shuffle_mean_p_loss']:.2f} (gap {seq['sequence_risk_gap']:+.2f}); "
          f"worst entry {fmt_pct(seq['observed_worst'])}, best entry {fmt_pct(seq['observed_best'])}")
    mc = mc_future_paths(oos, horizon_days=min(252, len(oos)), n_paths=2000, seed=args.seed)
    print(f"  forward MC (block bootstrap, 1y): P(loss) {mc['p_loss']:.2f}, terminal p5/p50/p95 "
          f"{mc['terminal_p05']:.2f}/{mc['terminal_median']:.2f}/{mc['terminal_p95']:.2f}, path-maxDD p95 {fmt_pct(mc['path_mdd_p95'])}")
    eb = expanded_bootstrap(oos, n_boot=max(args.boots // 2, 200), seed=args.seed)
    schemes = list(eb)
    agree_lo = min(eb[s]["sharpe_ci"][0] for s in schemes)
    agree_hi = max(eb[s]["sharpe_ci"][1] for s in schemes)
    print(f"  bootstrap agreement (stationary/circular/moving): Sharpe CI union [{agree_lo:.2f}, {agree_hi:.2f}]")

    print("\n[7/7] Probability of underperformance vs buy&hold (same window)...")
    bh_stream = fixed_candidate_streams(candles, folds, [BuyHold()], **engine_kwargs).get("BuyHold", {})
    bench_days = sorted(bh_stream)
    shared = sorted(set(days) & set(bench_days))
    strat_al = [wf["daily"][t] for t in shared]
    bench_al = [bh_stream[t] for t in shared]
    pou = probability_of_underperformance(strat_al, bench_al, n_boot=args.boots // 2, seed=args.seed)
    print(f"  P(bot CAGR < b&h): {pou['p_underperform_cagr']:.3f};  P(bot Sharpe < b&h): {pou['p_underperform_sharpe']:.3f}")
    print(f"  Sharpe gap 90% CI: [{pou['sharpe_gap_ci'][0]:+.2f}, {pou['sharpe_gap_ci'][1]:+.2f}]")
    return 0


def run_calibrate_costs(args) -> int:
    from .cost_calibration import (
        calibrate,
        format_report,
        load_observations,
        write_calibration,
    )

    obs = load_observations(args.observations)
    if not obs:
        print(f"No cost observations at {args.observations} — they accumulate automatically "
              "from 'forward --step' turnover events")
        return 2
    v1 = {
        "fee": args.fee,
        "spread_bps": args.spread_bps,
        "slippage_bps": args.slippage_v1_bps if hasattr(args, "slippage_v1_bps") else args.spread_bps,
    }
    report = calibrate(obs, v1_frictions=v1, min_observations=args.min)
    print(format_report(report))
    if args.write:
        path = write_calibration(report)
        print(f"\nwritten: {path}")
    return 0


def run_universe_snapshot(args) -> int:
    """Record TODAY's ranked universe — today's data recorded today is
    point-in-time by construction. Daily snapshots compound into the
    survivorship-free dataset future backtests deserve."""
    import json as _json

    from .universe import fetch_ticker_json, parse_symbols
    from .universe_pit import record_snapshot

    # ONE fetch, across mirrors. This used to fetch a second time straight from
    # the primary host, which meant a 451 on that host failed the step even
    # when the ranking itself had succeeded on a fallback mirror.
    try:
        raw = fetch_ticker_json()
    except Exception as exc:
        print(f"could not fetch Binance 24h ticker rankings from any mirror: {exc}")
        return 2
    ranked_raw = parse_symbols(raw, n=args.top)
    if not ranked_raw:
        print("could not fetch Binance 24h ticker rankings")
        return 2
    rows = _json.loads(raw)
    if not isinstance(rows, list):
        print(f"unexpected ticker payload: {type(rows).__name__}")
        return 2
    vol_by_sym = {}
    for t in rows:
        sym = t.get("symbol", "")
        try:
            vol_by_sym[sym] = float(t.get("quoteVolume", 0) or 0)
        except ValueError:
            continue
    ranked = [(s, vol_by_sym.get(s, 0.0)) for s in ranked_raw]
    res = record_snapshot(ranked, log_path=args.log)
    print(f"universe snapshot [{res['status']}] {res['date']}: {res.get('n', len(res.get('symbols', [])))} pairs")
    return 0


def run_ask(args) -> int:
    """Grounded Q&A over the bot's own computed state (advisory AI layer)."""
    import json as _json

    from .ai import DEFAULT_MODEL, complete, load_api_key
    if not load_api_key():
        print("No OPENROUTER_API_KEY found (set it or put it in .env). AI layer disabled.")
        return 2
    snapshot = {}
    try:
        from api.summary import build_summary

        snapshot = build_summary(args.symbol)
    except Exception as e:  # noqa: BLE001 - degrade gracefully, answer ungrounded
        print(f"[ai] live summary unavailable ({e}); answering from the question alone")

    system = (
        "You are the commentary layer of a deterministic paper-trading RESEARCH bot. "
        "You get a JSON snapshot of real computed metrics. Interpret and explain ONLY "
        "what the snapshot supports; never invent numbers. Flag caveats honestly "
        "(selection effects, deflated Sharpe, regime dependence). Plain prose, under 250 words."
    )
    user = _json.dumps(snapshot, indent=1)[:6000] + f"\n\nQuestion: {args.question}"
    print(f"Asking {DEFAULT_MODEL.split(' (')[0]} (rotation across all approved free models)...")
    answer = complete(user, system=system)
    if answer is None:
        return 3
    print("\n[AI commentary - advisory only; never feeds back into weights or decisions]\n")
    print(answer)
    return 0


def run_freeze(args) -> int:
    import subprocess

    from .cache import load_or_fetch
    from .data import fetch_daily_history, fetch_yahoo_daily, is_stale
    from .prospective import create_freeze
    from .universe import ETF_UNIVERSE, top_symbols
    from .walkforward import absolute_folds, walk_forward_at

    btc = load_or_fetch("BTCUSDT", lambda s: fetch_daily_history(s))[0]
    folds = absolute_folds(btc, args.train_days, args.test_days)
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
        risk_free_annual=args.risk_free,
        embargo_days=args.embargo_days,
    )
    universe = ["BTCUSDT"] + [s for s in top_symbols(args.assets) if s != "BTCUSDT"]
    universe = universe[: args.assets]
    specs: list[tuple[str, str, int]] = [(s, "binance", 365) for s in universe] + [
        (e["symbol"], "yahoo", e["periods_per_year"]) for e in ETF_UNIVERSE
    ]
    sessions = {e["symbol"]: e.get("session", "continuous") for e in ETF_UNIVERSE}
    min_history = args.train_days + args.test_days + 180
    assets: list[dict] = []
    print("Selecting per-asset strategies on data up to the freeze (never forward):")
    for symbol, source, ppy in specs:
        fetcher = fetch_yahoo_daily if source == "yahoo" else fetch_daily_history
        try:
            candles = load_or_fetch(symbol, fetcher)[0]
        except Exception as e:
            print(f"  {symbol:10} fetch failed ({e}), skipped")
            continue
        if len(candles) < min_history or is_stale(candles):
            print(f"  {symbol:10} skipped (history/stale)")
            continue
        wf = walk_forward_at(candles, folds, periods_per_year=ppy, **engine_kwargs)
        pick = wf["folds"][-1]["strategy"]
        from .strategy import build_candidates

        chosen = next(c for c in build_candidates() if repr(c) == pick)
        print(f"  {symbol:10} -> {pick}")
        assets.append(
            {
                "symbol": symbol,
                "source": source,
                "periods_per_year": ppy,
                "session": sessions.get(symbol, "continuous"),
                "strategy": chosen,
                "oos_sharpe_at_freeze": wf["sharpe"],
            }
        )
    # A freeze pins a COMMIT. If the tree is dirty, the code that produced the
    # manifest is not the code at that commit, so the manifest would name a
    # source that never ran. Refuse rather than record an unreproducible seal.
    from .canonical_identity import assert_primary_rule_is_frozen
    from .provenance import require_clean_tree

    try:
        _state = require_clean_tree(
            allow_dirty=bool(getattr(args, "allow_dirty_tree", False)),
            what="freeze",
        )
        commit = _state.head_commit
    except Exception as exc:
        print(f"::error::{exc}")
        return 1

    # Immutable-artifact trail: tag the frozen commit (keeps it reachable and
    # human-auditable) and record a container digest when one was built.
    tag = f"freeze/{args.tag or manifest_date()}"
    if commit and not args.no_tag:
        made = subprocess.run(["git", "tag", "-f", tag, commit], capture_output=True, text=True)
        if made.returncode == 0:
            print(f"tagged frozen commit as {tag}")
        else:
            print(f"warning: could not create git tag {tag} ({made.stderr.strip()})")
    if args.image_digest:
        print(f"recording image digest {args.image_digest}")

    from .algorithm import algorithm_fingerprint, build_algorithm

    algorithm = build_algorithm(
        selection_mode="fixed_risk_ensemble" if args.fixed else "walk_forward_selected",
        rebalance_band=args.band,
        use_tilt=not args.no_tilt,
        tilt_lookback=args.tilt_lookback,
        max_tilt=args.max_tilt,
        use_crisis=not args.no_crisis,
        corr_window=args.corr_window,
        corr_threshold=args.corr_threshold,
        derisk_factor=args.derisk,
        use_throttle=args.throttle,
        dd_trigger=args.dd_trigger,
        dd_exit=args.dd_exit,
        throttle_factor=args.throttle_factor,
        vol_window=args.vol_window,
        max_multiple_of_equal=args.max_multiple_of_equal,
        overlay_enabled=not args.no_overlay,
        target_vol=args.portfolio_vol,
    )

    # The headline grades ONE rule. A freeze of anything else would run a
    # forward experiment whose results can never support the canonical claim.
    try:
        assert_primary_rule_is_frozen(algorithm)
    except Exception as exc:
        print(f"::error::{exc}")
        return 1

    manifest = create_freeze(
        assets,
        frictions={
            "fee": args.fee,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "execution": args.execution,
            "risk_free_annual": args.risk_free,
        },
        algorithm=algorithm,
        path=args.freeze_file,
        git_commit=commit,
        git_tag=tag if commit and not args.no_tag else None,
        image_digest=args.image_digest,
        research_context=_research_context(),
    )

    print(f"\nFroze {len(assets)} assets at {manifest['frozen_at']}")
    print(f"config sha256: {manifest['config_sha256']}")
    print(f"code sha256  : {manifest['code_sha256']} ({manifest['code_fingerprint_algo']})")
    print(f"algo sha256  : {algorithm_fingerprint(algorithm)}")
    print(f"  mode={algorithm['selection_mode']} band={algorithm['rebalance_band']:.0%} "
          f"tilt={algorithm['xs_momentum']['enabled']} crisis={algorithm['crisis_derisk']['enabled']} "
          f"throttle={algorithm['drawdown_throttle']['enabled']} invvol(w={algorithm['weighting']['vol_window']}) "
          f"overlay(target={algorithm['overlay']['target_vol']:.0%})")
    print(f"commit       : {manifest['git_commit_at_freeze']}  tag: {manifest.get('git_tag')}")
    if manifest.get("image_digest"):
        print(f"image digest : {manifest['image_digest']}")
    print(f"PUSH THE TAG: git push origin {tag}")
    print(f"manifest: {args.freeze_file} — COMMIT IT NOW; CI will check out this exact commit, "
          "verify the code fingerprint, then trade the forward day on frozen code only")
    return 0


def manifest_date() -> str:

    return datetime.now(UTC).strftime("%Y%m%d")


def _research_context() -> dict | None:
    """Pin the research ledger AND the canonical benchmark at freeze time."""
    import hashlib as _hl
    import pathlib as _pl

    from .research_ledger import ledger_fingerprint, recommended_trial_count
    from .runs import RUNS_DIR

    fp = ledger_fingerprint()
    ctx: dict = {}
    if fp is not None:
        n_entries, sha = fp
        ctx.update(
            {
                "ledger_entries": n_entries,
                "ledger_sha256": sha,
                "recommended_trial_count": recommended_trial_count(),
            }
        )
    canonical = _pl.Path(RUNS_DIR) / "canonical-v1" / "run.json"
    if canonical.exists():
        raw = canonical.read_bytes()
        try:
            rec = json.loads(raw)
            run_id = rec.get("run_id")
            record_sha = rec.get("record_sha256")
        except json.JSONDecodeError:
            run_id, record_sha = None, None
        ctx["canonical_run"] = {
            "run_id": run_id,
            "record_sha256": record_sha,
            "file_sha256": _hl.sha256(raw).hexdigest(),
        }
    return ctx or None


def run_reproduce(args) -> int:
    from .runs import ReproduceRefused, list_runs, reproduce_run

    if args.run_id.lower() in ("list", "ls"):
        runs = list_runs(args.runs_dir)
        if not runs:
            print(f"No saved runs under {args.runs_dir}/")
            return 0
        print(f"Saved runs ({len(runs)}):")
        for r in runs:
            print(f"  {r['run_id']}  {r['created_at']}")
        return 0
    # Provenance first. A record that claims a commit the current tree is not at
    # cannot be reproduced by definition, and saying so up front beats a wall of
    # confusing metric diffs later.
    try:
        from .provenance import ProvenanceMismatch, assert_record_matches_source
        from .runs import load_run_record

        _rec = load_run_record(args.run_id, runs_dir=args.runs_dir)
        assert_record_matches_source(_rec)
    except ProvenanceMismatch as e:
        print(f"PROVENANCE FAILURE: {e}")
        print("  remedy: check out the commit the record names, or regenerate the record")
        print("          from the source you intend to certify.")
        return 2
    except Exception:
        pass  # record unreadable; reproduce_run will report it properly

    try:
        res = reproduce_run(args.run_id, runs_dir=args.runs_dir)
    except (ReproduceRefused, FileNotFoundError, ValueError) as e:
        print(f"REFUSED: {e}")
        return 2
    status = res["status"]
    print(f"reproduce {res['run_id']}: {status}  "
          f"({res['n_compared_paths']} metric blocks compared)")
    if res["diffs"]:
        for d in res["diffs"][:20]:
            print(f"  DIFF {d}")
        return 1
    print("every stored metric reproduced exactly — the validated engine is the engine that ran")
    return 0


def run_verdict(args) -> int:
    """The product: how much evidence actually supports the frozen system?"""
    import json as _json
    import pathlib as _pl

    from .prospective import load_freeze
    from .research_ledger import load_entries, recommended_trial_count
    from .runs import RUNS_DIR, load_run_record
    from .verdict import build_verdict, format_verdict

    canonical_path = _pl.Path(RUNS_DIR) / "canonical-v1" / "run.json"
    if not canonical_path.exists():
        print(f"REFUSED: no canonical record at {canonical_path} — generate it first "
              "(python -m bot compare --assets 20 --run-id canonical-v1)")
        return 2
    record = load_run_record("canonical-v1", runs_dir=RUNS_DIR)
    m = record["results"]["metrics"]
    try:
        from .strategy import build_candidates

        pool = len(build_candidates())
    except Exception:
        pool = 85  # documented fallback: the canonical-era pool size

    entries = load_entries("research_ledger.jsonl")
    ledger_n = recommended_trial_count("research_ledger.jsonl") if entries else None

    from .cost_calibration import calibrate, load_observations

    obs = load_observations("cost_observations.jsonl")
    v1_frictions = record["results"]["parameters"].get(
        "frictions", {"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0}
    )
    freeze_manifest = None
    try:
        freeze_manifest = load_freeze("freeze.json", verify_code=False)
    except (OSError, ValueError):
        freeze_manifest = None
    cost_report = calibrate(obs, v1_frictions=v1_frictions,
                            freeze_manifest=freeze_manifest) if obs else None

    from api.summary import build_forward_summary  # noqa: E402

    forward = build_forward_summary()

    v = build_verdict(
        canonical_rule_stats=m.get("rules", []),
        canonical_per_asset=record["results"].get("per_asset", []),
        canonical_n_folds=record["results"].get("n_folds"),
        pool_size=pool or 85,
        ledger_search_n=ledger_n,
        cost_report=cost_report,
        forward=forward,
        headline_rule_substring=args.headline_rule,
    )
    if args.json:
        print(_json.dumps(v, indent=2))
    else:
        print(format_verdict(v))
    return 0


def run_quarantine_costs(args) -> int:
    """Classify the legacy mixed cost tape; preserve every byte."""
    import pathlib as _pl

    from .evidence import quarantine_archive
    from .prospective import load_freeze

    if not _pl.Path(args.src).exists():
        print(f"No tape at {args.src} — nothing to quarantine")
        return 2
    manifest = load_freeze(args.freeze_file, verify_code=False)
    # verify_code=False is correct here: quarantine is a BOOKKEEPING migration
    # that executes no strategy code. The forward study stays pinned to its
    # frozen commit — the CI runner re-verifies before every step.
    report = quarantine_archive(
        args.src,
        archive_path=args.archive,
        quarantine_path=args.quarantine,
        keep_path=args.keep,
        freeze_manifest=manifest,
    )
    print(f"quarantined {report['rows_total']} rows from {args.src}")
    print(f"  kept forward-paper : {report['kept_forward_paper']}")
    for reason, count in sorted(report["excluded"].items()):
        print(f"  excluded {reason:16}: {count}")
    print(f"archive    : {report['archive_path']}  (original bytes)")
    print(f"audit copy : {report['quarantine_path']}")
    print(f"pure tape  : {report['keep_path']}  <- production reads this")
    return 0


def run_ledger(args) -> int:
    from .research_ledger import load_entries, summarize, verify_chain

    entries = load_entries(args.ledger)
    if not entries:
        print(f"No research ledger at {args.ledger} — seed one with scripts/seed_research_ledger.py")
        return 2
    verify_chain(entries)
    s = summarize(entries)
    print(f"Research ledger: {s['total_entries']} experiments (hash-chain verified)")
    print(f"  {'category':14}{'total':>7}{'accepted':>10}{'rejected':>10}")
    for cat in s["by_category"]:
        if s["by_category"][cat]:
            acc = s["accepted_by_category"][cat]
            print(f"  {cat:14}{s['by_category'][cat]:>7}{acc:>10}{s['by_category'][cat] - acc:>10}")
    print(f"\nSearch categories total (strategy+portfolio+execution+universe): {s['recommended_trial_count']}")
    print(f"Sharpe-valued trial results recorded: {s['sharpe_valued_trials']}")
    print("\nDefensible multiple-testing corrections count THESE, not just the "
          f"{sum(1 for e in entries if e['category'] == 'strategy')} strategy-family entries.")
    print("Rejected ideas stay in the file forever — the search was real even when the idea failed.")
    return 0


def run_verify_freeze(args) -> int:
    import json

    from .identity import verify_freeze_code

    try:
        raw = json.loads(open(args.freeze_file, encoding="utf-8").read())
    except FileNotFoundError:
        print(f"FAIL: {args.freeze_file} not found — nothing frozen to verify against")
        return 1
    # Experiment identity is reported FIRST, and unconditionally, because a
    # failed code seal hides everything after it — and "this tape is no longer
    # evidence of the current implementation" is the single most important
    # thing a reader of a superseded experiment needs to know.
    from .experiments import current_freeze_status, describe_freeze

    exp = describe_freeze(raw)
    print(f"experiment : {exp['experiment_version']} (accounting model {exp['accounting_model']})")
    if not exp["methodologically_current"]:
        print(f"STATUS     : {current_freeze_status(raw)}")
        print(f"             {exp['supersede_reason']}")
        print("             The tape is valid evidence of the implementation that produced")
        print("             it, NOT of the corrected implementation. A v2 freeze is needed")
        print("             for prospective evidence of the current code.")
    try:
        # config hash first for the clearer message on tampering
        from .prospective import _config_hash

        if _config_hash(raw["config"]) != raw.get("config_sha256"):
            print("FAIL: freeze.json config was modified after freezing (config_sha256 mismatch)")
            return 1
        verify_freeze_code(raw)
    except (ValueError, KeyError) as e:
        print(f"FAIL: {e}")
        return 1
    print("OK: running implementation matches the freeze")
    print(f"  frozen at : {raw['frozen_at']}")
    print(f"  commit    : {raw.get('git_commit_at_freeze')}")
    print(f"  code sha  : {raw.get('code_sha256')}")
    return 0


def _forward_fetch(symbol: str, source: str):
    from .data import fetch_daily_history, fetch_yahoo_daily, recent_window

    try:
        if source == "yahoo":
            candles = fetch_yahoo_daily(symbol)
        else:
            # FULL history then take the most recent window: max_candles alone
            # paginates from the FIRST bar and would hand the forward runner
            # 2017-era prices while staleness alerts scream about it.
            candles = recent_window(fetch_daily_history(symbol), 450)
        if not candles:
            return [], "empty history"
        return candles, None
    except Exception as e:
        return [], f"fetch failed: {e}"


def run_forward(args) -> int:
    from datetime import date as _date

    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .prospective import (
        alert_stats,
        checkpoints_due,
        experiment_stamp,
        forward_performance,
        load_freeze,
        load_log,
        monthly_returns,
        outage_stats,
        run_step,
        slippage_stats,
    )

    manifest = load_freeze(args.freeze_file)
    freeze_date = _date.fromisoformat(manifest["frozen_at_date"])

    if args.step:
        session_date = _date.fromisoformat(args.as_of_date) if args.as_of_date else None
        result = run_step(
            manifest,
            _forward_fetch,
            log_path=args.log_file,
            session_date=session_date,
        )
        e = result.get("entry")
        if e is not None:
            print(f"[{e['date']}] {result['status']}: port_ret {e['port_ret']:+.4%}, overlay {e['overlay_weight']:.2f}, "
                  f"assets {len(e['assets'])}, outages {len(e['outages'])}, missed_fills {len(e['missed_fills'])}")
        else:
            print(f"Forward step: {result['status']} ({result.get('date', 'n/a')})")

    entries = load_log(args.log_file)
    if not entries:
        print("No forward entries yet — run `python -m bot forward --step` daily")
        return 0

    perf = forward_performance(
        entries,
        freeze_date=freeze_date,
        risk_free_annual=float(manifest["config"]["frictions"].get("risk_free_annual", 0.0)),
        experiment=experiment_stamp(manifest),
    )
    total_return = perf["return"]
    as_of = _date.fromisoformat(entries[-1]["date"])

    print(f"\nProspective validation: frozen {freeze_date} -> last step {as_of} ({len(entries)} forward days)")
    if total_return is None:
        print("  bot forward: return unavailable (no measured forward path)")
    else:
        sharpe_text = f"{perf['sharpe']:.2f}" if perf["sharpe"] is not None else "withheld"
        mdd_text = f"{perf['max_drawdown']:.1%}" if perf["max_drawdown"] is not None else "n/a"
        ppy_text = f"{perf['periods_per_year']:.1f}/yr" if perf["periods_per_year"] is not None else "n/a"
        print(f"  bot forward: {1.0 + total_return:.3f}x ({total_return:+.1%}), Sharpe {sharpe_text}, maxDD {mdd_text}, cadence {ppy_text}")
        if perf["sharpe"] is None and perf["sharpe_reason"]:
            print(f"  Sharpe: {perf['sharpe_reason']}")

    cps = checkpoints_due(freeze_date, as_of)
    sp = fetch_sp500()
    sp_window = slice_window(sp, freeze_date, as_of)
    spx = equity_metrics(sp_window) if len(sp_window) > 3 else None
    print("\nCheckpoints (bot vs S&P 500 over the same span):")
    for cp in cps:
        if not cp["due"]:
            print(f"  {cp['label']:>10}: pending ({cp['elapsed']}/{cp['days']} days)")
        elif spx:
            if total_return is not None:
                print(f"  {cp['label']:>10}: bot {total_return:+.1%} vs S&P {(spx['final'] - 1):+.1%}  [elapsed {cp['elapsed']}d]")

    slip = slippage_stats(entries)
    out = outage_stats(entries)
    al = alert_stats(entries)
    print(f"\nIncidents: mean |decision->execution| {slip['mean_abs_bps'] or 0:.0f}bp over {slip['count']} turnover events; "
          f"{out['outage_days']} outage days ({out['outage_events']} events); {out['missed_fills']} missed fills")
    if al["total"]:
        print(f"Data-staleness alerts: {al['total']} ({al['by_level']}) on {', '.join(al['symbols'])}")

    from .cost_calibration import calibrate as _cal
    from .cost_calibration import format_report as _fmt
    from .cost_calibration import load_observations as _loadobs

    obs = _loadobs("cost_observations.jsonl")
    if obs:
        rep = _cal(obs, v1_frictions=manifest["config"]["frictions"], freeze_manifest=manifest)
        print("\n" + _fmt(rep))

    months = monthly_returns(entries)
    print("\nMonthly returns (negative periods published):")
    for m, r in months.items():
        marker = "  <- negative" if r < 0 else ""
        print(f"  {m}: {r:+.2%}{marker}")
    return 0


def _cum(rets):
    eq = 1.0
    for r in rets:
        eq *= 1.0 + r
        yield eq
