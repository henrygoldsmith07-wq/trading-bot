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
    canonical = _pl.Path(RUNS_DIR) / "canonical-v2" / "run.json"
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
    """The product: how much evidence actually supports the frozen system?

    RENDERS FROM THE EVIDENCE DOCUMENT. This command used to rebuild the
    verdict from canonical-v1's metrics while the README and API graded
    canonical-v2 — three surfaces, three sources. Now there is one, and no
    argument can retarget the claim at a different rule: the primary rule
    identity is resolved inside the evidence layer.
    """
    import json as _json

    from .evidence_model import render_verdict_sentence
    from .reporting import CANONICAL_RUN_ID, EvidenceError, build_evidence

    try:
        doc = build_evidence(args.root)
    except EvidenceError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if args.json:
        print(_json.dumps(doc, indent=2, sort_keys=True, allow_nan=False))
        return int(doc["verdict"].get("exit_code", 2))

    v = doc["verdict"]
    d = v.get("grades") or {}
    print("=" * 62)
    print("STRATEGY VERDICT")
    print("=" * 62)
    for key in ("historical_evidence", "walk_forward_robustness", "selection_bias_risk",
                "cost_robustness", "prospective_forward_evidence", "pre_registration"):
        print(f"{key:28}: {d.get(key)}")
    print("-" * 62)
    print(f"{'OVERALL':28}: {v.get('overall_grade')}")
    print("-" * 62)
    print(render_verdict_sentence(v))
    print(f"(graded from {CANONICAL_RUN_ID}; run `python -m bot evidence` for the full record)")
    return int(v.get("exit_code", 2))


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


# ---------------------------------------------------------------------------
# evidence architecture commands — every surface renders from one document
# ---------------------------------------------------------------------------

def run_evidence(args) -> int:
    """The single evidence report: what is tested, what proves it, how strong.

    This is the product. Every question a reader could ask of the repository is
    answered from ONE document (bot/reporting.build_evidence), so this output
    cannot disagree with the README, the API or the dashboard.
    """
    import json as _json
    import pathlib as _pl

    from .reporting import build_evidence, render_evidence_report

    doc = build_evidence(args.root)
    if args.out:
        from .reporting import evidence_bytes

        out_path = _pl.Path(args.out)
        out_path.write_bytes(evidence_bytes(doc))
        print(f"evidence document written to {out_path}")
        return 0
    if args.json:
        print(_json.dumps(doc, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_evidence_report(doc))
    return 0


def run_experiment_status(args) -> int:
    """Where the experiment is in its lifecycle, and why."""
    from .evidence_model import headline_block
    from .reporting import build_evidence

    doc = build_evidence(args.root)
    s = doc["status"]
    h = headline_block(doc)
    print(f"state          : {s['state']}")
    print(f"  {s['state_explanation']}")
    print(f"strategy       : {h['strategy']}")
    print(f"valid freeze   : {s['has_valid_freeze']}")
    if s.get("freeze_problem"):
        print(f"  freeze issue : {s['freeze_problem']}")
    if s.get("superseded_by"):
        print(f"  superseded by: {s['superseded_by']}")
    f = doc["forward"]
    print(f"forward days   : {f['clean_forward_days']} clean "
          f"(of {f['days_recorded']} recorded; {f['superseded_rows_not_counted']} row(s) belong to other experiments)")
    nc = s.get("next_checkpoint")
    if nc:
        print(f"next checkpoint: {nc['label']} — {nc['clean_days_remaining']} clean day(s) away ({nc['meaning']})")
    for cp in s.get("checkpoints", []):
        mark = "x" if cp["reached"] else " "
        print(f"  [{mark}] {cp['label']:28} {cp['meaning']}")
    print(f"biggest caveat : {h['biggest_caveat']}")
    return 0


def run_verify_evidence(args) -> int:
    """Pass/fail verification of the whole evidence chain, with exact reasons.

    Checks, in order: git provenance, source fingerprint, canonical run hash,
    freeze integrity, evidence-chain integrity (forward tape scoping),
    universe-chain integrity, experiment IDs, methodological currency, and
    README/evidence consistency. Fails closed: an unestablishable check is a
    failure with the reason stated, never a pass.
    """
    from .reporting import CANONICAL_RUN_ID, build_evidence, load_artifacts

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    artifacts = load_artifacts(args.root)
    try:
        doc = build_evidence(args.root, artifacts=artifacts)
    except Exception as exc:
        print(f"FAIL: evidence document cannot be built: {exc}")
        return 1

    prov = doc["provenance"]
    git_now = prov.get("git_now")
    record(
        "git provenance",
        bool(git_now and git_now.get("head_commit")),
        (f"HEAD {git_now['head_commit'][:12]} clean={not git_now['dirty']}" if git_now else "git state unavailable"),
    )

    record(
        "source fingerprint",
        bool(prov.get("source_fingerprint") or prov.get("experiment_fingerprint")),
        f"experiment fingerprint {str(prov.get('experiment_fingerprint'))[:16]}",
    )

    record(
        "canonical run hash",
        prov.get("canonical_record_intact") is True,
        {
            True: f"{CANONICAL_RUN_ID} record hash verified",
            False: f"{CANONICAL_RUN_ID} record hash MISMATCH — the record was edited after being written",
            None: f"no {CANONICAL_RUN_ID} record to verify",
        }[prov.get("canonical_record_intact")],
    )

    status = doc["status"]
    record(
        "freeze integrity",
        bool(status["has_valid_freeze"]),
        "config and code seals verify" if status["has_valid_freeze"] else (status.get("freeze_problem") or "no freeze committed"),
    )

    scope = doc["forward"].get("excluded_rows") or {}
    intact = not doc["forward"].get("integrity_error")
    record(
        "evidence chain integrity",
        intact,
        f"{scope.get('kept', 0)} of {scope.get('total', 0)} forward rows belong to this experiment; "
        f"{scope.get('excluded', 0)} excluded with reasons recorded",
    )

    chain = prov.get("universe_chain") or {}
    record(
        "universe chain integrity",
        bool(chain.get("snapshots")) and bool(chain.get("dates_monotone")),
        chain.get("note", "no universe snapshot information"),
    )

    exp = doc["experiment"]
    record(
        "experiment IDs",
        exp.get("primary_rule_id") == doc["verdict"]["primary_rule_id"],
        f"experiment {exp.get('experiment_id')} grades {doc['verdict']['primary_rule_id']}",
    )

    record(
        "methodological currency",
        bool(status["methodologically_current"]),
        "the frozen methodology is current" if status["methodologically_current"]
        else f"SUPERSEDED by {status.get('superseded_by')}: {status.get('supersede_reason')}",
    )

    # Registration integrity: a sealed plan that no longer matches its own seal
    # is an edited-after-sealing decision rule. That is the same class of
    # failure as an edited freeze or a broken ledger chain, so it fails the
    # gate. An ABSENT plan is not a failure — nobody is obliged to have one —
    # but a CORRUPT one is, because it means a committed artefact lies.
    reg = doc.get("registration") or {}
    n_drifted = int(reg.get("n_drifted", 0) or 0)
    drifted_detail = "; ".join(
        f"{p.get('file')}: {p.get('error')}" for p in reg.get("plans", []) if p.get("drifted")
    )
    record(
        "registration integrity",
        n_drifted == 0,
        (f"{reg.get('n_valid', 0)} valid pre-registration(s), all seals verify" if n_drifted == 0
         else f"{n_drifted} pre-registration(s) edited after sealing: {drifted_detail}"),
    )

    # Forward-tape liveness. Days-recorded alone cannot distinguish "the
    # experiment is young" from "the experiment quietly died": both report
    # zero, or a number that simply stops growing. That ambiguity cost this
    # repo its entire prospective record -- the tape last advanced on
    # 2026-09-22, while every scheduled run since reported success.
    #
    # This is a check evaluated at a moment in time, like `git_now`. It is not
    # evidence, it is not in the document, and it cannot move the fingerprint.
    # A young tape is not a failure; a tape that has stopped advancing while
    # the freeze is still current is, because no run is doing what it claims.
    record("forward tape liveness", *_forward_tape_liveness(doc, artifacts))

    # README/evidence consistency: the canonical block must render exactly as
    # the evidence document says. A stale README is a claim that no evidence
    # supports, so it fails the gate.
    readme_ok, readme_detail = _readme_consistency(args.root)
    record("README/evidence consistency", readme_ok, readme_detail)

    failures = [c for c in checks if not c[1]]
    print(f"verify-evidence: {len(checks) - len(failures)}/{len(checks)} checks pass")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28} {detail}")
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed — see reasons above")
        return 1
    print("\nPASS: every number in the evidence document is bound to its source")
    return 0


def _forward_tape_liveness(doc: dict, artifacts: dict | None = None) -> tuple[bool, str]:
    """Is the prospective tape still advancing?

    "0 clean forward days" is ambiguous, and the ambiguity is expensive: it is
    the same output for a young experiment and for one whose scheduled run
    silently stopped. This repo lost its entire prospective record to exactly
    that ambiguity -- the tape last advanced on 2026-09-22 while every run
    reported success, and nothing anywhere said the number had stopped moving.

    Wall clock is used HERE and nowhere else. This is a check run at a moment
    in time, like `git_now`; it is not evidence, it is not in the document, and
    it cannot move the fingerprint. The comparison is against the newest day
    the tape itself records, so a rebuild of unchanged artifacts still produces
    identical evidence -- only the pass/fail verdict of this one check ages.

    Reads the RAW tape, not the experiment-scoped rows. A row excluded because
    it predates experiment stamping still proves a run once happened, and its
    date still proves when the tape last moved. Using the scoped rows would
    report "the tape has not started" for a tape that stopped weeks ago --
    hiding precisely the failure this check exists to surface. The scoped count
    is what the VERDICT grades; liveness is a property of the tape itself.

    Deliberately NOT a failure when there is nothing to advance:
      * no freeze  -> no experiment is supposed to be running;
      * no tape    -> the experiment has not started.
    A stale tape with a current freeze is different: the run is claiming to
    produce evidence and is not.

    A SUPERSEDED freeze is reported separately, because the two cases have
    different causes and different fixes. A superseded methodology is *right*
    to stop accruing: its tape is evidence of the implementation that produced
    it, not of the code running now. Blaming "the scheduled run" for that would
    point at the wrong thing -- what is missing is a freeze of the current
    methodology. Both states still fail the gate, because either way the
    prospective test is producing nothing, and silence about that is how this
    repo lost two weeks of record.
    """
    from datetime import date as _date

    from .experiment_state import MAX_FORWARD_TAPE_GAP_DAYS

    status = doc.get("status") or {}

    if not status.get("has_valid_freeze"):
        return True, "no current freeze - no forward run is expected to advance"

    rows = (artifacts or {}).get("forward_rows") or []
    days = [r.get("date") for r in rows if isinstance(r.get("date"), str) and r.get("date")]
    if not days:
        return True, "no forward days recorded yet - the tape has not started"

    last_day = max(days)
    try:
        last = _date.fromisoformat(last_day)
    except ValueError:
        return False, f"forward tape's last day {last_day!r} is not a valid date"

    # A frozen experiment is expected to produce one day per calendar day. Allow
    # a week of slack for weekends, exchange holidays, outages and runner
    # backlog, so ordinary gaps are not reported as a dead experiment.
    age_days = (_date.today() - last).days
    if age_days <= MAX_FORWARD_TAPE_GAP_DAYS:
        return True, f"last forward day {last} ({age_days}d ago) is current"

    if not status.get("methodologically_current"):
        return False, (
            f"no prospective evidence is accruing: the frozen methodology is "
            f"SUPERSEDED by {status.get('superseded_by')} and its tape stopped at {last} "
            f"({age_days}d ago), while no freeze of the current methodology exists. "
            "The run is not at fault -- what is missing is a freeze of the current code"
        )

    return False, (
        f"forward tape stopped advancing: last day {last} is {age_days}d old "
        f"(limit {MAX_FORWARD_TAPE_GAP_DAYS}d) while the freeze is still current — "
        "the scheduled run is not producing evidence, and a green run that "
        "advances nothing is not progress"
    )


def _readme_consistency(root) -> tuple[bool, str]:
    from pathlib import Path as _P

    from .reporting import build_evidence as _be
    from .reporting import render_readme_block

    readme = _P(root) / "README.md"
    if not readme.exists():
        return False, "README.md not found"
    try:
        import scripts.derive_canonical_readme as dcr

        md = readme.read_text(encoding="utf-8")
        if dcr.BEGIN not in md or dcr.END not in md:
            return False, "canonical markers missing from README.md"
        rendered = render_readme_block(_be(root))
        current = md.split(dcr.BEGIN, 1)[1].split(dcr.END, 1)[0].strip()
        if current == rendered.strip():
            return True, "README canonical block matches the evidence document"
        return False, "README canonical block is out of sync — run `python scripts/derive_canonical_readme.py`"
    except Exception as exc:
        return False, f"README check error: {exc}"


def run_compare_experiments(args) -> int:
    """Compare two experiments, refusing to attribute a confounded delta.

    Every dimension that could explain a performance difference is shown:
    methodology, accounting model, universe, strategy, code and candidate pool.
    When more than one important variable changed, the comparison is labelled
    CONFUNDED and the metric difference is NOT presented as attributable to any
    single change.
    """
    from .experiments import describe_freeze
    from .reporting import load_artifacts

    def experiment_facts(run_id: str) -> dict:
        from pathlib import Path as _P

        artifacts = load_artifacts(args.root)
        record = None
        path = _P(args.root) / "runs" / run_id / "run.json"
        if path.exists():
            import json as _json

            record = _json.loads(path.read_text(encoding="utf-8"))
        results = (record or {}).get("results") or {}
        env = results.get("environment") or {}
        metrics = results.get("metrics") or {}
        freeze = artifacts["freeze"] if run_id == "current" else None
        return {
            "run_id": run_id,
            "accounting_model": (describe_freeze(freeze) if freeze else {}).get("accounting_model"),
            "methodology_version": env.get("code_fingerprint", {}).get("algo"),
            "code_fingerprint": env.get("code_fingerprint", {}).get("sha256"),
            "universe": results.get("universe"),
            "parameters": results.get("parameters"),
            "metrics": {k: metrics.get(k) for k in ("banded_rm", "spx")},
            "n_folds": results.get("n_folds"),
        }

    left = experiment_facts(args.left)
    right = experiment_facts(args.right)

    diffs: list[tuple[str, bool]] = [
        ("accounting_model", left.get("accounting_model") != right.get("accounting_model")),
        ("methodology_version", left.get("methodology_version") != right.get("methodology_version")),
        ("code_fingerprint", left.get("code_fingerprint") != right.get("code_fingerprint")),
        ("universe", left.get("universe") != right.get("universe")),
        ("parameters", left.get("parameters") != right.get("parameters")),
        ("n_folds", left.get("n_folds") != right.get("n_folds")),
    ]

    important_changes = [
        k for k, changed in diffs
        if changed and k in ("accounting_model", "methodology_version", "code_fingerprint", "universe", "parameters")
    ]
    confounded = len(important_changes) > 1

    print(f"compare {args.left} vs {args.right}")
    for key, changed in diffs:
        print(f"  {key:22} {'DIFFERS' if changed else 'identical'}")
    lm = (left["metrics"] or {}).get("banded_rm") or {}
    rm = (right["metrics"] or {}).get("banded_rm") or {}
    if lm and rm:
        print(f"\nperformance delta (primary rule): CAGR {lm['cagr'] - rm['cagr']:+.4f}, "
              f"Sharpe {lm['sharpe'] - rm['sharpe']:+.3f}, maxDD {lm['max_drawdown'] - rm['max_drawdown']:+.4f}")
    if confounded:
        print("\nCONFUNDED COMPARISON")
        print(f"  {len(important_changes)} important variables changed simultaneously: {', '.join(important_changes)}")
        print("  the metric difference is NOT attributable to any single change")
        return 2
    if not any(changed for _, changed in diffs):
        print("\ncontrolled comparison: no recorded inputs differ")
    else:
        single = important_changes[0] if important_changes else "metadata only"
        print(f"\ncontrolled comparison: exactly one important variable differs ({single})")
    return 0


# ---------------------------------------------------------------------------
# Pre-analysis plans: decide what counts BEFORE seeing the result
# ---------------------------------------------------------------------------

def run_register(args) -> int:
    """Write a sealed pre-analysis plan, or report why it cannot be sealed.

    Exit codes carry the same meaning as elsewhere in this CLI: 0 success,
    1 an honest refusal (a malformed plan), 2 a usage error. A plan that
    cannot be sealed is never written "mostly" — a registration whose fields
    are inconsistent is not a weaker registration, it is a different and
    unusable one.
    """
    from .identity import code_fingerprint
    from .registration import Arm, RegistrationError, create_registration, write_registration

    arms: list[Arm] = []
    for spec in args.arm:
        # "id:description[:primary]" — primary is marked here, in advance.
        parts = spec.split(":", 2)
        if len(parts) < 2:
            print(f"FAIL: arm {spec!r} must be 'id:description[:primary]'")
            return 2
        arm_id, description = parts[0], parts[1]
        primary = len(parts) > 2 and parts[2].strip().lower() in ("primary", "true", "yes", "1")
        arms.append(Arm(arm_id, description, primary=primary))

    try:
        fingerprint = code_fingerprint()
    except Exception:  # noqa: BLE001 - fingerprinting must never block a registration
        fingerprint = None

    try:
        reg = create_registration(
            registration_id=args.id,
            title=args.title,
            hypothesis=args.hypothesis,
            primary_metric=args.metric,
            threshold=args.threshold,
            direction=args.direction,
            min_evidence=args.min_evidence,
            alpha=args.alpha,
            arms=arms,
            stopping_rule=args.stopping,
            declared_trials=max(args.declared_trials, len(arms)),
            data_window=args.window,
            code_fingerprint=fingerprint,
            notes=args.notes or "",
        )
        path = write_registration(reg, args.dir)
    except RegistrationError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Registered pre-analysis plan -> {path}")
    print(f"  seal        : {reg.seal()}  (content-addressed; editing any field breaks it)")
    print(f"  primary arm : {reg.primary_arm.arm_id} — {reg.primary_arm.description}")
    print(f"  criterion   : {reg.primary_metric} {reg.direction} {reg.threshold} on >= {reg.min_evidence} observation(s)")
    print(f"  arms        : {len(reg.arms)} declared; a read across them is Holm-Bonferroni corrected")
    print(f"  stopping    : {reg.stopping_rule}")
    print("\nA result may only be graded against this plan. Anything graded without one is exploratory.")
    return 0


def run_registration_status(args) -> int:
    """List registered plans and report whether each has been read.

    The read log is the part that matters: it turns "we looked once, at the
    time we said we would" into something a reader can check, and it makes a
    second look at a losing tape visible.
    """
    from pathlib import Path

    from .registration import DEFAULT_EVENT_LOG, RegistrationError, load_events, load_registration

    directory = args.dir
    plan_files = sorted(Path(directory).glob("*.json")) if Path(directory).exists() else []
    log_path = args.log or DEFAULT_EVENT_LOG

    try:
        events = load_events(log_path, strict=False)
    except RegistrationError as exc:
        print(f"FAIL: registration event log is unusable: {exc}")
        return 1

    if not plan_files:
        print(f"No registrations in {directory}/ — nothing has been pre-decided yet.")
        print("Without a registration, every result in this repo is exploratory by definition.")
        return 2

    print(f"Pre-analysis plans in {directory}/  (event log: {log_path}, {len(events)} event(s))\n")
    unreadable = 0
    for path in plan_files:
        try:
            reg = load_registration(path)
        except RegistrationError as exc:
            unreadable += 1
            print(f"  {path.name}: CORRUPT — {exc}")
            continue
        reads = [e for e in events if e.get("registration_id") == reg.registration_id
                 and e.get("event_type") == "read"]
        state = f"read {len(reads)}x" if reads else "NOT YET READ"
        print(f"  {path.name}")
        print(f"    seal     : {reg.seal()}")
        print(f"    primary  : {reg.primary_arm.arm_id} — {reg.primary_metric} {reg.direction} {reg.threshold}")
        print(f"    arms     : {len(reg.arms)}   minimum evidence: {reg.min_evidence}")
        print(f"    status   : {state}")
    if unreadable:
        print(f"\n{unreadable} registration(s) failed integrity and are excluded from any claim.")
    return 1 if unreadable else 0
