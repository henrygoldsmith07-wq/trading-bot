"""The canonical compare/benchmark pipeline.

The walk-forward portfolio-vs-index evaluation that produces a canonical
run record. This was the single largest function in the CLI entry point; it
lives here so the claim-making logic (which rule is primary, how the verdict
is worded, what counts as evidence) is a reviewable module rather than a wall
inside one file.

It takes an argparse namespace and returns the results dict, exactly as
before, so the CLI contract is unchanged.
"""
from __future__ import annotations

import pathlib as _pl
import platform
import time as _time

from .canonical_identity import (
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
    build_primary_verdict,
)
from .compare_common import (
    CODE_FP_ALGO,
    DAY_MS,
    as_date,
    fmt_pct,
    print_regimes,
    vol_overlay,
)
from .compare_common import (
    equity_metrics as returns_metrics,
)
from .identity import code_fingerprint, source_file_fingerprint
from .provenance import SEALED_MODULES, DirtyTreeError, git_state


def _environment_block(args):
    """Provenance for a canonical record, from one place.

    Records HEAD, the source fingerprint, per-module hashes, and whether the
    tree was clean. A canonical run refuses a dirty tree unless the caller
    explicitly asks for a non-canonical run.
    """

    from .canonical_identity import PRIMARY_RULE_ID

    # Only a CANONICAL run must be refused on a dirty tree. Exploratory and
    # development runs are legitimate from a working tree; they are simply
    # recorded as non-canonical, and the record says so.
    run_id = str(getattr(args, "run_id", "") or "")
    is_canonical = bool(getattr(args, "canonical", False)) or run_id.lower().startswith("canonical")
    # the tree you ran in is the tree that produced the evidence
    state = git_state(_pl.Path.cwd())
    if is_canonical and state.dirty and not getattr(args, "allow_dirty_tree", False):
        from .provenance import DIRTY_REFUSAL_HINT

        raise DirtyTreeError(
            f"refusing to create the canonical run {run_id!r} from a dirty working tree."
            f"\n  {state.describe()}\n  {DIRTY_REFUSAL_HINT}"
        )

    return {
        "python": platform.python_version(),
        "git_commit": state.head_commit,
        "code_fingerprint": {"algo": CODE_FP_ALGO, "sha256": code_fingerprint()},
        "module_hashes": source_file_fingerprint(list(SEALED_MODULES))["files"],
        "source_clean": not state.dirty,
        "dirty_paths": list(state.dirty_paths),
        "is_canonical": is_canonical,
        "primary_rule_id": PRIMARY_RULE_ID,
        "strategy_definitions_hash": source_file_fingerprint(["bot/strategy.py"]),
        "portfolio_rules_hash": source_file_fingerprint(["bot/portfolio_rules.py"]),
        "universe_hash": source_file_fingerprint(["bot/universe.py", "bot/universe_pit.py"]),
    }

def compute_compare_results(args, fetch=None, log=print, save_run=True):
    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .cache import load_or_fetch_meta
    from .data import extend_returns_to_timeline, fetch_daily_history, fetch_yahoo_daily, is_stale
    from .universe import ETF_UNIVERSE, top_symbols
    from .walkforward import absolute_folds, combine_portfolio, combine_portfolio_invvol, walk_forward_at

    def _real_fetch(kind):
        if fetch is not None:
            fn = fetch.get(kind) if isinstance(fetch, dict) else fetch
            if fn is not None:
                return fn
        return {"crypto": fetch_daily_history, "yahoo": fetch_yahoo_daily, "sp500": fetch_sp500}[kind]

    t_start = _time.perf_counter()
    datasets_meta: dict[str, dict] = {}
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        latency_days=args.latency_days,
        execution=args.execution,
        risk_free_annual=args.risk_free,
    )

    download_stamps: dict[str, float | None] = {}

    def cached(symbol, kind):
        real_fetch = _real_fetch(kind)
        candles, _from_cache, fetched_at = load_or_fetch_meta(
            symbol, lambda s: real_fetch(s), cache_only=getattr(args, "cache_only", False)
        )
        download_stamps[symbol] = fetched_at
        return candles

    btc = cached("BTCUSDT", "crypto")
    folds_abs = absolute_folds(btc, args.train_days, args.test_days)
    if not folds_abs:
        log("Not enough BTC history for one walk-forward fold")
        return {"exit_code": 2}
    oos_start_ms, oos_end_ms = folds_abs[0][0], folds_abs[-1][1]

    min_history = args.train_days + args.test_days + 180
    # The CLI passes a comma-separated string; accept either form so the
    # override is usable from argparse and from a direct call.
    override = getattr(args, "universe_symbols", None)
    if isinstance(override, str):
        override = [s.strip() for s in override.split(",") if s.strip()]
    universe = list(override) if override else ["BTCUSDT"] + [s for s in top_symbols(args.assets) if s != "BTCUSDT"]
    universe = universe[: len(override) if override else args.assets]
    specs: list[tuple[str, str, int]] = [(s, "crypto", 365) for s in universe] + [
        (e["symbol"], e["asset_class"], e["periods_per_year"]) for e in ETF_UNIVERSE
    ]

    if override:
        # Stated loudly: a pinned universe is what makes two runs comparable.
        log(f"Universe: PINNED to {len(universe)} symbols (--universe-symbols), NOT the live volume ranking")
    else:
        log(f"Universe: {args.assets} crypto pairs by quote volume + {len(ETF_UNIVERSE)} cross-class ETFs")
        log("  (live ranking — a re-run may select DIFFERENT symbols than the record it is compared to)")
    log("  (paper-only validation; no live orders are ever placed)")
    histories: dict[str, tuple[list[dict], int]] = {}
    skipped: list[tuple[str, str]] = []
    for symbol, asset_class, ppy in specs:
        try:
            kind = "crypto" if asset_class == "crypto" else "yahoo"
            candles = cached(symbol, kind)
        except Exception as e:
            skipped.append((symbol, f"fetch failed: {e}"))
            continue
        if len(candles) < min_history:
            skipped.append((symbol, f"only {len(candles)} candles (<{min_history})"))
            continue
        if is_stale(candles):
            skipped.append((symbol, "history stopped >45d ago (delisted/stale)"))
            continue
        log(f"  {symbol:10} [{asset_class:16}] {len(candles)} candles since {as_date(candles[0]['open_time'])}")
        histories[symbol] = (candles, ppy)
    for symbol, why in skipped:
        log(f"  {symbol:10} skipped: {why}")
    if not histories:
        log("No tradeable assets with enough history")
        return {"exit_code": 2}
    t_fetch = _time.perf_counter()

    timeline = [c["open_time"] for c in btc if oos_start_ms <= c["open_time"] < oos_end_ms]
    n_selected = len(histories)
    asset_dailies = {}
    banded_dailies = {}
    fixed_dailies = {}
    per_asset = []
    picks_counter: dict[str, int] = {}
    from .strategy import risk_ensemble

    for symbol, (candles, ppy) in histories.items():
        wf = walk_forward_at(candles, folds_abs, periods_per_year=ppy, **engine_kwargs)
        per_asset.append({"symbol": symbol, "cagr": wf["cagr"], "sharpe": wf["sharpe"], "max_drawdown": wf["max_drawdown"]})
        for p in wf["folds"]:
            picks_counter[p["strategy"]] = picks_counter.get(p["strategy"], 0) + 1
        asset_dailies[symbol] = extend_returns_to_timeline(
            {t: r for t, r in wf["daily"].items() if oos_start_ms <= t < oos_end_ms}, timeline
        )

        banded_kwargs = dict(engine_kwargs)
        banded_kwargs["rebalance_band"] = 0.05
        wb = walk_forward_at(candles, folds_abs, periods_per_year=ppy, **banded_kwargs)
        banded_dailies[symbol] = extend_returns_to_timeline(
            {t: r for t, r in wb["daily"].items() if oos_start_ms <= t < oos_end_ms}, timeline
        )

        wfx = walk_forward_at(candles, folds_abs, periods_per_year=ppy, candidates=[risk_ensemble()], **banded_kwargs)
        fixed_dailies[symbol] = extend_returns_to_timeline(
            {t: r for t, r in wfx["daily"].items() if oos_start_ms <= t < oos_end_ms}, timeline
        )
    t_compute = _time.perf_counter()

    from .portfolio_rules import combine_portfolio_rule

    port_returns = combine_portfolio(asset_dailies, timeline, n_selected)
    iv_returns = combine_portfolio_invvol(asset_dailies, timeline, n_selected)
    # point-in-time survivorship control: per-day ELIGIBILITY from each asset's
    # OWN history (listing age, trailing dollar volume, alive-at-day). This is
    # a real MEMBERSHIP MASK, not just a denominator: an asset that is not
    # eligible on day t is removed from that day's contributors entirely, so
    # it cannot contribute return or allocation before it was holdable.
    from .universe_pit import point_in_time_universe

    pit = point_in_time_universe({s: c for s, (c, _) in histories.items()}, timeline)
    eligible_by_day = {t: set(pit[t]) for t in timeline}
    denominator_by_day = {t: max(1, len(pit[t])) for t in timeline}
    elig_counts = [denominator_by_day[t] for t in timeline]
    if elig_counts:
        log(f"\nPoint-in-time eligibility (listed+aged+liquid at each date): "
              f"min {min(elig_counts)}, median {sorted(elig_counts)[len(elig_counts)//2]}, max {max(elig_counts)} of {n_selected} fetched")
        log("  ineligible sleeves are MASKED OUT of that day (no return, no weight); "
            "their capital stays in cash rather than being redistributed")
    port_returns = combine_portfolio(asset_dailies, timeline, n_selected, eligible_by_day=eligible_by_day)
    iv_returns = combine_portfolio_invvol(asset_dailies, timeline, n_selected, eligible_by_day=eligible_by_day)
    base_rule: dict = dict(use_tilt=True, use_crisis=True, eligible_by_day=eligible_by_day)
    full_returns = combine_portfolio_rule(asset_dailies, timeline, n_selected, **base_rule)
    throttle_returns = combine_portfolio_rule(asset_dailies, timeline, n_selected, use_dd_throttle=True, **base_rule)
    banded_returns = combine_portfolio_rule(banded_dailies, timeline, n_selected, **base_rule)
    fixed_returns = combine_portfolio_rule(fixed_dailies, timeline, n_selected, use_dd_throttle=True, **base_rule)

    port = returns_metrics(port_returns, risk_free_annual=args.risk_free)
    port_rm = returns_metrics(vol_overlay(port_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    iv_rm = returns_metrics(vol_overlay(iv_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    full_rm = returns_metrics(vol_overlay(full_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    throttle_rm = returns_metrics(vol_overlay(throttle_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    banded_rm = returns_metrics(vol_overlay(banded_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    fixed_rm = returns_metrics(vol_overlay(fixed_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)

    log("\nFetching S&P 500 daily history (FRED)...")
    from datetime import date as _date

    def _sp_rows():
        rows = _real_fetch("sp500")()
        normalized = []
        for r in rows:
            d = r["date"]
            d = d.isoformat() if hasattr(d, "isoformat") else str(d)
            normalized.append({"date": d, "close": float(r["close"])})
        return normalized

    sp_blob, _sp_cached, sp_fetched = load_or_fetch_meta("SP500", lambda s: _sp_rows(),
                                                         cache_only=getattr(args, "cache_only", False))
    download_stamps["SP500"] = sp_fetched
    sp = [{"date": _date.fromisoformat(r["date"]), "close": r["close"]} for r in sp_blob]
    sp_window = slice_window(sp, as_date(oos_start_ms), as_date(oos_end_ms - DAY_MS))
    spx = equity_metrics(sp_window, risk_free_annual=args.risk_free)

    bh_returns = []
    for i in range(1, len(btc)):
        if oos_start_ms <= btc[i]["open_time"] < oos_end_ms:
            bh_returns.append(btc[i]["close"] / btc[i - 1]["close"] - 1.0)
    bh = returns_metrics(bh_returns, risk_free_annual=args.risk_free)

    log()
    log(f"Out-of-sample window: {as_date(oos_start_ms)} -> {as_date(oos_end_ms - DAY_MS)} ({len(folds_abs)} yearly folds, {n_selected} assets)")
    log("Most-picked strategies across all folds and assets:")
    for name, count in sorted(picks_counter.items(), key=lambda kv: -kv[1])[:5]:
        log(f"  {name} x{count}")
    log()
    header = f"{'':18}{'Bot inv-vol':>14}{'Bot equal':>13}{'Bot raw eq':>12}{'S&P 500':>12}{'BTC b&h':>11}"
    log(header)
    log("-" * len(header))
    log(f"{'CAGR':18}{fmt_pct(iv_rm['cagr']):>14}{fmt_pct(port_rm['cagr']):>13}{fmt_pct(port['cagr']):>12}{fmt_pct(spx['cagr']):>12}{fmt_pct(bh['cagr']):>11}")
    log(f"{'Volatility':18}{fmt_pct(iv_rm['vol']):>14}{fmt_pct(port_rm['vol']):>13}{fmt_pct(port['vol']):>12}{fmt_pct(spx['vol']):>12}{fmt_pct(bh['vol']):>11}")
    log(f"{'Sharpe (excess)':18}{iv_rm['sharpe']:>14.2f}{port_rm['sharpe']:>13.2f}{port['sharpe']:>12.2f}{spx['sharpe']:>12.2f}{bh['sharpe']:>11.2f}")
    log(f"{'Max drawdown':18}{fmt_pct(iv_rm['max_drawdown']):>14}{fmt_pct(port_rm['max_drawdown']):>13}{fmt_pct(port['max_drawdown']):>12}{fmt_pct(spx['max_drawdown']):>12}{fmt_pct(bh['max_drawdown']):>11}")
    log(f"{'Sortino':18}{iv_rm['sortino']:>14.2f}{port_rm['sortino']:>13.2f}{port['sortino']:>12.2f}{spx['sortino']:>12.2f}{bh['sortino']:>11.2f}")
    log(f"{'Calmar':18}{iv_rm['calmar']:>14.2f}{port_rm['calmar']:>13.2f}{port['calmar']:>12.2f}{spx['calmar']:>12.2f}{bh['calmar']:>11.2f}")
    log(f"{'ES 95% (1d)':18}{fmt_pct(iv_rm['es95']):>14}{fmt_pct(port_rm['es95']):>13}{fmt_pct(port['es95']):>12}{fmt_pct(spx['es95']):>12}{fmt_pct(bh['es95']):>11}")
    log(f"{'Growth of $1':18}{iv_rm['final']:>14.2f}{port_rm['final']:>13.2f}{port['final']:>12.2f}{spx['final']:>12.2f}{bh['final']:>11.2f}")

    log("\nFixed portfolio rules (a-priori overlays, all risk-managed):")
    rules = [
        ("inv-vol (selected underlying)", iv_rm, vol_overlay(iv_returns, target=args.portfolio_vol)),
        ("+ tilt + crisis de-risk", full_rm, vol_overlay(full_returns, target=args.portfolio_vol)),
        ("+ drawdown throttle", throttle_rm, vol_overlay(throttle_returns, target=args.portfolio_vol)),
        ("+ tilt + crisis, banded 5% rebalance", banded_rm, vol_overlay(banded_returns, target=args.portfolio_vol)),
        ("fully-fixed: RiskEnsemble everywhere, banded, all overlays", fixed_rm, vol_overlay(fixed_returns, target=args.portfolio_vol)),
    ]
    header = f"  {'rule':58}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'ES95':>7}{'Calmar':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for name, m, _ in rules:
        log(f"  {name:58}{fmt_pct(m['cagr']):>8}{m['sharpe']:>8.2f}{fmt_pct(m['max_drawdown']):>8}{fmt_pct(m['es95']):>7}{m['calmar']:>8.2f}")

    from .stats_validation import format_dsr, selection_adjusted_stats

    # Every rule below rides on a per-asset strategy chosen by walk-forward
    # selection, inside a research program recorded in research_ledger.jsonl.
    # So none of them is a pre-specified single trial, and a trial-count-1 DSR
    # would correct for nothing. The trial count comes from the ledger plus the
    # pool this run actually searched, and is reported as unavailable when it
    # cannot be established.
    candidate_pool_size = len(picks_counter)
    rule_stats = []
    for name, m, r_ in rules:
        _adj = selection_adjusted_stats(r_, pool_size=candidate_pool_size, prespecified=False)
        row = {"name": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
               "max_drawdown": m["max_drawdown"], "es95": m["es95"],
               "calmar": m["calmar"], "final": m["final"],
               "psr": round(_adj["psr"], 3),
               "dsr": round(_adj["dsr"], 3) if _adj["dsr_available"] else None,
               "dsr_available": _adj["dsr_available"],
               "dsr_unavailable_reason": _adj["dsr_unavailable_reason"],
               "dsr_n_trials": _adj["n_trials"],
               "dsr_trial_count_sources": _adj["trial_count_sources"],
               "prespecified": False}
        rule_stats.append(row)
    # attached to metrics after its construction below

    # ---- baselines: cash / momentum / mean-reversion / buy&hold -----------
    from .engine import run_strategy as _run_strat
    from .sensitivity import _summarize_returns
    from .strategy import RsiDipBuy as _MR_cls
    from .strategy import SmaCrossover as _Mom_cls

    baselines_out = {}
    baseline_defs = [
        ("cash (risk-free)", None),
        ("momentum SMA10/50", _Mom_cls(10, 50).weight_at),
        ("mean-reversion RSI-2", _MR_cls(2, 60, 150).weight_at),
        ("buy&hold", lambda c, i: 1.0),
    ]
    for bname, bfn in baseline_defs:
        if bfn is None:  # cash: pure risk-free accrual on the OOS timeline
            rf_daily = engine_kwargs["risk_free_annual"] / 365
            b_rets = {t: rf_daily for t in timeline}
        else:
            bres = _run_strat(btc, bfn, **engine_kwargs)
            b_rets = {t: r for t, r in bres["return_days"]
                      if oos_start_ms <= t < oos_end_ms}
        days_b = sorted(b_rets)
        if days_b:
            baselines_out[bname] = _summarize_returns([b_rets[t] for t in days_b])
    log("\nBaselines (BTC, same frictions & window):")
    for bname, mm in baselines_out.items():
        log(f"  {bname:28} CAGR {fmt_pct(mm['cagr']):>7}  Sharpe {mm['sharpe']:>5.2f}  "
            f"maxDD {fmt_pct(mm['max_drawdown']):>7}")

    log("\nStatistical standing (deflated against the recorded search, not a single trial):")
    for row in rule_stats:
        adj = {"dsr_available": row["dsr_available"],
               "dsr": row["dsr"],
               "n_trials": row["dsr_n_trials"],
               "dsr_unavailable_reason": row["dsr_unavailable_reason"]}
        log(f"  {row['name']:58} PSR {row['psr']:.3f}  {format_dsr(adj)}")
    _unavail = [r_["name"] for r_ in rule_stats if not r_["dsr_available"]]
    if _unavail:
        log(f"  {len(_unavail)} rule(s) report DSR as unavailable rather than as an uncorrected number.")
    log(f"\nFrictions: execution={args.execution}, fee={args.fee:.2%}, spread={args.spread_bps:.0f}bp, "
          f"slippage={args.slippage_bps:.0f}bp, latency={args.latency_days}d, cash yield={args.risk_free:.0%}/yr")
    log("Benchmark consistency: same window/calendar-day CAGR; Sharpe in excess of the same risk-free rate; index is untradeable so carries no costs.")

    log("\nPer-asset out-of-sample results:")
    for a in sorted(per_asset, key=lambda x: -x["sharpe"]):
        log(f"  {a['symbol']:10} CAGR {fmt_pct(a['cagr']):>7}  Sharpe {a['sharpe']:>5.2f}  maxDD {fmt_pct(a['max_drawdown']):>7}")

    print_regimes(btc, timeline, port_returns, sp_window, args)

    log(f"\nTiming: fetch/cache {t_fetch - t_start:.1f}s, walk-forward compute {t_compute - t_fetch:.1f}s, total {_time.perf_counter() - t_start:.1f}s")
    log("Survivorship control: per-day denominators come from POINT-IN-TIME eligibility (listing age + trailing dollar volume, bot/universe_pit.py), and daily snapshots accumulate in universe_log.jsonl. Residual: symbols purged from Binance before first fetch remain invisible.")

    # THE CANONICAL VERDICT IS ABOUT THE PRIMARY RULE.
    #
    # This used to grade "port_rm", the risk-managed EQUAL-WEIGHT portfolio,
    # which is a comparator and not the strategy the freeze actually trades.
    # That made the headline report a benchmark win the frozen rule never
    # earned. The primary rule identity lives in bot/canonical_identity.py and
    # every other headline surface resolves the same constant.

    # Bind the identity to the block actually computed in THIS run, and fail
    # closed if the canonical rule was not produced at all. A headline must
    # never fall back onto whichever portfolio happens to exist.
    _blocks = {
        "inv_vol_rm": iv_rm,
        "full_rm": full_rm,
        "throttle_rm": throttle_rm,
        "banded_rm": banded_rm,
        "fixed_rm": fixed_rm,
        "equal_rm": port_rm,
    }
    if PRIMARY_RULE_ID not in _blocks:
        raise RuntimeError(f"primary rule {PRIMARY_RULE_ID} was not computed by this run; refusing to headline another one")
    primary = _blocks[PRIMARY_RULE_ID]
    _verdict = build_primary_verdict(primary, spx)
    verdict_txt = _verdict["verdict"]
    log(f"\nPrimary rule under evaluation: {PRIMARY_RULE_ID} = {PRIMARY_RULE_DESCRIPTION}")
    log()
    log(f"VERDICT: {verdict_txt}")

    # ---------------- reproducibility record -------------------------------

    def _iso(ms):
        return as_date(ms).isoformat()

    for s_name in list(histories) + ["SP500"]:
        src_candles = histories[s_name][0] if s_name in histories else sp_blob
        from .snapshot import dataset_hash

        datasets_meta[s_name] = {
            "sha256": dataset_hash(src_candles),
            "provider": ("binance" if s_name.endswith("USDT") or s_name == "SP500" and False else
                         ("yahoo" if s_name in {"SPY", "GLD", "TLT"} else
                          ("fred" if s_name == "SP500" else "binance"))),
            "downloaded_at": download_stamps.get(s_name),
            "start": (src_candles[0]["open_time"] if s_name != "SP500" else None),
            "end": (src_candles[-1]["open_time"] if s_name != "SP500" else None),
        }
        if s_name == "SP500":
            datasets_meta[s_name]["provider"] = "fred"
            datasets_meta[s_name]["start"] = sp_window[0]["date"].isoformat() if sp_window else None
            datasets_meta[s_name]["end"] = sp_window[-1]["date"].isoformat() if sp_window else None

    params = {k: v for k, v in vars(args).items()}
    metrics = {
        "rules": rule_stats,
        "window": {"start": as_date(oos_start_ms).isoformat(), "end": as_date(oos_end_ms - DAY_MS).isoformat()},
        "equal_raw": port, "inv_vol_rm": iv_rm, "equal_rm": port_rm,
        "full_rm": full_rm, "throttle_rm": throttle_rm,
        "banded_rm": banded_rm, "fixed_rm": fixed_rm,
        "spx": spx, "btc_bh": bh,
    }
    results = {
        "exit_code": _verdict["exit_code"],
        "verdict": verdict_txt,
        "primary_rule_id": PRIMARY_RULE_ID,
        "primary_rule_description": PRIMARY_RULE_DESCRIPTION,
        "metrics": metrics,
        "per_asset": sorted(per_asset, key=lambda x: -x["sharpe"]),
        "picks_counter": picks_counter,
        "datasets": datasets_meta,
        "oos_window": {"start": _iso(oos_start_ms), "end": _iso(oos_end_ms - DAY_MS)},
        "n_folds": len(folds_abs),
        "n_assets_selected": n_selected,
        "universe": universe,
        "pit_eligibility": {"min": min(elig_counts), "median": sorted(elig_counts)[len(elig_counts) // 2],
                            "max": max(elig_counts)} if elig_counts else None,
        "environment": _environment_block(args),
        "seeds": {"seed": getattr(args, "seed", 42), "note": "compare pipeline is deterministic; seed reserved"},
        "parameters": params,
        "timing_s": {
            "fetch_cache": round(t_fetch - t_start, 3),
            "walk_forward": round(t_compute - t_fetch, 3),
            "total": round(_time.perf_counter() - t_start, 3),
        },
    }
    metrics["baselines"] = baselines_out
    if save_run:
        from .runs import save_run_record

        run_id = save_run_record(results, run_id=getattr(args, "run_id", None))
        log(f"run saved: runs/{run_id}/run.json")
        results["run_id"] = run_id
    return results
