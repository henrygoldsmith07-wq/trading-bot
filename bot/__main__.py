"""CLI entry point.

Usage:
  python -m bot backtest --symbol BTCUSDT --interval 1h
  python -m bot trade    --symbol BTCUSDT [--once]   # paper loop (paper only!)
  python -m bot compare  [--assets 20]                # portfolio vs S&P 500
  python -m bot sensitivity [--symbol BTCUSDT]        # backtesting-quality sweeps
  python -m bot validate   [--symbol BTCUSDT]         # statistical battery
  python -m bot research   [--symbol BTCUSDT]         # methodology battery:
      # SPA test, effective trial count / clustering / duplicates,
      # family ablation, Bayesian Sharpe, drawdown CIs, sequence risk,
      # probability-of-underperformance vs buy&hold
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import UTC, datetime

from .backtest import backtest
from .data import fetch_candles
from .strategy import SmaCrossover

DAY_MS = 86_400_000


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _equity_metrics(returns: list[float], periods_per_year: int = 365, risk_free_annual: float = 0.0) -> dict:
    from .metrics import cagr, calmar, expected_shortfall, max_drawdown, sharpe, sortino, var_hist, volatility

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = len(returns)  # daily returns: one calendar day each
    mdd = max_drawdown(equity)
    cagr_v = cagr(equity, days)
    return {
        "final": equity[-1],
        "cagr": cagr_v,
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "sortino": sortino(returns, periods_per_year, risk_free_annual),
        "calmar": calmar(cagr_v, mdd),
        "max_drawdown": mdd,
        "var95": var_hist(returns, 0.95),
        "es95": expected_shortfall(returns, 0.95),
    }


def _vol_overlay(returns: list[float], target: float = 0.25, window: int = 20, fee: float = 0.0015) -> list[float]:
    """Scale portfolio exposure to a trailing-vol target.

    The weight for day t uses only returns up to t-1 — no lookahead. This is
    the risk-management layer that de-risks a crypto portfolio to
    equity-like volatility.
    """
    import math

    out = []
    w = 0.0
    for i, r in enumerate(returns):
        if i >= window:
            hist = returns[i - window : i]
            m = sum(hist) / window
            var = sum((x - m) ** 2 for x in hist) / (window - 1)
            rv = math.sqrt(max(var, 0.0) * 365)
            w_new = min(1.0, target / rv) if rv > 0 else 1.0
        else:
            w_new = 1.0
        out.append(w_new * r - fee * abs(w_new - w))
        w = w_new
    return out


def _d(ms):
    return datetime.fromtimestamp(ms / 1000, tz=UTC).date()


def run_compare(args) -> int:
    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .cache import load_or_fetch
    from .data import extend_returns_to_timeline, fetch_daily_history, fetch_yahoo_daily, is_stale
    from .universe import ETF_UNIVERSE, top_symbols
    from .walkforward import absolute_folds, combine_portfolio, combine_portfolio_invvol, walk_forward_at

    t_start = _time.perf_counter()
    engine_kwargs = dict(
        fee=args.fee,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        latency_days=args.latency_days,
        execution=args.execution,
        risk_free_annual=args.risk_free,
    )

    def cached(symbol, fetcher):
        return load_or_fetch(symbol, lambda s: fetcher(s))[0]

    btc = cached("BTCUSDT", lambda s: fetch_daily_history(s))
    folds_abs = absolute_folds(btc, args.train_days, args.test_days)
    if not folds_abs:
        print("Not enough BTC history for one walk-forward fold")
        return 2
    oos_start_ms, oos_end_ms = folds_abs[0][0], folds_abs[-1][1]

    min_history = args.train_days + args.test_days + 180
    universe = ["BTCUSDT"] + [s for s in top_symbols(args.assets) if s != "BTCUSDT"]
    universe = universe[: args.assets]
    specs: list[tuple[str, str, int]] = [(s, "crypto", 365) for s in universe] + [
        (e["symbol"], e["asset_class"], e["periods_per_year"]) for e in ETF_UNIVERSE
    ]

    print(f"Universe: {args.assets} crypto pairs by quote volume + {len(ETF_UNIVERSE)} cross-class ETFs")
    print("  (paper-only validation; no live orders are ever placed)")
    histories: dict[str, tuple[list[dict], int]] = {}
    skipped: list[tuple[str, str]] = []
    for symbol, asset_class, ppy in specs:
        try:
            candles = cached(symbol, fetch_yahoo_daily if asset_class != "crypto" else fetch_daily_history)
        except Exception as e:
            skipped.append((symbol, f"fetch failed: {e}"))
            continue
        if len(candles) < min_history:
            skipped.append((symbol, f"only {len(candles)} candles (<{min_history})"))
            continue
        if is_stale(candles):
            skipped.append((symbol, "history stopped >45d ago (delisted/stale)"))
            continue
        print(f"  {symbol:10} [{asset_class:16}] {len(candles)} candles since {_d(candles[0]['open_time'])}")
        histories[symbol] = (candles, ppy)
    for symbol, why in skipped:
        print(f"  {symbol:10} skipped: {why}")
    if not histories:
        print("No tradeable assets with enough history")
        return 2
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
    base_rule = dict(use_tilt=True, use_crisis=True)
    full_returns = combine_portfolio_rule(asset_dailies, timeline, n_selected, **base_rule)
    throttle_returns = combine_portfolio_rule(asset_dailies, timeline, n_selected, use_dd_throttle=True, **base_rule)
    banded_returns = combine_portfolio_rule(banded_dailies, timeline, n_selected, **base_rule)
    fixed_returns = combine_portfolio_rule(fixed_dailies, timeline, n_selected, use_dd_throttle=True, **base_rule)

    port = _equity_metrics(port_returns, risk_free_annual=args.risk_free)
    port_rm = _equity_metrics(_vol_overlay(port_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    iv_rm = _equity_metrics(_vol_overlay(iv_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    full_rm = _equity_metrics(_vol_overlay(full_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    throttle_rm = _equity_metrics(_vol_overlay(throttle_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    banded_rm = _equity_metrics(_vol_overlay(banded_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)
    fixed_rm = _equity_metrics(_vol_overlay(fixed_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)

    print("\nFetching S&P 500 daily history (FRED)...")
    sp = fetch_sp500()
    sp_window = slice_window(sp, _d(oos_start_ms), _d(oos_end_ms - DAY_MS))
    spx = equity_metrics(sp_window, risk_free_annual=args.risk_free)

    bh_returns = []
    for i in range(1, len(btc)):
        if oos_start_ms <= btc[i]["open_time"] < oos_end_ms:
            bh_returns.append(btc[i]["close"] / btc[i - 1]["close"] - 1.0)
    bh = _equity_metrics(bh_returns, risk_free_annual=args.risk_free)

    print()
    print(f"Out-of-sample window: {_d(oos_start_ms)} -> {_d(oos_end_ms - DAY_MS)} ({len(folds_abs)} yearly folds, {n_selected} assets)")
    print("Most-picked strategies across all folds and assets:")
    for name, count in sorted(picks_counter.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {name} x{count}")
    print()
    header = f"{'':18}{'Bot inv-vol':>14}{'Bot equal':>13}{'Bot raw eq':>12}{'S&P 500':>12}{'BTC b&h':>11}"
    print(header)
    print("-" * len(header))
    print(f"{'CAGR':18}{_fmt_pct(iv_rm['cagr']):>14}{_fmt_pct(port_rm['cagr']):>13}{_fmt_pct(port['cagr']):>12}{_fmt_pct(spx['cagr']):>12}{_fmt_pct(bh['cagr']):>11}")
    print(f"{'Volatility':18}{_fmt_pct(iv_rm['vol']):>14}{_fmt_pct(port_rm['vol']):>13}{_fmt_pct(port['vol']):>12}{_fmt_pct(spx['vol']):>12}{_fmt_pct(bh['vol']):>11}")
    print(f"{'Sharpe (excess)':18}{iv_rm['sharpe']:>14.2f}{port_rm['sharpe']:>13.2f}{port['sharpe']:>12.2f}{spx['sharpe']:>12.2f}{bh['sharpe']:>11.2f}")
    print(f"{'Max drawdown':18}{_fmt_pct(iv_rm['max_drawdown']):>14}{_fmt_pct(port_rm['max_drawdown']):>13}{_fmt_pct(port['max_drawdown']):>12}{_fmt_pct(spx['max_drawdown']):>12}{_fmt_pct(bh['max_drawdown']):>11}")
    print(f"{'Sortino':18}{iv_rm['sortino']:>14.2f}{port_rm['sortino']:>13.2f}{port['sortino']:>12.2f}{spx['sortino']:>12.2f}{bh['sortino']:>11.2f}")
    print(f"{'Calmar':18}{iv_rm['calmar']:>14.2f}{port_rm['calmar']:>13.2f}{port['calmar']:>12.2f}{spx['calmar']:>12.2f}{bh['calmar']:>11.2f}")
    print(f"{'ES 95% (1d)':18}{_fmt_pct(iv_rm['es95']):>14}{_fmt_pct(port_rm['es95']):>13}{_fmt_pct(port['es95']):>12}{_fmt_pct(spx['es95']):>12}{_fmt_pct(bh['es95']):>11}")
    print(f"{'Growth of $1':18}{iv_rm['final']:>14.2f}{port_rm['final']:>13.2f}{port['final']:>12.2f}{spx['final']:>12.2f}{bh['final']:>11.2f}")

    print("\nFixed portfolio rules (a-priori overlays, all risk-managed):")
    rules = [
        ("inv-vol (selected underlying)", iv_rm, _vol_overlay(iv_returns, target=args.portfolio_vol)),
        ("+ tilt + crisis de-risk", full_rm, _vol_overlay(full_returns, target=args.portfolio_vol)),
        ("+ drawdown throttle", throttle_rm, _vol_overlay(throttle_returns, target=args.portfolio_vol)),
        ("+ tilt + crisis, banded 5% rebalance", banded_rm, _vol_overlay(banded_returns, target=args.portfolio_vol)),
        ("fully-fixed: RiskEnsemble everywhere, banded, all overlays", fixed_rm, _vol_overlay(fixed_returns, target=args.portfolio_vol)),
    ]
    header = f"  {'rule':58}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'ES95':>7}{'Calmar':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m, _ in rules:
        print(f"  {name:58}{_fmt_pct(m['cagr']):>8}{m['sharpe']:>8.2f}{_fmt_pct(m['max_drawdown']):>8}{_fmt_pct(m['es95']):>7}{m['calmar']:>8.2f}")

    from .metrics import sharpe as _sharpe_ann
    from .stats_validation import dsr, psr

    print("\nStatistical standing (trial count 1 for the fixed rows; selected underlying carries the 85-trial caveat):")
    for name, _, rets in rules:
        p1 = psr(rets)
        d1 = dsr(rets, [_sharpe_ann(rets, 365)], 1)
        print(f"  {name:58} PSR {p1:.3f}  DSR {d1:.3f}")
    print(f"\nFrictions: execution={args.execution}, fee={args.fee:.2%}, spread={args.spread_bps:.0f}bp, "
          f"slippage={args.slippage_bps:.0f}bp, latency={args.latency_days}d, cash yield={args.risk_free:.0%}/yr")
    print("Benchmark consistency: same window/calendar-day CAGR; Sharpe in excess of the same risk-free rate; index is untradeable so carries no costs.")

    print("\nPer-asset out-of-sample results:")
    for a in sorted(per_asset, key=lambda x: -x["sharpe"]):
        print(f"  {a['symbol']:10} CAGR {_fmt_pct(a['cagr']):>7}  Sharpe {a['sharpe']:>5.2f}  maxDD {_fmt_pct(a['max_drawdown']):>7}")

    _print_regimes(btc, timeline, port_returns, sp_window, args)

    print(f"\nTiming: fetch/cache {t_fetch - t_start:.1f}s, walk-forward compute {t_compute - t_fetch:.1f}s, total {_time.perf_counter() - t_start:.1f}s")
    print("Survivorship note: universe is today's top-volume list (point-in-time constituents are not freely available); missing days are held in cash, not redistributed.")

    beats_cagr = port_rm["cagr"] > spx["cagr"]
    beats_sharpe = port_rm["sharpe"] > spx["sharpe"]
    beats_mdd = port_rm["max_drawdown"] > spx["max_drawdown"]
    print()
    print(
        f"VERDICT: risk-managed portfolio OOS CAGR {'BEATS' if beats_cagr else 'trails'} S&P 500 "
        f"({_fmt_pct(port_rm['cagr'])} vs {_fmt_pct(spx['cagr'])}); "
        f"Sharpe {'beats' if beats_sharpe else 'trails'} ({port_rm['sharpe']:.2f} vs {spx['sharpe']:.2f}); "
        f"max drawdown {'better' if beats_mdd else 'worse'} ({_fmt_pct(port_rm['max_drawdown'])} vs {_fmt_pct(spx['max_drawdown'])})"
    )
    return 0 if beats_cagr and beats_sharpe else 1


def _print_regimes(btc, timeline, port_returns, sp_window, args) -> None:
    from .regimes import label_regimes, segment, segment_metrics, stress_mask

    labels = label_regimes(btc, timeline)
    segments = segment(labels, timeline)
    port_by_day = {t: r for t, r in zip(timeline, port_returns, strict=True)}

    sp_times = [
        int(datetime(r["date"].year, r["date"].month, r["date"].day, tzinfo=UTC).timestamp() * 1000)
        for r in sp_window
    ]
    sp_closes = [r["close"] for r in sp_window]
    spx_by_day = {sp_times[k]: sp_closes[k] / sp_closes[k - 1] - 1.0 for k in range(1, len(sp_window))}
    spx_days = [t for t in timeline if t in spx_by_day]

    print("\nRegime analysis (BTC-proxy trailing 180d return labels):")
    print(f"  {'regime':10}{'period':>15}{'days':>7}{'bot CAGR':>11}{'S&P CAGR':>11}")
    for seg in segments:
        bot_m = segment_metrics(port_by_day, timeline, seg["start"], seg["end"])
        sp_m = segment_metrics(spx_by_day, spx_days, seg["start"], seg["end"])
        span = f"{_d(seg['start'])}..{str(_d(seg['end']))[5:]}"
        print(f"  {seg['label']:10}{span:>15}{bot_m['days']:>7}{_fmt_pct(bot_m['cagr']):>11}{_fmt_pct(sp_m['cagr']):>11}")

    mask = stress_mask(btc, timeline)
    stress_days = [t for t in timeline if mask.get(t)]
    if stress_days:
        sm = segment_metrics(port_by_day, timeline, stress_days[0], stress_days[-1])
        ssm = segment_metrics(spx_by_day, spx_days, stress_days[0], stress_days[-1])
        print(f"\nStress windows (30d crash <=-20% or vol in top decile): {len(stress_days)} days")
        print(f"  bot over stressed span: CAGR {_fmt_pct(sm['cagr'])}, maxDD {_fmt_pct(sm['max_drawdown'])}")
        print(f"  S&P over stressed span: CAGR {_fmt_pct(ssm['cagr'])}, maxDD {_fmt_pct(ssm['max_drawdown'])}")


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
            row += f"{m['sharpe']:>8.2f}/{_fmt_pct(m['cagr']):>7}"
        print(row)
    sharpes = [m["sharpe"] for m in res["parameters"].values()]
    print(f"  grid Sharpe min/median/max: {min(sharpes):.2f} / {sorted(sharpes)[len(sharpes) // 2]:.2f} / {max(sharpes):.2f}")

    print("\n2) Rolling parameter sensitivity (consecutive 2y blocks, walk-forward winner):")
    for b in res["rolling"]:
        m = b["metrics"]
        print(f"  {_d(b['start'])}..{_d(b['end'])}: CAGR {_fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}  maxDD {_fmt_pct(m['max_drawdown']):>7}")

    print("\n3) Transaction-cost sensitivity (combined fee+spread+slippage):")
    for bps, m in sorted(res["costs"].items()):
        print(f"  {bps:>4}bp: CAGR {_fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}  maxDD {_fmt_pct(m['max_drawdown']):>7}")

    print("\n4) Latency & execution realism:")
    for lat, m in sorted(res["latency"].items()):
        print(f"  latency {lat}d (next_open): CAGR {_fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}")
    for mode, m in res["execution"].items():
        print(f"  execution={mode:10}: CAGR {_fmt_pct(m['cagr']):>7}  Sharpe {m['sharpe']:>5.2f}")
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
    print(f"  OOS: CAGR {_fmt_pct(wf['cagr'])}, Sharpe {wf['sharpe']:.2f}, maxDD {_fmt_pct(wf['max_drawdown'])}, "
          f"exposure {wf['exposure']:.2f}, turnover {wf['turnover']:.1f}x/yr-ish")

    print("\n[2/7] Nested walk-forward (inner purged CV picks the strategy)...")
    nested = walk_forward_at(
        candles, folds, candidates=candidates, embargo_days=args.embargo_days,
        selection_fn=nested_selection_fn(purge_days=args.purge_days, embargo_days=args.embargo_days),
        **engine_kwargs,
    )
    print(f"  OOS: CAGR {_fmt_pct(nested['cagr'])}, Sharpe {nested['sharpe']:.2f}, maxDD {_fmt_pct(nested['max_drawdown'])}")
    print(f"  picks: {[p['strategy'] for p in nested['folds']]}")
    print("  (nested vs standard difference is the selection-overfitting estimate)")

    print("\n[3/7] Sharpe ratios: probabilistic & deflated...")
    import statistics as _stats

    p = psr(oos)  # PSR vs zero benchmark
    trials = len(candidates)
    d = dsr(oos, wf["trial_sharpes"], trials)
    print(f"  PSR (vs Sharpe 0): {p:.3f}")
    print(f"  DSR (deflated for {trials} trials, trial-Sharpe sd {_stats.stdev(wf['trial_sharpes']):.2f}): {d:.3f}")
    print("  PSR/DSR > 0.95 is the usual 'real edge' bar; DSR is the honest number")

    print("\n[4/7] Stationary block bootstrap (20d blocks)...")
    boot = bootstrap_metrics(oos, n_boot=args.boots, seed=args.seed)
    print(f"  CAGR 90% CI: [{_fmt_pct(boot['cagr_ci'][0])}, {_fmt_pct(boot['cagr_ci'][1])}]")
    print(f"  Sharpe 90% CI: [{boot['sharpe_ci'][0]:.2f}, {boot['sharpe_ci'][1]:.2f}]")
    print(f"  Max-drawdown distribution: median {_fmt_pct(boot['mdd_median'])}, p95 {_fmt_pct(boot['mdd_p95'])}, worst {_fmt_pct(boot['mdd_worst'])}")

    print(f"\n[5/7] White's Reality Check across all {trials} candidates ({args.rc_boots} bootstrap max-tests)...")
    streams = fixed_candidate_streams(candles, folds, candidates, **engine_kwargs)
    common_days = sorted(set.intersection(*(set(s) for s in streams.values()))) if streams else []
    matrix = [[s[t] for t in common_days] for s in streams.values()]
    rc = reality_check(matrix, n_boot=args.rc_boots, seed=args.seed)
    print(f"  best fixed-candidate OOS Sharpe: {rc['best_sharpe']:.2f}")
    print(f"  RC p-value: {rc['p_value']:.3f}  (< 0.05 => best strategy unlikely to be pure selection luck)")

    print("\n[6/7] Path & window robustness...")
    sh = shuffle_test(oos, n_boot=args.boots, seed=args.seed)
    print(f"  trade-order shuffle: actual maxDD {_fmt_pct(sh['actual_mdd'])} vs shuffled median {_fmt_pct(sh['shuffled_mdd_median'])}; "
          f"protection percentile {sh['dd_percentile']:.2f} (high = return sequencing itself avoids drawdowns)")
    print("  start/end-date sensitivity (CAGR / Sharpe by trimming the window):")
    for row in start_end_sensitivity(oos):
        print(f"    trim start {row['trim_start']:>3}d end {row['trim_end']:>3}d: {_fmt_pct(row['cagr']):>7} / {row['sharpe']:.2f}")

    print("\n[7/7] Distribution, tail risk & parameter stability...")
    print(f"  daily returns: skew {skewness(oos):.2f}, kurtosis {kurtosis(oos):.1f} (normal=3)")
    print(f"  VaR 95% (descriptive only): {_fmt_pct(var_hist(oos, 0.95))}, VaR 99%: {_fmt_pct(var_hist(oos, 0.99))}")
    print(f"  Expected shortfall 95%: {_fmt_pct(expected_shortfall(oos, 0.95))}, 97.5%: {_fmt_pct(expected_shortfall(oos, 0.975))}")
    print(f"  worst day: {_fmt_pct(min(oos))}")
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
    print(f"  RiskEnsemble OOS: CAGR {_fmt_pct(fixed['cagr'])}, Sharpe {fixed['sharpe']:.2f}, maxDD {_fmt_pct(fixed['max_drawdown'])}")
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
    print(f"  OOS: CAGR {_fmt_pct(wf['cagr'])}, Sharpe {wf['sharpe']:.2f}, maxDD {_fmt_pct(wf['max_drawdown'])}")

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
    print(f"  maxDD 90% CI: [{_fmt_pct(dd['mdd_90_ci'][0])}, {_fmt_pct(dd['mdd_90_ci'][1])}], worst {_fmt_pct(dd['mdd_worst'])}; "
          f"time-under-water median {dd['time_under_water_median']:.0%}, p95 {dd['time_under_water_p95']:.0%}")
    seq = sequence_risk(oos, horizon_days=min(252, len(oos) // 2), n_shuffles=100, seed=args.seed)
    print(f"  sequence risk over {seq['horizon_days']}d windows: P(loss) observed {seq['observed_p_loss']:.2f} vs shuffled "
          f"{seq['shuffle_mean_p_loss']:.2f} (gap {seq['sequence_risk_gap']:+.2f}); "
          f"worst entry {_fmt_pct(seq['observed_worst'])}, best entry {_fmt_pct(seq['observed_best'])}")
    mc = mc_future_paths(oos, horizon_days=min(252, len(oos)), n_paths=2000, seed=args.seed)
    print(f"  forward MC (block bootstrap, 1y): P(loss) {mc['p_loss']:.2f}, terminal p5/p50/p95 "
          f"{mc['terminal_p05']:.2f}/{mc['terminal_median']:.2f}/{mc['terminal_p95']:.2f}, path-maxDD p95 {_fmt_pct(mc['path_mdd_p95'])}")
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
                "strategy": chosen,
                "oos_sharpe_at_freeze": wf["sharpe"],
            }
        )
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        commit = None

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

    manifest = create_freeze(
        assets,
        frictions={
            "fee": args.fee,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "execution": args.execution,
            "risk_free_annual": args.risk_free,
        },
        overlay={"target_vol": args.portfolio_vol},
        path=args.freeze_file,
        git_commit=commit,
        git_tag=tag if commit and not args.no_tag else None,
        image_digest=args.image_digest,
    )

    print(f"\nFroze {len(assets)} assets at {manifest['frozen_at']}")
    print(f"config sha256: {manifest['config_sha256']}")
    print(f"code sha256  : {manifest['code_sha256']} ({manifest['code_fingerprint_algo']})")
    print(f"commit       : {manifest['git_commit_at_freeze']}  tag: {manifest.get('git_tag')}")
    if manifest.get("image_digest"):
        print(f"image digest : {manifest['image_digest']}")
    print(f"PUSH THE TAG: git push origin {tag}")
    print(f"manifest: {args.freeze_file} — COMMIT IT NOW; CI will check out this exact commit, "
          "verify the code fingerprint, then trade the forward day on frozen code only")
    return 0


def manifest_date() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d")


def run_verify_freeze(args) -> int:
    import json

    from .identity import verify_freeze_code

    try:
        raw = json.loads(open(args.freeze_file, encoding="utf-8").read())
    except FileNotFoundError:
        print(f"FAIL: {args.freeze_file} not found — nothing frozen to verify against")
        return 1
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
    from .data import fetch_daily_history, fetch_yahoo_daily

    try:
        candles = fetch_yahoo_daily(symbol) if source == "yahoo" else fetch_daily_history(symbol, max_candles=400)
        if not candles:
            return [], "empty history"
        return candles, None
    except Exception as e:
        return [], f"fetch failed: {e}"


def run_forward(args) -> int:
    from datetime import date as _date

    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .prospective import alert_stats, checkpoints_due, load_freeze, load_log, monthly_returns, outage_stats, run_step, slippage_stats

    manifest = load_freeze(args.freeze_file)
    freeze_date = _date.fromisoformat(manifest["frozen_at_date"])

    if args.step:
        result = run_step(manifest, _forward_fetch, log_path=args.log_file)
        e = result["entry"]
        print(f"[{e['date']}] {result['status']}: port_ret {e['port_ret']:+.4%}, overlay {e['overlay_weight']:.2f}, "
              f"assets {len(e['assets'])}, outages {len(e['outages'])}, missed_fills {len(e['missed_fills'])}")

    entries = load_log(args.log_file)
    if not entries:
        print("No forward entries yet — run `python -m bot forward --step` daily")
        return 0

    eq = 1.0
    for e in entries:
        eq *= 1.0 + e["port_ret"]
    rets = [e["port_ret"] for e in entries]
    as_of = _date.fromisoformat(entries[-1]["date"])
    from .metrics import max_drawdown as _mdd
    from .metrics import sharpe as _sharpe

    print(f"\nProspective validation: frozen {freeze_date} -> last step {as_of} ({len(entries)} forward days)")
    print(f"  bot forward: {eq:.3f}x ({(eq - 1):+.1%}), Sharpe {_sharpe(rets, 365):.2f}, maxDD {_mdd([1.0] + [x for x in _cum(rets)]):.1%}")

    cps = checkpoints_due(freeze_date, as_of)
    sp = fetch_sp500()
    sp_window = slice_window(sp, freeze_date, as_of)
    spx = equity_metrics(sp_window) if len(sp_window) > 3 else None
    print("\nCheckpoints (bot vs S&P 500 over the same span):")
    for cp in cps:
        if not cp["due"]:
            print(f"  {cp['label']:>10}: pending ({cp['elapsed']}/{cp['days']} days)")
        elif spx:
            print(f"  {cp['label']:>10}: bot {(eq - 1):+.1%} vs S&P {(spx['final'] - 1):+.1%}  [elapsed {cp['elapsed']}d]")

    slip = slippage_stats(entries)
    out = outage_stats(entries)
    al = alert_stats(entries)
    print(f"\nIncidents: mean |decision->execution| {slip['mean_abs_bps'] or 0:.0f}bp over {slip['count']} turnover events; "
          f"{out['outage_days']} outage days ({out['outage_events']} events); {out['missed_fills']} missed fills")
    if al["total"]:
        print(f"Data-staleness alerts: {al['total']} ({al['by_level']}) on {', '.join(al['symbols'])}")

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


def main():
    # Windows consoles default to legacy codepages that cannot encode model
    # output (arrows, approx signs); UTF-8 with replacement keeps prints alive
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    parser = argparse.ArgumentParser(description="Paper trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Backtest the strategy on historical candles")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--fast", type=int, default=20)
    bt.add_argument("--slow", type=int, default=50)

    tr = sub.add_parser("trade", help="Run the live paper-trading loop (paper only)")
    tr.add_argument("--symbol", default="BTCUSDT", help="single symbol or comma-separated list")
    tr.add_argument("--interval", default=None, help="candle interval (default: 1d for trendvol, 1h otherwise)")
    tr.add_argument("--poll", type=int, default=None)
    tr.add_argument("--fast", type=int, default=20)
    tr.add_argument("--slow", type=int, default=50)
    tr.add_argument("--strategy", choices=["sma", "trendvol"], default="sma")
    tr.add_argument("--once", action="store_true", help="run one cycle, write the audit report, exit")
    tr.add_argument("--reports-dir", default="reports")
    tr.add_argument("--ai-note", action="store_true", help="append advisory AI commentary to the audit report")

    ask = sub.add_parser("ask", help="Ask a question grounded in the bot's computed state (OpenRouter, allowlisted free models)")
    ask.add_argument("question")
    ask.add_argument("--symbol", default="BTCUSDT")

    cmp = sub.add_parser("compare", help="Walk-forward out-of-sample portfolio comparison vs the S&P 500")
    cmp.add_argument("--assets", type=int, default=20, help="number of top-volume crypto assets (+3 ETFs)")
    cmp.add_argument("--train-days", type=int, default=1095)
    cmp.add_argument("--test-days", type=int, default=365)
    cmp.add_argument("--fee", type=float, default=0.001)
    cmp.add_argument("--spread-bps", type=float, default=5.0)
    cmp.add_argument("--slippage-bps", type=float, default=5.0)
    cmp.add_argument("--latency-days", type=int, default=0)
    cmp.add_argument("--execution", choices=["close", "next_open"], default="next_open")
    cmp.add_argument("--risk-free", type=float, default=0.03)
    cmp.add_argument("--portfolio-vol", type=float, default=0.25, help="risk-managed overlay target vol")

    sen = sub.add_parser("sensitivity", help="Backtesting-quality sensitivity sweeps")
    sen.add_argument("--symbol", default="BTCUSDT")
    sen.add_argument("--train-days", type=int, default=1095)
    sen.add_argument("--test-days", type=int, default=365)
    sen.add_argument("--fee", type=float, default=0.001)
    sen.add_argument("--spread-bps", type=float, default=5.0)
    sen.add_argument("--slippage-bps", type=float, default=5.0)
    sen.add_argument("--execution", choices=["close", "next_open"], default="next_open")
    sen.add_argument("--risk-free", type=float, default=0.03)

    val = sub.add_parser("validate", help="Statistical validation battery (PSR/DSR, bootstrap, Reality Check, ...)")
    val.add_argument("--symbol", default="BTCUSDT")
    val.add_argument("--train-days", type=int, default=1095)
    val.add_argument("--test-days", type=int, default=365)
    val.add_argument("--fee", type=float, default=0.001)
    val.add_argument("--spread-bps", type=float, default=5.0)
    val.add_argument("--slippage-bps", type=float, default=5.0)
    val.add_argument("--execution", choices=["close", "next_open"], default="next_open")
    val.add_argument("--risk-free", type=float, default=0.03)
    val.add_argument("--embargo-days", type=int, default=30)
    val.add_argument("--purge-days", type=int, default=220)
    val.add_argument("--boots", type=int, default=1000)
    val.add_argument("--rc-boots", type=int, default=100)
    val.add_argument("--seed", type=int, default=42)

    res = sub.add_parser("research", help="Methodology battery: SPA, clustering/effective trials, ablation, Bayesian, sequence risk")
    res.add_argument("--symbol", default="BTCUSDT")
    res.add_argument("--train-days", type=int, default=1095)
    res.add_argument("--test-days", type=int, default=365)
    res.add_argument("--fee", type=float, default=0.001)
    res.add_argument("--spread-bps", type=float, default=5.0)
    res.add_argument("--slippage-bps", type=float, default=5.0)
    res.add_argument("--execution", choices=["close", "next_open"], default="next_open")
    res.add_argument("--risk-free", type=float, default=0.03)
    res.add_argument("--embargo-days", type=int, default=30)
    res.add_argument("--boots", type=int, default=600)
    res.add_argument("--rc-boots", type=int, default=150)
    res.add_argument("--seed", type=int, default=42)

    frz = sub.add_parser("freeze", help="Freeze strategies/params/universe into a tamper-evident manifest")
    frz.add_argument("--assets", type=int, default=20)
    frz.add_argument("--train-days", type=int, default=1095)
    frz.add_argument("--test-days", type=int, default=365)
    frz.add_argument("--fee", type=float, default=0.001)
    frz.add_argument("--spread-bps", type=float, default=5.0)
    frz.add_argument("--slippage-bps", type=float, default=5.0)
    frz.add_argument("--execution", choices=["close", "next_open"], default="next_open")
    frz.add_argument("--risk-free", type=float, default=0.03)
    frz.add_argument("--portfolio-vol", type=float, default=0.25)
    frz.add_argument("--embargo-days", type=int, default=30)
    frz.add_argument("--freeze-file", default="freeze.json")
    frz.add_argument("--no-tag", action="store_true", help="skip creating the freeze/<date> git tag")
    frz.add_argument("--tag", default=None, help="override the tag name (default: freeze/<YYYYMMDD>)")
    frz.add_argument("--image-digest", default=None, help='record a container digest, e.g. "sha256:..."')

    ver = sub.add_parser("verify-freeze", help="Refuse unless running code matches the frozen implementation")
    ver.add_argument("--freeze-file", default="freeze.json")

    fwd = sub.add_parser("forward", help="Prospective paper trading: --step one day, --report checkpoints")
    fwd.add_argument("--step", action="store_true", help="execute one forward day from the freeze")
    fwd.add_argument("--report", action="store_true", help="print the checkpoint report")
    fwd.add_argument("--freeze-file", default="freeze.json")
    fwd.add_argument("--log-file", default="forward_log.jsonl")

    args = parser.parse_args()

    if args.command == "backtest":
        candles = fetch_candles(args.symbol, args.interval, limit=500)
        results = backtest(candles, SmaCrossover(args.fast, args.slow))
        print(f"Backtest {args.symbol} {args.interval} (SMA {args.fast}/{args.slow}):")
        for k, v in results.items():
            print(f"  {k}: {v}")
    elif args.command == "trade":
        from .paper import run

        # TrendVol's volatility targeting assumes daily bars; default accordingly.
        interval = args.interval or ("1d" if args.strategy == "trendvol" else "1h")
        poll = args.poll or (3600 if args.strategy == "trendvol" else 60)
        ai_note_fn = None
        if args.ai_note:
            from .ai import complete as _ai_complete

            def ai_note_fn(report: str) -> str | None:
                return _ai_complete(
                    "Below is today's paper-trading audit report. Write a 4-6 sentence "
                    "commentary section for it: what changed, one risk to watch, one honest "
                    "caveat. Use ONLY numbers present in the report.\n\n" + report,
                    system="You write the 'AI commentary' appendix of an audit report for a "
                    "deterministic paper-trading bot. Advisory only; concise; no invented data.",
                )

        run(args.symbol, interval, poll, args.fast, args.slow, args.strategy, once=args.once, reports_dir=args.reports_dir, ai_note_fn=ai_note_fn)
    elif args.command == "compare":
        raise SystemExit(run_compare(args))
    elif args.command == "sensitivity":
        raise SystemExit(run_sensitivity(args))
    elif args.command == "validate":
        raise SystemExit(run_validate(args))
    elif args.command == "research":
        raise SystemExit(run_research(args))
    elif args.command == "ask":
        raise SystemExit(run_ask(args))
    elif args.command == "freeze":
        raise SystemExit(run_freeze(args))
    elif args.command == "verify-freeze":
        raise SystemExit(run_verify_freeze(args))
    elif args.command == "forward":
        if not (args.step or args.report):
            print("use --step and/or --report")
        raise SystemExit(run_forward(args))


if __name__ == "__main__":
    main()
