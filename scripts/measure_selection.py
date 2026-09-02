"""Measure walk-forward SELECTION RULES against each other on real data.

This is a research instrument, not a tuning rig. The four rules below are
fixed at their textbook settings; nothing here searches for a good
configuration. Its job is to answer one question honestly:

    does replacing "argmax in-sample Sharpe" with a robustness-aware rule
    produce better out-of-sample results?

Usage:
    python scripts/measure_selection.py
    python scripts/measure_selection.py --assets BTCUSDT,ETHUSDT --train-days 1095

Rules compared:
  argmax              the historical default: highest in-sample Sharpe wins
  one_se_worst        Breiman 1-SE eligible set, minimax sub-window tie-break
  one_se_turnover     Breiman 1-SE eligible set, lowest-turnover tie-break
  robust_shrunk       1-SE + minimax + credibility shrinkage to the prior

Every rule sees the SAME folds, the SAME candidate pool and the SAME
frictions, so any difference is attributable to the selection rule alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.data import clean_candles, fetch_daily_history, fetch_yahoo_daily, fill_small_gaps  # noqa: E402
from bot.metrics import cagr, max_drawdown, sharpe  # noqa: E402
from bot.selection import make_one_se_selection_fn, make_robust_selection_fn  # noqa: E402
from bot.stats_validation import dsr, psr  # noqa: E402
from bot.strategy import build_candidates  # noqa: E402
from bot.walkforward import walk_forward  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "measure_selection"
CRYPTO = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"}
ETFS = {"SPY", "GLD", "TLT"}

DEFAULT_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SPY", "GLD", "TLT"]


def load_candles(symbol: str) -> list[dict]:
    """Fetch (and disk-cache) cleaned daily candles for one symbol."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raw = fetch_yahoo_daily(symbol) if symbol in ETFS else fetch_daily_history(symbol)
    cleaned = fill_small_gaps(clean_candles(raw))
    path.write_text(json.dumps(cleaned), encoding="utf-8")
    return cleaned


RULE_NAMES = ["argmax", "one_se_worst", "one_se_turnover", "robust_shrunk"]


def fresh_rule(name: str):
    """A NEW selector instance per asset.

    Selectors publish diagnostics on themselves (`last_diagnostics`), so
    reusing one across assets would let a fold's diagnostics leak into the
    next asset's record. None means the built-in argmax default.
    """
    if name == "argmax":
        return None
    if name == "robust_shrunk":
        return make_robust_selection_fn(shrink=True)
    return make_one_se_selection_fn(
        tie_break="worst_subwindow" if name == "one_se_worst" else "lowest_turnover"
    )


def run_asset(candles: list[dict], candidates: list, rule_fn, args) -> dict:
    """One walk-forward for one asset under one selection rule."""
    wf = walk_forward(
        candles,
        candidates=candidates,
        train_days=args.train_days,
        test_days=args.test_days,
        fee=args.fee,
        periods_per_year=args.ppy,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
        risk_free_annual=args.risk_free,
        selection_fn=rule_fn,
    )
    rets = [wf["daily"][t] for t in sorted(wf["daily"])]
    days = (wf["last_day"] - wf["first_day"]) / 86_400_000 + 1
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1.0 + r))
    trials = wf["trial_sharpes"]
    n_trials = len(candidates)
    return {
        "daily": wf["daily"],
        "sharpe": sharpe(rets, args.ppy, args.risk_free),
        "cagr": cagr(equity, days),
        "max_dd": max_drawdown(equity),
        "psr": psr(rets, args.ppy),
        "dsr": dsr(rets, trials, n_trials, args.ppy),
        "n_days": len(rets),
        "n_folds": wf["n_folds"],
        "turnover": wf["turnover"],
        "exposure": wf["exposure"],
        "picks": [p["strategy"] for p in wf["folds"]],
    }


def bootstrap_sharpe_diff(a: list[float], b: list[float], ppy: int, rf: float,
                          boots: int = 400, block: int = 20, seed: int = 11):
    """Paired stationary-bootstrap CI for Sharpe(a) - Sharpe(b).

    Both series are resampled on the SAME day indices, so the comparison is
    paired: a market regime that hurts one rule hurts the other in the same
    resample. `p_better` is the fraction of resamples in which `a` wins.
    """
    import random

    from bot.stats_validation import stationary_bootstrap_indices

    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(boots):
        idx = stationary_bootstrap_indices(n, block, rng)
        diffs.append(sharpe([a[i] for i in idx], ppy, rf) - sharpe([b[i] for i in idx], ppy, rf))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return lo, hi, sum(1 for d in diffs if d > 0) / len(diffs)


def portfolio_phase(results: dict, rules: list[str], args) -> None:
    """Combine each rule's per-asset OOS streams into one portfolio.

    Per-asset Sharpe is the honest unit for judging a *selection* rule, but
    the repo actually trades a diversified multi-asset portfolio, where
    cross-asset averaging can either wash the selection effect out or
    amplify it. Both views are reported.
    """
    from bot.walkforward import combine_portfolio_invvol

    common: set | None = None
    for name in rules:
        for r in results[name].values():
            keys = set(r["daily"])
            common = keys if common is None else (common & keys)
    if not common:
        print("\nno days common to every asset/rule; skipping the portfolio view")
        return
    timeline = sorted(common)
    n_assets = min(len(results[name]) for name in rules)

    print(f"\nPortfolio view — {n_assets} assets, {len(timeline)} common OOS days, "
          f"inverse-vol weighted (20d, 2x cap)")
    hdr = (f"{'rule':16s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} "
           f"{'dSharpe':>16s} {'P(better)':>10s}")
    print(hdr)
    print("-" * len(hdr))

    port: dict[str, list[float]] = {}
    for name in rules:
        dailies = {sym: r["daily"] for sym, r in results[name].items()}
        port[name] = combine_portfolio_invvol(dailies, timeline, n_assets, window=20,
                                              max_multiple_of_equal=2.0)

    base = port[rules[0]]
    for name in rules:
        rets = port[name]
        days = (timeline[-1] - timeline[0]) / 86_400_000 + 1
        equity = [1.0]
        for r in rets:
            equity.append(equity[-1] * (1.0 + r))
        if name == rules[0]:
            diff, pbetter = "", ""
        else:
            lo, hi, pbetter = bootstrap_sharpe_diff(rets, base, args.ppy, args.risk_free)
            diff = f"[{lo:+.2f},{hi:+.2f}]"
            pbetter = f"{pbetter:.0%}"
        print(f"{name:16s} {sharpe(rets, args.ppy, args.risk_free):+7.3f} "
              f"{cagr(equity, days):+7.1%} {max_drawdown(equity):+8.1%} "
              f"{diff:>16s} {pbetter:>10s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    ap.add_argument("--train-days", type=int, default=1095)
    ap.add_argument("--test-days", type=int, default=365)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--spread-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--risk-free", type=float, default=0.03)
    ap.add_argument("--execution", default="next_open")
    ap.add_argument("--ppy", type=int, default=365)
    ap.add_argument("--extended", action="store_true", help="use the extended candidate pool")
    args = ap.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    candidates = build_candidates(extended=args.extended)
    rules = RULE_NAMES

    print(f"assets={len(assets)} pool={len(candidates)} train={args.train_days}d test={args.test_days}d "
          f"fee={args.fee} spread={args.spread_bps}bp slip={args.slippage_bps}bp exec={args.execution}")
    print()

    results: dict[str, dict[str, dict]] = {name: {} for name in rules}
    for sym in assets:
        try:
            candles = load_candles(sym)
        except Exception as exc:  # network/provider hiccups must not kill the run
            print(f"  !! {sym}: could not load data ({type(exc).__name__}: {exc})")
            continue
        if len(candles) < args.train_days + args.test_days + 30:
            print(f"  -- {sym}: only {len(candles)} bars, skipped")
            continue
        for name in rules:
            try:
                results[name][sym] = run_asset(candles, candidates, fresh_rule(name), args)
            except ValueError as exc:
                print(f"  -- {sym}/{name}: {exc}")
        print(f"  {sym:10s} " + "  ".join(
            f"{n}:{results[n][sym]['sharpe']:+.2f}" for n in rules if sym in results[n]))

    print()
    hdr = f"{'rule':16s} {'meanSR':>8s} {'medSR':>7s} {'meanCAGR':>9s} {'meanDD':>8s} {'meanDSR':>8s} {'assets':>7s}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for name in rules:
        rows = list(results[name].values())
        if not rows:
            continue
        srs = [r["sharpe"] for r in rows]
        summary[name] = {
            "mean_sharpe": sum(srs) / len(srs),
            "median_sharpe": median(srs),
            "mean_cagr": sum(r["cagr"] for r in rows) / len(rows),
            "mean_max_dd": sum(r["max_dd"] for r in rows) / len(rows),
            "mean_dsr": sum(r["dsr"] for r in rows) / len(rows),
            "n_assets": len(rows),
        }
        s = summary[name]
        print(f"{name:16s} {s['mean_sharpe']:+8.3f} {s['median_sharpe']:+7.3f} "
              f"{s['mean_cagr']:+8.1%} {s['mean_max_dd']:+8.1%} {s['mean_dsr']:8.3f} {s['n_assets']:7d}")

    portfolio_phase(results, rules, args)

    out = Path(__file__).resolve().parent.parent / "selection_measurement.json"
    # the per-day streams are thousands of floats per asset; keep the summary
    slim = {
        name: {sym: {k: v for k, v in row.items() if k != "daily"} for sym, row in rows.items()}
        for name, rows in results.items()
    }
    out.write_text(json.dumps({"args": vars(args), "per_asset": slim, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
