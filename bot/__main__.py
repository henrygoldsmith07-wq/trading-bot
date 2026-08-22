"""CLI entry point.

Usage:
  python -m bot backtest --symbol BTCUSDT --interval 1h
  python -m bot trade    --symbol BTCUSDT --strategy trendvol
  python -m bot compare  [--assets 20]                # portfolio vs S&P 500
  python -m bot sensitivity [--symbol BTCUSDT]        # backtesting-quality sweeps
"""
from __future__ import annotations

import argparse
import time as _time
from datetime import datetime, timezone

from .backtest import backtest
from .data import fetch_candles
from .strategy import SmaCrossover

DAY_MS = 86_400_000


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _equity_metrics(returns: list[float], periods_per_year: int = 365, risk_free_annual: float = 0.0) -> dict:
    from .metrics import cagr, max_drawdown, sharpe, volatility

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = len(returns)  # daily returns: one calendar day each
    return {
        "final": equity[-1],
        "cagr": cagr(equity, days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "max_drawdown": max_drawdown(equity),
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
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


def run_compare(args) -> int:
    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .cache import load_or_fetch
    from .data import DAY_MS as _D, fetch_daily_history, fetch_yahoo_daily, is_stale
    from .universe import ETF_UNIVERSE, top_symbols
    from .walkforward import absolute_folds, combine_portfolio, walk_forward_at

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
    specs = [(s, "crypto", 365) for s in universe] + [(e["symbol"], e["asset_class"], e["periods_per_year"]) for e in ETF_UNIVERSE]

    print(f"Universe: {args.assets} crypto pairs by quote volume + {len(ETF_UNIVERSE)} cross-class ETFs")
    print("  (paper-only validation; no live orders are ever placed)")
    histories = {}
    skipped = []
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
    per_asset = []
    picks_counter: dict[str, int] = {}
    for symbol, (candles, ppy) in histories.items():
        wf = walk_forward_at(candles, folds_abs, periods_per_year=ppy, **engine_kwargs)
        per_asset.append({"symbol": symbol, "cagr": wf["cagr"], "sharpe": wf["sharpe"], "max_drawdown": wf["max_drawdown"]})
        for p in wf["folds"]:
            picks_counter[p["strategy"]] = picks_counter.get(p["strategy"], 0) + 1
        asset_dailies[symbol] = {t: r for t, r in wf["daily"].items() if oos_start_ms <= t < oos_end_ms}
    port_returns = combine_portfolio(asset_dailies, timeline, n_selected)
    t_compute = _time.perf_counter()

    port = _equity_metrics(port_returns, risk_free_annual=args.risk_free)
    port_rm = _equity_metrics(_vol_overlay(port_returns, target=args.portfolio_vol), risk_free_annual=args.risk_free)

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
    header = f"{'':18}{'Bot (risk-mgd)':>16}{'Bot (raw)':>13}{'S&P 500':>12}{'BTC b&h':>11}"
    print(header)
    print("-" * len(header))
    print(f"{'CAGR':18}{_fmt_pct(port_rm['cagr']):>16}{_fmt_pct(port['cagr']):>13}{_fmt_pct(spx['cagr']):>12}{_fmt_pct(bh['cagr']):>11}")
    print(f"{'Volatility':18}{_fmt_pct(port_rm['vol']):>16}{_fmt_pct(port['vol']):>13}{_fmt_pct(spx['vol']):>12}{_fmt_pct(bh['vol']):>11}")
    print(f"{'Sharpe (excess)':18}{port_rm['sharpe']:>16.2f}{port['sharpe']:>13.2f}{spx['sharpe']:>12.2f}{bh['sharpe']:>11.2f}")
    print(f"{'Max drawdown':18}{_fmt_pct(port_rm['max_drawdown']):>16}{_fmt_pct(port['max_drawdown']):>13}{_fmt_pct(spx['max_drawdown']):>12}{_fmt_pct(bh['max_drawdown']):>11}")
    print(f"{'Growth of $1':18}{port_rm['final']:>16.2f}{port['final']:>13.2f}{spx['final']:>12.2f}{bh['final']:>11.2f}")
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
    port_by_day = {t: r for t, r in zip(timeline, port_returns)}

    sp_times = [
        int(datetime(r["date"].year, r["date"].month, r["date"].day, tzinfo=timezone.utc).timestamp() * 1000)
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


def main():
    parser = argparse.ArgumentParser(description="Paper trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Backtest the strategy on historical candles")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--fast", type=int, default=20)
    bt.add_argument("--slow", type=int, default=50)

    tr = sub.add_parser("trade", help="Run the live paper-trading loop (paper only)")
    tr.add_argument("--symbol", default="BTCUSDT")
    tr.add_argument("--interval", default=None, help="candle interval (default: 1d for trendvol, 1h otherwise)")
    tr.add_argument("--poll", type=int, default=None)
    tr.add_argument("--fast", type=int, default=20)
    tr.add_argument("--slow", type=int, default=50)
    tr.add_argument("--strategy", choices=["sma", "trendvol"], default="sma")

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
        run(args.symbol, interval, poll, args.fast, args.slow, args.strategy)
    elif args.command == "compare":
        raise SystemExit(run_compare(args))
    elif args.command == "sensitivity":
        raise SystemExit(run_sensitivity(args))


if __name__ == "__main__":
    main()
