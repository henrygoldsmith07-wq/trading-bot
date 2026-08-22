"""CLI entry point.

Usage:
  python -m bot backtest --symbol BTCUSDT --interval 1h
  python -m bot trade    --symbol BTCUSDT --strategy trendvol
  python -m bot compare  [--assets 10]                # portfolio vs S&P 500
"""
from __future__ import annotations

import argparse
import time as _time

from .backtest import backtest
from .data import fetch_candles
from .strategy import SmaCrossover


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _equity_metrics(returns: list[float], periods_per_year: int = 365) -> dict:
    from .metrics import cagr, max_drawdown, sharpe, volatility

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = len(returns)  # daily returns: one calendar day each
    return {
        "final": equity[-1],
        "cagr": cagr(equity, days),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
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


def run_compare(args) -> int:
    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .cache import load_or_fetch
    from .data import fetch_daily_history
    from .engine import DAY_MS, candle_date
    from .universe import top_symbols
    from .walkforward import absolute_folds, walk_forward, walk_forward_at

    t_start = _time.perf_counter()

    def cached_history(symbol):
        return load_or_fetch(symbol, lambda s: fetch_daily_history(s))[0]

    btc = cached_history("BTCUSDT")
    folds_abs = absolute_folds(btc, args.train_days, args.test_days)
    if not folds_abs:
        print("Not enough BTC history for one walk-forward fold")
        return 2

    symbols = ["BTCUSDT"]
    if args.assets > 1:
        extra = [s for s in top_symbols(args.assets - 1) if s != "BTCUSDT"]
        symbols += extra

    min_history = args.train_days + args.test_days + 180
    histories = {}
    print(f"Universe ({args.assets} assets by quote volume, daily history):")
    for sym in symbols:
        try:
            candles = cached_history(sym)
        except Exception as e:
            print(f"  {sym}: fetch failed ({e}), skipped")
            continue
        if len(candles) < min_history:
            print(f"  {sym}: only {len(candles)} candles (<{min_history}), skipped")
            continue
        first = candle_date(candles[0]).isoformat()
        print(f"  {sym}: {len(candles)} candles since {first}")
        histories[sym] = candles
    t_fetch = _time.perf_counter()

    oos_start_ms, oos_end_ms = folds_abs[0][0], folds_abs[-1][1]
    timeline = [c["open_time"] for c in btc if oos_start_ms <= c["open_time"] < oos_end_ms]

    portfolio_daily: dict[int, list[float]] = {}
    per_asset = []
    picks_counter: dict[str, int] = {}
    for sym, candles in histories.items():
        wf = walk_forward_at(candles, folds_abs, fee=args.fee)
        per_asset.append(
            {
                "symbol": sym,
                "cagr": wf["cagr"],
                "sharpe": wf["sharpe"],
                "max_drawdown": wf["max_drawdown"],
                "top_pick": max(set(p["strategy"] for p in wf["folds"]), key=lambda s: sum(1 for p in wf["folds"] if p["strategy"] == s)),
            }
        )
        for p in wf["folds"]:
            picks_counter[p["strategy"]] = picks_counter.get(p["strategy"], 0) + 1
        for t, r in wf["daily"].items():
            if oos_start_ms <= t < oos_end_ms:
                portfolio_daily.setdefault(t, []).append(r)
    t_compute = _time.perf_counter()

    port_returns = []
    for t in timeline:
        rets = portfolio_daily.get(t)
        port_returns.append(sum(rets) / len(rets) if rets else 0.0)
    port = _equity_metrics(port_returns)
    port_rm = _equity_metrics(_vol_overlay(port_returns, target=args.portfolio_vol))

    from datetime import datetime, timezone

    def _d(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()

    print(f"\nFetching S&P 500 daily history (FRED)...")
    sp = fetch_sp500()
    sp_window = slice_window(sp, _d(oos_start_ms), _d(oos_end_ms))
    spx = equity_metrics(sp_window)

    bh_returns = []
    for i in range(1, len(btc)):
        if oos_start_ms <= btc[i]["open_time"] < oos_end_ms:
            bh_returns.append(btc[i]["close"] / btc[i - 1]["close"] - 1.0)
    bh = _equity_metrics(bh_returns)

    print()
    print(f"Out-of-sample window: {_d(oos_start_ms)} -> {_d(oos_end_ms - DAY_MS)} ({len(folds_abs)} yearly folds, {len(histories)} assets)")
    print("Most-picked strategies across all folds and assets:")
    for name, count in sorted(picks_counter.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {name} x{count}")
    print()
    header = f"{'':18}{'Bot (risk-mgd)':>16}{'Bot (raw)':>13}{'S&P 500':>12}{'BTC b&h':>11}"
    print(header)
    print("-" * len(header))
    print(f"{'CAGR':18}{_fmt_pct(port_rm['cagr']):>16}{_fmt_pct(port['cagr']):>13}{_fmt_pct(spx['cagr']):>12}{_fmt_pct(bh['cagr']):>11}")
    print(f"{'Volatility':18}{_fmt_pct(port_rm['vol']):>16}{_fmt_pct(port['vol']):>13}{_fmt_pct(spx['vol']):>12}{_fmt_pct(bh['vol']):>11}")
    print(f"{'Sharpe':18}{port_rm['sharpe']:>16.2f}{port['sharpe']:>13.2f}{spx['sharpe']:>12.2f}{bh['sharpe']:>12.2f}")
    print(f"{'Max drawdown':18}{_fmt_pct(port_rm['max_drawdown']):>16}{_fmt_pct(port['max_drawdown']):>13}{_fmt_pct(spx['max_drawdown']):>12}{_fmt_pct(bh['max_drawdown']):>11}")
    print(f"{'Growth of $1':18}{port_rm['final']:>16.2f}{port['final']:>13.2f}{spx['final']:>12.2f}{bh['final']:>11.2f}")

    print("\nPer-asset out-of-sample results:")
    for a in sorted(per_asset, key=lambda x: -x["sharpe"]):
        print(
            f"  {a['symbol']:12} CAGR {_fmt_pct(a['cagr']):>7}  Sharpe {a['sharpe']:>5.2f}  "
            f"maxDD {_fmt_pct(a['max_drawdown']):>7}  picks: {a['top_pick']}"
        )

    print(f"\nTiming: fetch/cache {t_fetch - t_start:.1f}s, walk-forward compute {t_compute - t_fetch:.1f}s, total {_time.perf_counter() - t_start:.1f}s")

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


def main():
    parser = argparse.ArgumentParser(description="Paper trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Backtest the strategy on historical candles")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--fast", type=int, default=20)
    bt.add_argument("--slow", type=int, default=50)

    tr = sub.add_parser("trade", help="Run the live paper-trading loop")
    tr.add_argument("--symbol", default="BTCUSDT")
    tr.add_argument("--interval", default=None, help="candle interval (default: 1d for trendvol, 1h otherwise)")
    tr.add_argument("--poll", type=int, default=None)
    tr.add_argument("--fast", type=int, default=20)
    tr.add_argument("--slow", type=int, default=50)
    tr.add_argument("--strategy", choices=["sma", "trendvol"], default="sma")

    cmp = sub.add_parser("compare", help="Walk-forward out-of-sample portfolio comparison vs the S&P 500")
    cmp.add_argument("--assets", type=int, default=10, help="number of top-volume assets (1 = BTC only)")
    cmp.add_argument("--train-days", type=int, default=1095)
    cmp.add_argument("--test-days", type=int, default=365)
    cmp.add_argument("--fee", type=float, default=0.001)
    cmp.add_argument("--portfolio-vol", type=float, default=0.25, help="risk-managed overlay target vol")

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


if __name__ == "__main__":
    main()
