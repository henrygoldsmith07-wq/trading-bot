"""CLI entry point.

Usage:
  python -m bot backtest --symbol BTCUSDT --interval 1h
  python -m bot trade    --symbol BTCUSDT --interval 1h
  python -m bot compare  --symbol BTCUSDT            # walk-forward vs S&P 500
"""
from __future__ import annotations

import argparse

from .backtest import backtest
from .data import fetch_candles
from .strategy import SmaCrossover


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def run_compare(args) -> int:
    from .benchmark import equity_metrics, fetch_sp500, slice_window
    from .data import fetch_daily_history
    from .engine import candle_date, run_strategy
    from .walkforward import _fold_boundaries, walk_forward

    print(f"Fetching daily history for {args.symbol}...")
    candles = fetch_daily_history(args.symbol)
    print(f"  {len(candles)} candles: {candle_date(candles[0])} -> {candle_date(candles[-1])}")

    print("Fetching S&P 500 daily history (FRED)...")
    sp = fetch_sp500()
    print(f"  {len(sp)} rows: {sp[0]['date']} -> {sp[-1]['date']}")

    print(f"Walk-forward: train {args.train_days}d / test {args.test_days}d, fee {args.fee:.2%} per unit turnover...")
    oos = walk_forward(candles, train_days=args.train_days, test_days=args.test_days, fee=args.fee)

    folds = _fold_boundaries(candles, args.train_days, args.test_days)
    oos_start = candle_date(candles[folds[0][1]])
    oos_end = candle_date(candles[folds[-1][2]])

    sp_window = slice_window(sp, oos_start, oos_end)
    spx = equity_metrics(sp_window)

    bh = run_strategy(candles[folds[0][1]:], lambda c: 1.0, fee=0.0)

    print()
    print(f"Out-of-sample window: {oos_start} -> {oos_end} ({oos['n_folds']} yearly folds)")
    picks = {}
    for f in oos["folds"]:
        picks[f["strategy"]] = picks.get(f["strategy"], 0) + 1
    for name, count in picks.items():
        print(f"  picked {name} x{count}")
    print()
    header = f"{'':16}{'Bot (OOS)':>14}{'S&P 500':>14}{'BTC buy&hold':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'CAGR':16}{_fmt_pct(oos['cagr']):>14}{_fmt_pct(spx['cagr']):>14}{_fmt_pct(bh['cagr']):>14}")
    print(f"{'Volatility':16}{_fmt_pct(oos['vol']):>14}{_fmt_pct(spx['vol']):>14}{_fmt_pct(bh['vol']):>14}")
    print(f"{'Sharpe':16}{oos['sharpe']:>14.2f}{spx['sharpe']:>14.2f}{bh['sharpe']:>14.2f}")
    print(f"{'Max drawdown':16}{_fmt_pct(oos['max_drawdown']):>14}{_fmt_pct(spx['max_drawdown']):>14}{_fmt_pct(bh['max_drawdown']):>14}")
    print(f"{'Growth of $1':16}{oos['equity']:>14.2f}{spx['final']:>14.2f}{bh['final']:>14.2f}")
    print()

    beats_cagr = oos["cagr"] > spx["cagr"]
    beats_sharpe = oos["sharpe"] > spx["sharpe"]
    beats_mdd = oos["max_drawdown"] > spx["max_drawdown"]
    verdict = (
        f"VERDICT: bot OOS CAGR {'BEATS' if beats_cagr else 'trails'} S&P 500 "
        f"({_fmt_pct(oos['cagr'])} vs {_fmt_pct(spx['cagr'])}); "
        f"Sharpe {'beats' if beats_sharpe else 'trails'} ({oos['sharpe']:.2f} vs {spx['sharpe']:.2f}); "
        f"max drawdown {'better' if beats_mdd else 'worse'} ({_fmt_pct(oos['max_drawdown'])} vs {_fmt_pct(spx['max_drawdown'])})"
    )
    print(verdict)
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

    cmp = sub.add_parser("compare", help="Walk-forward out-of-sample comparison vs the S&P 500")
    cmp.add_argument("--symbol", default="BTCUSDT")
    cmp.add_argument("--train-days", type=int, default=1095)
    cmp.add_argument("--test-days", type=int, default=365)
    cmp.add_argument("--fee", type=float, default=0.001)

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
