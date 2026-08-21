"""CLI entry point.

Usage:
  python -m bot backtest --symbol BTCUSDT --interval 1h
  python -m bot trade    --symbol BTCUSDT --interval 1h
"""
from __future__ import annotations

import argparse

from .backtest import backtest
from .data import fetch_candles
from .strategy import SmaCrossover


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
    tr.add_argument("--interval", default="1h")
    tr.add_argument("--poll", type=int, default=60)
    tr.add_argument("--fast", type=int, default=20)
    tr.add_argument("--slow", type=int, default=50)

    args = parser.parse_args()

    if args.command == "backtest":
        candles = fetch_candles(args.symbol, args.interval, limit=500)
        results = backtest(candles, SmaCrossover(args.fast, args.slow))
        print(f"Backtest {args.symbol} {args.interval} (SMA {args.fast}/{args.slow}):")
        for k, v in results.items():
            print(f"  {k}: {v}")
    elif args.command == "trade":
        from .paper import run

        run(args.symbol, args.interval, args.poll, args.fast, args.slow)


if __name__ == "__main__":
    main()
