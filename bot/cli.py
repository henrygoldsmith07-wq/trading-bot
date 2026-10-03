"""CLI argument parsing and command dispatch.

Separated from the command implementations so the entry point is a
list of wiring rather than a mix of wiring and behaviour.
"""
from __future__ import annotations

import argparse
import sys

from .backtest import backtest
from .commands import (
    run_ask,
    run_calibrate_costs,
    run_compare,
    run_compare_experiments,
    run_evidence,
    run_experiment_status,
    run_forward,
    run_freeze,
    run_ledger,
    run_quarantine_costs,
    run_reproduce,
    run_research,
    run_sensitivity,
    run_universe_snapshot,
    run_validate,
    run_verdict,
    run_verify_evidence,
    run_verify_freeze,
)
from .data import fetch_candles
from .strategy import SmaCrossover


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
    cmp.add_argument("--seed", type=int, default=42, help="recorded for reproducibility (pipeline is deterministic)")
    cmp.add_argument("--cache-only", action="store_true", help="never refresh datasets from the network")
    cmp.add_argument("--run-id", default=None, help='name the saved run record (e.g. "canonical-v1")')
    # Pin the exact symbol set instead of re-ranking today's 24h volume. Two
    # runs over the SAME symbols isolate a methodology change; two runs over
    # whatever happens to be trending today conflate it with a universe change.
    # Re-running a canonical record without this flag silently trades a
    # different portfolio than the one it is meant to be compared against.
    cmp.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="allow a canonical run from a dirty working tree; the record is then NOT canonical evidence",
    )
    cmp.add_argument(
        "--canonical",
        action="store_true",
        help="mark this run canonical (implied by a --run-id starting with canonical-); requires a clean tree",
    )

    cmp.add_argument(
        "--universe-symbols",
        default=None,
        help="comma-separated symbols to use verbatim, in place of the live top-N volume ranking",
    )

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
    frz.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="allow freezing from a dirty working tree; the manifest then cannot name its own source",
    )
    frz.add_argument("--no-tag", action="store_true", help="skip creating the freeze/<date> git tag")
    frz.add_argument("--tag", default=None, help="override the tag name (default: freeze/<YYYYMMDD>)")
    frz.add_argument("--image-digest", default=None, help='record a container digest, e.g. "sha256:..."')
    # --- full-algorithm knobs: every return-affecting quantity is frozen ----
    frz.add_argument("--fixed", action="store_true", help="freeze the fully-fixed RiskEnsemble (N=1) pipeline")
    frz.add_argument("--band", type=float, default=0.05, help="rebalance band fraction (default 5%%)")
    frz.add_argument("--no-tilt", action="store_true", help="disable XS-momentum tilt")
    frz.add_argument("--tilt-lookback", type=int, default=90)
    frz.add_argument("--max-tilt", type=float, default=0.5)
    frz.add_argument("--no-crisis", action="store_true", help="disable crisis de-risking")
    frz.add_argument("--corr-window", type=int, default=60)
    frz.add_argument("--corr-threshold", type=float, default=0.60)
    frz.add_argument("--derisk", type=float, default=0.60)
    frz.add_argument("--throttle", action="store_true", help="enable drawdown throttle")
    frz.add_argument("--dd-trigger", type=float, default=-0.10)
    frz.add_argument("--dd-exit", type=float, default=-0.05)
    frz.add_argument("--throttle-factor", type=float, default=0.5)
    frz.add_argument("--no-overlay", action="store_true", help="disable the trailing-vol target overlay")
    frz.add_argument("--vol-window", type=int, default=20, help="inverse-vol weighting window")
    frz.add_argument("--max-multiple-of-equal", type=float, default=2.0)

    ver = sub.add_parser("verify-freeze", help="Refuse unless running code matches the frozen implementation")
    ver.add_argument("--freeze-file", default="freeze.json")

    led = sub.add_parser("ledger", help="Report the research ledger: every experiment ever searched")
    led.add_argument("--ledger", default="research_ledger.jsonl")

    vd = sub.add_parser("verdict", help="How much evidence supports the frozen system? (the product)")
    vd.add_argument("--json", action="store_true")
    vd.add_argument("--root", default=".", help="repository root (default: cwd)")

    uni = sub.add_parser("universe-snapshot", help="Record today's ranked universe (point-in-time dataset)")
    uni.add_argument("--top", type=int, default=20)
    uni.add_argument("--log", default="universe_log.jsonl")

    cal = sub.add_parser("calibrate-costs", help="Predicted vs observed trading costs from paper observations")
    cal.add_argument("--observations", default="cost_observations.jsonl")
    cal.add_argument("--fee", type=float, default=0.001)
    cal.add_argument("--spread-bps", type=float, default=5.0)
    cal.add_argument("--slippage-v1-bps", type=float, default=5.0, dest="slippage_v1_bps")
    cal.add_argument("--min", type=int, default=30, dest="min")
    cal.add_argument("--write", action="store_true", help="write cost_calibration.json (V2 proposal)")

    rep = sub.add_parser("reproduce", help="Re-execute a saved benchmark run and verify identical results")
    rep.add_argument("run_id", help="run id, or 'list' to enumerate saved runs")
    rep.add_argument("--runs-dir", default="runs")

    q = sub.add_parser("quarantine-costs", help="Classify legacy cost observations; preserve every byte")
    q.add_argument("--src", default="cost_observations.jsonl")
    q.add_argument("--archive", default="archive/cost_observations.pre-purity.jsonl")
    q.add_argument("--quarantine", default="cost_observations_quarantine.jsonl")
    q.add_argument("--keep", default=None, help="pure forward-paper tape (default: overwrite --src)")
    q.add_argument("--freeze-file", default="freeze.json")

    fwd = sub.add_parser("forward", help="Prospective paper trading: --step one day, --report checkpoints")
    fwd.add_argument("--step", action="store_true", help="execute one forward day from the freeze")
    fwd.add_argument("--report", action="store_true", help="print the checkpoint report")
    fwd.add_argument("--freeze-file", default="freeze.json")
    fwd.add_argument("--log-file", default="forward_log.jsonl")
    fwd.add_argument(
        "--as-of-date",
        default=None,
        help="ISO session date to account (decouples the market day from delayed runner wall-clock time)",
    )

    fve = sub.add_parser(
        "forward-evidence",
        help="Merge or verify forward evidence tapes (the scheduled paper run's evidence transfer)",
    )
    fve.add_argument("--staging", default=None, help="staged evidence directory produced by the frozen worktree")
    fve.add_argument("--repo", default=".", help="repository root (default: cwd)")
    fve.add_argument("--verify", action="store_true", help="validate the repository evidence tapes instead of merging")
    fve.add_argument(
        "--allow-freeze-change",
        action="store_true",
        help="merge even if the active freeze moved (quarantine a run; never for scoring)",
    )

    # --- evidence architecture: one document, every surface ---------------
    ev = sub.add_parser("evidence", help="The evidence document: what is tested, what proves it, how strong")
    ev.add_argument("--json", action="store_true", help="print the full evidence document as JSON")
    ev.add_argument("--out", default=None, help="write the canonical evidence.json to this path and exit")
    ev.add_argument("--root", default=".", help="repository root (default: cwd)")

    es = sub.add_parser("experiment-status", help="Experiment state machine status and forward checkpoints")
    es.add_argument("--root", default=".", help="repository root (default: cwd)")

    ve = sub.add_parser("verify-evidence", help="Pass/fail verification of the whole evidence chain")
    ve.add_argument("--root", default=".", help="repository root (default: cwd)")

    cex = sub.add_parser("compare-experiments", help="Compare two experiments; flag confounded deltas")
    cex.add_argument("left", help="left run id (or 'current')")
    cex.add_argument("right", help="right run id (or 'current')")
    cex.add_argument("--root", default=".", help="repository root (default: cwd)")

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
    elif args.command == "ledger":
        raise SystemExit(run_ledger(args))
    elif args.command == "verdict":
        raise SystemExit(run_verdict(args))
    elif args.command == "universe-snapshot":
        raise SystemExit(run_universe_snapshot(args))
    elif args.command == "calibrate-costs":
        raise SystemExit(run_calibrate_costs(args))
    elif args.command == "reproduce":
        raise SystemExit(run_reproduce(args))
    elif args.command == "quarantine-costs":
        if args.keep is None:
            args.keep = args.src  # production tape replaced by verified-only rows
        raise SystemExit(run_quarantine_costs(args))
    elif args.command == "forward":
        if not (args.step or args.report):
            print("use --step and/or --report")
        raise SystemExit(run_forward(args))
    elif args.command == "forward-evidence":
        from .forward_evidence import main as _fe_main

        if args.verify:
            raise SystemExit(_fe_main(["verify", "--repo", args.repo]))
        if not args.staging:
            print("::error::forward-evidence needs --staging (to merge) or --verify (to validate)")
            raise SystemExit(2)
        _argv = ["merge", "--staging", args.staging, "--repo", args.repo]
        if args.allow_freeze_change:
            _argv.append("--allow-freeze-change")
        raise SystemExit(_fe_main(_argv))
    elif args.command == "evidence":
        raise SystemExit(run_evidence(args))
    elif args.command == "experiment-status":
        raise SystemExit(run_experiment_status(args))
    elif args.command == "verify-evidence":
        raise SystemExit(run_verify_evidence(args))
    elif args.command == "compare-experiments":
        raise SystemExit(run_compare_experiments(args))


if __name__ == "__main__":
    main()
