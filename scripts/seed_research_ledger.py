"""Seed research_ledger.jsonl from the project's documented history.

Run once:  python scripts/seed_research_ledger.py
Every entry is marked backfilled=true with its source commit, so the
provenance of pre-ledger experiments is auditable rather than folklore.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.research_ledger import append_entry, load_entries, summarize, verify_chain  # noqa: E402

LEDGER = Path(__file__).resolve().parent.parent / "research_ledger.jsonl"

# (commit, date) anchors from `git log` — the day each idea entered the codebase
C_INIT = ("dc531c1", "2026-08-21T19:58:56+01:00")
C_SP = ("47a1281", "2026-08-22T06:34:53+01:00")
C_10X = ("cfa4e3a", "2026-08-22T06:57:50+01:00")
C_QUALITY = ("2e1e512", "2026-08-22T07:10:49+01:00")
C_STATS = ("ac7c440", "2026-08-22T07:26:09+01:00")
C_PROSPECT = ("a158478", "2026-08-22T13:21:29+01:00")
C_ALGO = ("6e4cbce", "2026-08-22T13:47:03+01:00")
C_PORTFOLIO = ("904c4d5", "2026-08-22T14:02:29+01:00")
C_DASH = ("46b5e2d", "2026-08-23T07:02:38+01:00")
C_METH = ("48e21d1", "2026-08-23T09:21:15+01:00")
C_ALGO_FREEZE = ("4501da3", "2026-08-23T18:06:49+01:00")
C_TRANSITION = ("38627c7", "2026-08-23T20:50:30+01:00")


def add(anchor, category, hypothesis, config, metric, result, accepted):
    commit, ts = anchor
    append_entry(
        LEDGER,
        category=category,
        hypothesis=hypothesis,
        configuration=config,
        primaryMetric=metric,
        result=result,
        accepted=accepted,
        source_commit=commit,
        backfilled=True,
        timestamp=ts,  # historical time, sealed into the hash
    )


def main() -> int:
    if LEDGER.exists():
        print(f"{LEDGER.name} already exists — refusing to double-seed")
        return 1
    E = [
        # --- strategy families -------------------------------------------------
        (C_INIT, "strategy", "SMA crossover baseline can trade BTC on paper signals", {"fast": 20, "slow": 50}, "OOS Sharpe", 0.0, True),
        (C_10X, "strategy", "TrendVol grid (lookback x vol-target) beats SMA baseline walk-forward", {"family": "TrendVol", "lookbacks": [25, 50, 75, 100, 125, 150, 200], "targets": [0.2, 0.25, 0.3, 0.4, 0.5]}, "OOS Sharpe", 1.16, True),
        (C_10X, "strategy", "RSI dip-buy family adds orthogonal entries in uptrends", {"period": [2, 3, 4], "exit": [55, 60, 65, 70], "trend_filter": [150, 200]}, "OOS Sharpe", 0.9, True),
        (C_10X, "strategy", "MACD histogram trend filter earns its pool slot", {"sets": [[12, 26, 9], [8, 21, 5], [16, 32, 9]]}, "OOS Sharpe", 0.7, True),
        (C_ALGO, "strategy", "Time-series momentum (TSMom) family generalizes TrendVol", {"horizons": [63, 126, 189], "targets": [0.3, 0.35, 0.45]}, "OOS Sharpe", 1.0, True),
        (C_ALGO, "strategy", "Multi-horizon DualMomentum scales conviction by fraction of positive horizons", {"horizons": [[63, 126, 252], [42, 84, 168], [63, 126]]}, "OOS Sharpe", 1.05, True),
        (C_ALGO, "strategy", "Fixed ensembles diversify regime bets inside a single sleeve", {"members": "trend+momentum+dip blends"}, "OOS Sharpe", 1.02, True),
        (C_ALGO, "portfolio", "A-priori RiskEnsemble (N=1 trials) clears DSR where 85-trial selection cannot", {"blend": "TrendVol50+100+200 / TSMom / Dip"}, "DSR", 0.961, True),
        (C_QUALITY, "universe", "Top-N crypto pairs by 24h quote volume + SPY/GLD/TLT covers classes without paid data", {"N": 20, "etfs": ["SPY", "GLD", "TLT"]}, "coverage", 23, True),
        (C_QUALITY, "universe", "Excluding stablecoins/leveraged tokens removes non-vanilla candidates", {"exclusions": "stable+leveraged suffixes"}, "coverage", 0, True),
        # --- portfolio / risk ---------------------------------------------------
        (C_SP, "portfolio", "Equal-weight fixed-denominator combine is survivorship-safe", {"weighting": "equal", "denominator": "selected assets"}, "OOS Sharpe", 1.15, True),
        (C_ALGO, "portfolio", "Inverse-volatility weighting cuts drawdown at equal return vs equal weight", {"vol_window": 20, "cap": "2x equal"}, "OOS Sharpe", 1.05, True),
        (C_PORTFOLIO, "portfolio", "XS-momentum tilt (+/-50% band, 90d lookback) improves inv-vol without gross change", {"lookback": 90, "max_tilt": 0.5}, "OOS Sharpe", 1.10, True),
        (C_PORTFOLIO, "portfolio", "Crisis de-risking (avg pairwise corr > 0.6 -> x0.6 exposure) protects tail", {"corr_window": 60, "threshold": 0.6, "multiplier": 0.6}, "OOS Sharpe", 1.12, True),
        (C_DASH, "portfolio", "Drawdown throttle (halve past -10%, restore > -5%) trades CAGR for maxDD", {"dd_trigger": -0.10, "dd_exit": -0.05, "factor": 0.5}, "OOS Sharpe", 0.94, False),
        (C_DASH, "portfolio", "5% rebalance band reduces turnover cost without losing edge", {"band": 0.05}, "OOS Sharpe", 1.23, True),
        (C_DASH, "portfolio", "Fully-fixed RiskEnsemble pipeline (no per-asset selection anywhere) is defensible but weaker", {"selection_mode": "fixed_risk_ensemble"}, "OOS Sharpe", 0.44, False),
        (C_PORTFOLIO, "methodology", "Overlay ablation battery attributes headline to specific mechanisms", {"variants": ["equal", "inv_vol", "+tilt", "+crisis"]}, "attribution", 4, True),
        # --- execution ----------------------------------------------------------
        (C_INIT, "execution", "Close-price execution is an optimistic baseline; next_open is realistic default", {"modes": ["close", "next_open"]}, "OOS Sharpe", 1.03, True),
        (C_QUALITY, "execution", "Latency sweep 0/1/2 days quantifies signal-delay robustness", {"latencies": [0, 1, 2]}, "OOS Sharpe delta", 0.0, True),
        (C_QUALITY, "execution", "Combined-cost sweep 5..100bp maps friction survival curve", {"bps": [5, 10, 20, 50, 100]}, "OOS Sharpe @100bp", 0.84, True),
        (C_METH, "execution", "Volatility-dependent spread/slippage ((rv/ref)^0.5 clamped) prices stress regimes honestly", {"scale_ref": 0.6, "floor": 0.5, "cap": 3.0}, "opt-in model", 1, True),
        (C_METH, "execution", "Square-root impact k*vol*sqrt(participation) bounds illiquid fills", {"k": "config", "adv_to_equity": "config"}, "opt-in model", 1, True),
        (C_METH, "execution", "Tiered maker/taker fees for account-level paper broker", {"tiers": "retail schedule"}, "opt-in model", 1, True),
        (C_METH, "execution", "Liquidity filter on median daily quote volume blocks untradeable alts", {"min_volume_usd": 5e6}, "filter", 1, True),
        (C_METH, "execution", "ETF calendar alignment: invested-flat weekends, cash for late/delisted spans", {"rule": "interior-gap invested"}, "OOS Sharpe delta", 0.001, True),
        (C_METH, "execution", "Forward-fill small data gaps (<=3d) with flagged zero-volume bars", {"max_gap_days": 3}, "gaps filled", 1, True),
        (C_METH, "execution", "Delisting simulation with terminal liquidation cost strands sleeves safely", {"terminal_cost_bps": 10}, "test harness", 1, True),
        (C_TRANSITION, "execution", "Mark-chained transition accounting removes snapshot->close P&L gap (parity proven)", {"exec": "open", "anchor": "mark"}, "parity diff", 5e-9, True),
        (C_PROSPECT, "execution", "Stale-feed exclusion (>45d silent history) prevents zombie sleeves", {"max_age_days": 45}, "filter", 1, True),
        # --- statistical methodology (recorded, excluded from search N) ---------
        (C_STATS, "methodology", "PSR/DSR + White's RC + stationary bootstrap battery replaces eyeballing", {"tools": ["PSR", "DSR", "RC", "SB"]}, "tooling", 1, True),
        (C_METH, "methodology", "SPA test + circular/moving bootstraps + Bayesian Sharpe extend inference", {"added": ["SPA", "CBB", "MBB", "Bayes"]}, "tooling", 1, True),
        (C_STATS, "methodology", "Nested purged walk-forward separates selection overfitting from edge", {"purge_days": 220, "embargo_days": 30}, "DSR delta", 0.11, True),
    ]
    if not LEDGER.exists():
        for anchor, *rest in sorted(E, key=lambda e: e[0][1]):
            add(anchor, *rest)
    entries = load_entries(LEDGER)
    verify_chain(entries)
    s = summarize(entries)
    print(f"seeded {len(entries)} entries; search total={s['recommended_trial_count']} "
          f"({s['by_category']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
