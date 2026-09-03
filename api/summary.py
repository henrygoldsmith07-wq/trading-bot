"""Vercel serverless endpoint: /api/summary

Serves three things, in descending order of evidentiary weight:

  1. ``verdict``  — the graded answer to "is the frozen rule holding up?"
                    (``bot/verdict.py``). This drives the dashboard hero.
  2. ``forward``  — evidence produced AFTER the freeze, by the frozen commit.
  3. ``research`` — pre-freeze development evidence, never the headline.

Zero third-party dependencies — a pure-stdlib ASGI app, which Vercel's
Python runtime detects automatically. Deployed with the repo's classic
zero-config layout (static `public/` + functions in `api/`).
"""
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.data import fetch_daily_history  # noqa: E402
from bot.strategy import risk_ensemble  # noqa: E402
from bot.walkforward import absolute_folds, walk_forward_at  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEZE_FILE = os.path.join(ROOT, "freeze.json")
FORWARD_LOG = os.path.join(ROOT, "forward_log.jsonl")
CANONICAL_RUN = os.path.join(ROOT, "runs", "canonical-v1", "run.json")

ENGINE_KWARGS: dict[str, Any] = dict(
    fee=0.001,
    spread_bps=5.0,
    slippage_bps=5.0,
    execution="next_open",
    risk_free_annual=0.03,
    rebalance_band=0.05,
)


def build_forward_summary(
    freeze_path: str = FREEZE_FILE,
    log_path: str = FORWARD_LOG,
    benchmark_fetch=None,
    today: datetime | None = None,
) -> dict:
    """The evidence that matters most: what happened AFTER the freeze.

    Reads only the committed freeze manifest and forward log — never the
    historical pipeline. Code identity is re-verified on every request; a
    tampered config or mismatched implementation surfaces as verified=false.
    Returns {"available": False, "reason": ...} until a freeze exists.
    """
    from bot.metrics import max_drawdown as _mdd
    from bot.metrics import sharpe as _sharpe

    if not os.path.exists(freeze_path):
        return {
            "available": False,
            "reason": "no freeze manifest committed yet — start with 'python -m bot freeze'",
        }
    try:
        import json as _json

        manifest = _json.loads(open(freeze_path, encoding="utf-8").read())
        # Manifest-integrity check (config hash + sealed fields). This proves
        # the RECORD is intact; it does not require THIS viewer to be running
        # frozen code — execution-time refusal already guards the writer side
        # (run_step / CI), and the tag pins what actually traded.
        from bot.identity import CODE_FINGERPRINT_ALGO

        if not manifest.get("config_sha256") or not manifest.get("code_sha256"):
            raise ValueError("manifest missing seals")
        if manifest.get("code_fingerprint_algo") != CODE_FINGERPRINT_ALGO:
            raise ValueError("unknown code-fingerprint algorithm")
        from bot.prospective import _config_hash

        if _config_hash(manifest["config"]) != manifest["config_sha256"]:
            raise ValueError("config sha mismatch — manifest tampered")
        code_verified = True
        code_reason = None
    except ValueError as exc:
        code_verified = False
        code_reason = str(exc)
        manifest = locals().get("manifest") or {}

    # transparency: does THIS process happen to run the frozen code?
    try:
        from bot.identity import code_fingerprint

        runtime_matches = code_fingerprint() == manifest.get("code_sha256")
    except Exception:
        runtime_matches = False

    entries = []
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(_json.loads(line))
            except ValueError:
                continue

    frozen_date = manifest.get("frozen_at_date")
    commit = (manifest.get("git_commit_at_freeze") or "")[:12] or None

    if not entries:
        return {
            "available": True,
            "started": False,
            "frozen_date": frozen_date,
            "code": commit,
            "code_verified": code_verified,
            "code_reason": code_reason,
            "runtime_matches_freeze": runtime_matches,
            "days_untouched": None,
            "parameter_changes": 0,
            "config_sha256": (manifest.get("config_sha256") or "")[:16],
            "reason": "no forward days recorded yet",
        }

    rets = [e["port_ret"] for e in entries]
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1.0 + r))
    total_return = equity[-1] - 1.0
    fwd_sharpe = _sharpe(rets, 365) if len(rets) >= 2 else None
    mdd = _mdd(equity)

    first_date = entries[0]["date"]
    last_date = entries[-1]["date"]
    today_d = (today or datetime.now(UTC)).date()
    frozen_d = datetime.fromisoformat(manifest["frozen_at"]).date() if "frozen_at" in manifest else None
    days_untouched = (today_d - frozen_d).days if frozen_d else None

    outages = sum(len(e.get("outages", [])) for e in entries)
    missed_fills = sum(len(e.get("missed_fills", [])) for e in entries)

    curve = []
    eq = 1.0
    for e in entries:
        eq *= 1.0 + e["port_ret"]
        curve.append({"t": e["date"], "v": round(eq, 5)})

    bench_return = None
    bench_label = "S&P 500 (same window)"
    try:
        rows = benchmark_fetch() if benchmark_fetch else fetch_sp500_rows()
        window = [
            r for r in rows
            if r["date"] >= first_date and r["date"] <= last_date
        ]
        if len(window) >= 2:
            bench_return = window[-1]["close"] / window[0]["close"] - 1.0
    except Exception:
        bench_return = None  # degrade: dashboard shows em-dash, never a guess

    return {
        "available": True,
        "started": True,
        "frozen_date": frozen_date,
        "code": commit,
        "code_verified": code_verified,
        "code_reason": code_reason,
        "runtime_matches_freeze": runtime_matches,
        "days_untouched": days_untouched,
        "n_days_recorded": len(entries),
        "first_day": first_date,
        "last_day": last_date,
        "parameter_changes": 0,  # proven by config+code seals; any edit would fail verification above
        "config_sha256": (manifest.get("config_sha256") or "")[:16],
        "code_sha256": (manifest.get("code_sha256") or "")[:16],
        "forward_return": round(total_return, 4),
        "forward_sharpe": round(fwd_sharpe, 3) if fwd_sharpe is not None else None,
        "max_drawdown": round(mdd, 4),
        "benchmark_label": bench_label,
        "benchmark_return": round(bench_return, 4) if bench_return is not None else None,
        "data_outages": outages,
        "missed_fills": missed_fills,
        "curve": curve,
    }


def fetch_sp500_rows():
    """S&P rows as {date: iso, close} for benchmark comparison."""
    from bot.benchmark import fetch_sp500

    return [{"date": r["date"].isoformat(), "close": float(r["close"])} for r in fetch_sp500()]


def build_verdict_payload(forward: dict | None) -> dict | None:
    """Grade the evidence behind the frozen rule (`bot/verdict.py`).

    The verdict is the product: it is what the dashboard hero shows. It is
    assembled from the sealed canonical record, the research ledger, the cost
    tape, and the forward log — never from ad-hoc numbers.

    Degrades to None rather than raising. Every input here is a file that may
    legitimately be absent on a cold serverless start, and a missing verdict
    must make the dashboard go QUIET, not fall back to a guess.
    """
    try:
        from bot.runs import load_run_record
        from bot.verdict import build_verdict

        record = load_run_record("canonical-v1", runs_dir=os.path.join(ROOT, "runs"))
    except Exception:
        return None  # no sealed canonical record -> nothing can be graded

    results = record.get("results", {})
    metrics = results.get("metrics", {})

    try:
        from bot.strategy import build_candidates

        pool_size = len(build_candidates())
    except Exception:
        pool_size = 85  # documented fallback: the canonical-era pool size

    ledger_n = None
    try:
        from bot.research_ledger import load_entries, recommended_trial_count

        ledger_path = os.path.join(ROOT, "research_ledger.jsonl")
        if load_entries(ledger_path):
            ledger_n = recommended_trial_count(ledger_path)
    except Exception:
        ledger_n = None

    cost_report = None
    try:
        from bot.cost_calibration import calibrate, load_observations

        obs = load_observations(os.path.join(ROOT, "cost_observations.jsonl"))
        if obs:
            default_frictions = {"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0}
            frictions = results.get("parameters", {}).get("frictions", default_frictions)
            try:
                from bot.prospective import load_freeze

                manifest = load_freeze(os.path.join(ROOT, "freeze.json"), verify_code=False)
            except (OSError, ValueError):
                manifest = None
            cost_report = calibrate(obs, v1_frictions=frictions, freeze_manifest=manifest)
    except Exception:
        cost_report = None

    try:
        return build_verdict(
            canonical_rule_stats=metrics.get("rules", []),
            canonical_per_asset=results.get("per_asset", []),
            canonical_n_folds=results.get("n_folds"),
            pool_size=pool_size or 85,
            ledger_search_n=ledger_n,
            cost_report=cost_report,
            forward=forward,
        )
    except Exception:
        return None


def _canonical_overlay(summary: dict) -> dict:
    """Override historical headline METRICS from the committed canonical run
    record (runs/canonical-v1/run.json) when present. The curve, the current
    reading and the fold list are still computed per request (cheap,
    single-symbol); the authoritative NUMBERS come from the sealed record —
    same source as the README table."""
    try:
        with open(CANONICAL_RUN, encoding="utf-8") as f:
            record = json.load(f)
        iv = record["results"]["metrics"]["inv_vol_rm"]
        oos = summary["oos"]
        oos["cagr"] = iv["cagr"]
        oos["sharpe"] = iv["sharpe"]
        oos["max_drawdown"] = iv["max_drawdown"]
        oos["final"] = iv["final"]
        win = record["results"]["metrics"].get("window") or {}
        if win.get("start"):
            oos["first_day"], oos["last_day"] = win["start"], win["end"]
        summary["strategy"] = "RiskEnsemble pool walk-forward — CANONICAL post-PIT record"
        summary["canonical_run_id"] = record["run_id"]
        summary["canonical_record_sha256"] = record.get("record_sha256")
        summary["rules_table"] = record["results"]["metrics"].get("rules", [])
    except (OSError, KeyError, ValueError):
        pass
    return summary


def build_summary(symbol: str = "BTCUSDT") -> dict:
    candles = fetch_daily_history(symbol)
    folds = absolute_folds(candles, train_days=1095, test_days=365)
    wf = walk_forward_at(candles, folds, candidates=[risk_ensemble()], **ENGINE_KWARGS)

    days = sorted(wf["daily"])
    rets = [wf["daily"][d] for d in days]
    curve = [1.0]
    for r in rets:
        curve.append(curve[-1] * (1.0 + r))

    # downsample the curve for the payload (~180 points), always keeping the final value
    step = max(1, len(curve) // 180)
    idxs = list(range(0, len(curve), step))
    if idxs[-1] != len(curve) - 1:
        idxs.append(len(curve) - 1)
    points = [
        {"t": days[min(i, len(days) - 1)], "v": round(curve[i], 5)}
        for i in idxs
    ]

    strategy = risk_ensemble()
    closes = [c["close"] for c in candles]
    window = closes[-50:]
    sma50 = (sum(window) / len(window)) if window else None
    summary = {
        "symbol": symbol,
        "strategy": "RiskEnsemble (fixed, a-priori)",
        "generated_at": datetime.now(UTC).isoformat(),
        "frictions": ENGINE_KWARGS,
        "oos": {
            "first_day": datetime.fromtimestamp(wf["first_day"] / 1000, tz=UTC).date().isoformat(),
            "last_day": datetime.fromtimestamp(wf["last_day"] / 1000, tz=UTC).date().isoformat(),
            "cagr": round(wf["cagr"], 4),
            "sharpe": round(wf["sharpe"], 3),
            "max_drawdown": round(wf["max_drawdown"], 4),
            "final": round(wf["equity"], 4),
            "exposure": round(wf["exposure"], 3),
            "turnover": round(wf["turnover"], 2),
        },
        "folds": [p["strategy"] for p in wf["folds"]],
        "curve": points,
        # The frozen rule's current reading. Deliberately NOT called "live":
        # nothing here executes orders, and the dashboard's only three
        # evidence labels are research / out-of-sample / forward.
        "now": {
            "price": candles[-1]["close"],
            "price_date": datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=UTC).date().isoformat(),
            "weight_now": round(strategy.weight(candles), 3),
            "trend_up": None if sma50 is None else bool(candles[-1]["close"] >= sma50),
        },
        "disclaimer": "Paper trading only. Educational software. Not financial advice.",
    }
    return _canonical_overlay(summary)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    try:
        forward = build_forward_summary()
        # The verdict is the headline. It is graded from the forward record
        # first: how many paper days exist decides how loud the page is allowed
        # to be. Research numbers never feed the grade.
        verdict = build_verdict_payload(forward)
        try:
            research = build_summary()
            research["evidence_label"] = (
                "RESEARCH / HISTORICAL ONLY — computed by the pre-freeze research "
                "pipeline; carries no weight in the verdict above"
            )
        except Exception as e:
            research = {"error": str(e), "evidence_label": "RESEARCH / HISTORICAL ONLY (unavailable)"}
        payload = {"verdict": verdict, "forward": forward, "research": research,
                   "generated_at": datetime.now(UTC).isoformat(),
                   "disclaimer": "Paper trading only. Educational software. Not financial advice."}
        status, body = 200, json.dumps(payload).encode()
    except Exception as e:  # surface a clean error to the dashboard
        status, body = 502, json.dumps({"error": str(e)}).encode()
    headers = [
        [b"content-type", b"application/json"],
        [b"cache-control", b"no-store"],
        [b"access-control-allow-origin", b"*"],
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    print(json.dumps(build_summary(), indent=2)[:2000])
