"""Prospective validation: freeze, then walk forward without touching anything.

Discipline enforced by design:
- `create_freeze` snapshots the exact per-asset strategy picks, universe,
  frictions, and overlay settings into a hash-sealed manifest with a UTC
  timestamp (committing it to git makes the freeze tamper-evident). It also
  seals the IMPLEMENTATION: a code fingerprint over the bot's source is
  recorded alongside the config hash.
- `load_freeze` verifies the config hash and — by default — the code
  fingerprint. `run_step` refuses to trade when the running source differs
  from the frozen implementation: a freeze pins behaviour, not just numbers,
  so the scheduled runner must execute the frozen commit (the CI workflow
  checks out `git_commit_at_freeze`), never an edited main.
- Every step appends to a JSONL log recording prices, weights, realized
  slippage, data outages, and missed fills.
- `report` compares bot vs S&P 500 vs BTC at 1/3/6/12-month checkpoints and
  publishes the monthly return table — negative months included.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from .identity import CODE_FINGERPRINT_ALGO, code_fingerprint, verify_freeze_code
from .strategy import strategy_from_spec, strategy_to_spec

FREEZE_FILE = "freeze.json"
LOG_FILE = "forward_log.jsonl"
CHECKPOINTS = [("1 month", 30), ("3 months", 91), ("6 months", 182), ("12 months", 365)]


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def create_freeze(
    assets: list[dict],
    frictions: dict,
    overlay: dict,
    path: str | Path = FREEZE_FILE,
    now: datetime | None = None,
    git_commit: str | None = None,
    image_digest: str | None = None,
    git_tag: str | None = None,
) -> dict:
    """Write the freeze manifest. `assets`: [{symbol, source, periods_per_year,
    strategy (object)}]. The config hash seals the exact parameterization and
    the code fingerprint seals the exact implementation that must execute it.

    `image_digest` (sha256:... of a built container) and `git_tag` are
    recorded when provided so the artifact trail is auditable end to end.
    """
    config = {
        "assets": [
            {
                "symbol": a["symbol"],
                "source": a["source"],
                "periods_per_year": a["periods_per_year"],
                "strategy": strategy_to_spec(a["strategy"]),
            }
            for a in assets
        ],
        "frictions": frictions,
        "overlay": overlay,
    }
    manifest = {
        "frozen_at": (now or datetime.now(UTC)).isoformat(),
        "frozen_at_date": (now or datetime.now(UTC)).date().isoformat(),
        "git_commit_at_freeze": git_commit,
        "git_tag": git_tag,
        "image_digest": image_digest,
        "retune_policy": "FORWARD PERIOD IS NEVER USED FOR SELECTION OR TUNING",
        "code_policy": "RUNNER MUST EXECUTE THE FROZEN COMMIT; REFUSE ON CODE MISMATCH",
        "config": config,
        "config_sha256": _config_hash(config),
        "code_fingerprint_algo": CODE_FINGERPRINT_ALGO,
        "code_sha256": code_fingerprint(),
    }
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_freeze(path: str | Path = FREEZE_FILE, verify_code: bool = True) -> dict:
    """Load and verify the manifest. Raises on config tampering and, unless
    `verify_code=False`, on any implementation mismatch with the running tree."""
    manifest = json.loads(Path(path).read_text())
    actual = _config_hash(manifest["config"])
    if actual != manifest.get("config_sha256"):
        raise ValueError("freeze manifest hash mismatch — config was modified after freezing")
    if verify_code:
        verify_freeze_code(manifest)
    return manifest


def load_log(path: str | Path = LOG_FILE) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_log(entry: dict, path: str | Path = LOG_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def trailing_overlay_weight(port_rets: list[float], target: float, window: int = 20) -> float:
    """Risk-overlay weight for today from the trailing window (no lookahead)."""
    import math

    if len(port_rets) < window:
        return 1.0
    hist = port_rets[-window:]
    m = sum(hist) / window
    var = sum((x - m) ** 2 for x in hist) / (window - 1)
    rv = math.sqrt(max(var, 0.0) * 365)
    if rv <= 0:
        return 1.0
    return min(1.0, target / rv)


def run_step(
    manifest: dict,
    fetcher,
    now: datetime | None = None,
    log_path: str | Path = LOG_FILE,
    allow_code_mismatch: bool = False,
) -> dict:
    """One forward day of paper trading from the frozen config only.

    `fetcher(symbol, source)` -> (candles, problem) where problem is None or a
    string describing an outage. Returns the log entry (or the existing entry
    if today was already logged — steps are idempotent per date).

    Refuses to execute unless the running source matches the manifest's
    frozen implementation. `allow_code_mismatch=True` is an explicit
    research-replay escape hatch — the scheduled runner never passes it.
    """
    if not allow_code_mismatch:
        verify_freeze_code(manifest)
    now = now or datetime.now(UTC)
    today = now.date().isoformat()
    log = load_log(log_path)
    for e in log:
        if e["date"] == today:
            return {"status": "already_logged", "entry": e}

    config = manifest["config"]
    frictions = config["frictions"]
    cost_rate = frictions["fee"] + (frictions.get("spread_bps", 0) + frictions.get("slippage_bps", 0)) / 10_000.0
    prev_entries = log
    prev_weights = {}
    if prev_entries:
        for sym, detail in prev_entries[-1].get("assets", {}).items():
            prev_weights[sym] = detail.get("weight", 0.0)
    port_history = [e["port_ret"] for e in prev_entries]

    n_assets = len(config["assets"])
    outages = []
    alerts = []  # non-fatal warnings (stale prints) that still make the audit
    missed_fills = []
    sleeve_rets = []
    asset_details = {}
    for a in config["assets"]:
        sym = a["symbol"]
        strategy = strategy_from_spec(a["strategy"])  # raises on tampered spec: no fallback
        candles, problem = fetcher(sym, a["source"])
        if problem:
            outages.append({"symbol": sym, "problem": problem})
            w_prev = prev_weights.get(sym, 0.0)
            sleeve_rets.append(0.0)
            asset_details[sym] = {"weight": w_prev, "price": None, "slippage_bps": None, "note": problem}
            continue
        # use only completed candles for the decision
        completed = [c for c in candles if datetime.fromtimestamp(c["open_time"] / 1000, tz=UTC).date() < now.date()]
        if len(completed) < 2:
            outages.append({"symbol": sym, "problem": "insufficient completed candles"})
            sleeve_rets.append(0.0)
            asset_details[sym] = {"weight": prev_weights.get(sym, 0.0), "price": None, "slippage_bps": None, "note": "insufficient history"}
            continue
        # data-staleness alert: a print much older than one bar means the feed
        # is silently frozen; trade the stale weight but say so loudly
        age_days = (now - datetime.fromtimestamp(completed[-1]["open_time"] / 1000, tz=UTC)).total_seconds() / 86_400.0
        if age_days > 3.0:
            alerts.append({"symbol": sym, "level": "stale_data", "age_days": round(age_days, 1)})
        w_target = max(0.0, min(1.0, strategy.weight(completed)))
        prev_w = prev_weights.get(sym, 0.0)
        decision_close = completed[-1]["close"]
        exec_price = candles[-1]["close"]  # latest print (may be today's live candle)
        move = exec_price / decision_close - 1.0
        sleeve_ret = w_target * move - cost_rate * abs(w_target - prev_w)
        sleeve_rets.append(sleeve_ret)
        slip_bps = None
        if abs(w_target - prev_w) > 0.01:
            slip_bps = move * 10_000.0  # decision-to-execution gap on a turnover day
        # a data gap that delayed a weight change = missed fill(s)
        last_gap_days = (completed[-1]["open_time"] - completed[-2]["open_time"]) / 86_400_000
        if last_gap_days > 1.5 and abs(w_target - prev_w) > 0.05:
            missed_fills.append({"symbol": sym, "delayed_days": int(last_gap_days - 1)})
        asset_details[sym] = {
            "weight": w_target,
            "price": exec_price,
            "decision_close": decision_close,
            "sleeve_ret": sleeve_ret,
            "slippage_bps": slip_bps,
        }

    port_gross = sum(sleeve_rets) / max(n_assets, 1)
    overlay_w = trailing_overlay_weight(port_history, config["overlay"]["target_vol"])
    prev_overlay = prev_entries[-1].get("overlay_weight", 1.0) if prev_entries else 1.0
    port_ret = overlay_w * port_gross - 0.0015 * abs(overlay_w - prev_overlay)
    entry = {
        "ts": now.isoformat(),
        "date": today,
        "assets": asset_details,
        "port_ret": port_ret,
        "overlay_weight": overlay_w,
        "outages": outages,
        "missed_fills": missed_fills,
    }
    if alerts:
        entry["alerts"] = alerts
    append_log(entry, log_path)
    return {"status": "logged", "entry": entry}


def checkpoints_due(freeze_date: date, as_of: date) -> list[dict]:
    elapsed = (as_of - freeze_date).days
    return [
        {"label": label, "days": days, "elapsed": elapsed, "due": elapsed >= days}
        for label, days in CHECKPOINTS
    ]


def monthly_returns(entries: list[dict]) -> dict[str, float]:
    """Compound forward-log returns per calendar month (negatives kept)."""
    months: dict[str, list[float]] = {}
    for e in entries:
        months.setdefault(e["date"][:7], []).append(e["port_ret"])
    out = {}
    for m, rets in sorted(months.items()):
        eq = 1.0
        for r in rets:
            eq *= 1.0 + r
        out[m] = eq - 1.0
    return out


def slippage_stats(entries: list[dict]) -> dict:
    obs = [d["slippage_bps"] for e in entries for d in e.get("assets", {}).values() if isinstance(d, dict) and d.get("slippage_bps") is not None]
    if not obs:
        return {"count": 0, "mean_abs_bps": None}
    return {"count": len(obs), "mean_abs_bps": sum(abs(x) for x in obs) / len(obs)}


def outage_stats(entries: list[dict]) -> dict:
    return {
        "outage_days": sum(1 for e in entries if e.get("outages")),
        "outage_events": sum(len(e.get("outages", [])) for e in entries),
        "missed_fills": sum(len(e.get("missed_fills", [])) for e in entries),
    }


def alert_stats(entries: list[dict]) -> dict:
    """Data-staleness (and other) alerts across the forward log."""
    alerts = [a for e in entries for a in e.get("alerts", [])]
    return {
        "total": len(alerts),
        "by_level": {lvl: sum(1 for a in alerts if a.get("level") == lvl) for lvl in sorted({a.get("level", "?") for a in alerts})},
        "symbols": sorted({a["symbol"] for a in alerts}),
    }
