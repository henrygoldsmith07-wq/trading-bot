"""Cost calibration: measure trading frictions from prospective observations.

V1 ships flat frictions (fee + spread + slippage bps) with optional richer
models. This module closes the loop:

1. Every paper turnover records an OBSERVATION: signal timestamp, decision
   close, execution print, best bid/ask/mid when available, spread, the
   decision->execution drift (the paper slippage proxy), turnover, realized
   volatility and average daily volume context.
2. `calibrate()` compares the model's PREDICTED cost per unit turnover
   against the mean observed proxy and reports the error in bp.
3. It proposes V2 parameters (recalibrated friction rates) WITHOUT touching
   the frozen V1 configuration — recalibration is a suggestion written to
   cost_calibration.json for human review.

Honesty notes baked into the report:
- The proxy is implementation shortfall against the DECISION price. It
  contains true trading costs PLUS any short-horizon drift of the signal;
  with hundreds of decisions whose alpha averages ~0 over one bar, the mean
  approximates total friction. This assumption is printed, not hidden.
- Bid/ask may be unavailable (feed lag); entries record nulls and the
  calibration falls back to the decision-price proxy only.
"""
from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

OBSERVATIONS_LOG = Path(
    os.environ.get("COST_OBSERVATIONS_LOG", "cost_observations.jsonl")
)
CALIBRATION_FILE = Path("cost_calibration.json")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_observation(
    *,
    symbol: str,
    side: str,                      # "BUY" | "SELL" | "FLAT"
    target_weight: float,
    previous_weight: float,
    decision_close: float,
    exec_price: float,
    mark_price: float,
    bid: float | None,
    ask: float | None,
    predicted_cost_bps: float,
    realized_vol_annual: float | None,
    adv30_usd: float | None,
    day_volume_base: float | None,
    signal_ts: str | None = None,
    ts: str | None = None,
) -> dict:
    """One observation per turnover event. All bp fields are signed where
    direction matters; magnitudes are used by the calibrator."""
    turnover = abs(target_weight - previous_weight)
    mid: float | None = None
    quoted_spread_bps: float | None = None
    fill_vs_mid_bps: float | None = None
    if bid is not None and ask is not None and bid > 0 < ask:
        mid = (bid + ask) / 2
        quoted_spread_bps = (ask - bid) / mid * 10_000.0
        fill_vs_mid_bps = (exec_price / mid - 1.0) * 10_000.0
    drift_bps: float | None = (
        (exec_price / decision_close - 1.0) * 10_000.0 if decision_close > 0 else None
    )
    side_sign = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
    # implementation-shortfall proxy: adverse move of the fill vs the decision
    observed_proxy_bps: float | None = side_sign * drift_bps if (side_sign and drift_bps is not None) else None

    return {
        "ts": ts or _now(),
        "signal_ts": signal_ts,
        "symbol": symbol,
        "side": side,
        "target_weight": round(target_weight, 6),
        "previous_weight": round(previous_weight, 6),
        "turnover": round(turnover, 6),
        "decision_close": decision_close,
        "exec_price": exec_price,
        "mark_price": mark_price,
        "bid": bid,
        "ask": ask,
        "mid": _r(mid),
        "quoted_spread_bps": _r(quoted_spread_bps),
        "fill_vs_mid_bps": _r(fill_vs_mid_bps),
        "decision_to_exec_drift_bps": _r(drift_bps),
        "observed_cost_proxy_bps": _r(observed_proxy_bps),
        "predicted_cost_bps": _r(predicted_cost_bps),
        "realized_vol_annual": _r(realized_vol_annual),
        "adv30_usd": adv30_usd,
        "day_volume_base": day_volume_base,
    }


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x, digits)


def append_observation(observation: dict, path: str | Path = OBSERVATIONS_LOG) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(observation, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_observations(path: str | Path = OBSERVATIONS_LOG) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    observations: list[dict],
    v1_frictions: dict,
    min_observations: int = 30,
) -> dict:
    """Predicted-vs-observed friction accounting and a V2 proposal.

    v1_frictions: {"fee","spread_bps","slippage_bps"} — the frozen baseline.
    Only turnover events with a usable proxy are measured; the count is
    reported so sample size can never be faked.
    """
    fee_bps = float(v1_frictions.get("fee", 0.0)) * 10_000.0
    spread_bps = float(v1_frictions.get("spread_bps", 0.0))
    slip_bps = float(v1_frictions.get("slippage_bps", 0.0))
    predicted_model_bps = fee_bps + spread_bps + slip_bps

    rows = [
        o for o in observations
        if o.get("turnover", 0) > 1e-9
        and o.get("observed_cost_proxy_bps") is not None
        and math.isfinite(o["observed_cost_proxy_bps"])
    ]
    n = len(rows)
    insufficient = n < min_observations

    def _mean(field: str) -> float | None:
        vals = [o[field] for o in rows if o.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    mean_pred = predicted_model_bps  # constant under the flat V1 model
    mean_obs = _mean("observed_cost_proxy_bps")
    mean_drift = _mean("decision_to_exec_drift_bps")
    mean_abs_drift = (
        sum(abs(o["decision_to_exec_drift_bps"]) for o in rows
            if o.get("decision_to_exec_drift_bps") is not None) / n
        if n else None
    )
    mean_spread = _mean("quoted_spread_bps")

    error_bp = (mean_obs - mean_pred) if mean_obs is not None else None
    ratio = (mean_obs / mean_pred) if (mean_obs is not None and mean_pred > 0) else None

    # V2 proposal: keep the contractual fee; recalibrate the estimated
    # frictions to what the paper tape actually showed.
    v2 = {
        "fee": v1_frictions.get("fee", 0.0),
        "effective_spread_plus_slippage_bps": _r(max(mean_obs, 0.0)) if mean_obs is not None else None,
        "scale_vs_v1": _r(ratio),
        "status": "insufficient_data" if insufficient else "proposed",
    }

    return {
        "n_turnover_events": n,
        "min_observations": min_observations,
        "sufficient": not insufficient,
        "predicted_cost_bps": _r(mean_pred),
        "observed_cost_proxy_bps": _r(mean_obs),
        "error_bp": _r(error_bp),
        "scale_vs_v1": _r(ratio),
        "mean_decision_to_exec_drift_bps": _r(mean_drift),
        "mean_abs_decision_to_exec_drift_bps": _r(mean_abs_drift),
        "mean_quoted_spread_bps": _r(mean_spread),
        "v2_proposal": v2,
        "assumptions": [
            "proxy = implementation shortfall vs the DECISION close",
            "valid when average one-bar signal drift is ~0 across many trades",
            "bid/ask recorded opportunistically; absent quotes fall back to decision-price proxy",
        ],
    }


def write_calibration(report: dict, path: str | Path = CALIBRATION_FILE) -> Path:
    payload = {"generated_at": _now(), **report}
    Path(path).write_text(json.dumps(payload, indent=2))
    return Path(path)


def load_calibration(path: str | Path = CALIBRATION_FILE) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def format_report(report: dict) -> str:
    lines = []
    n = report["n_turnover_events"]
    status = "OK" if report["sufficient"] else f"INSUFFICIENT (<{report['min_observations']})"
    lines.append(f"Cost calibration from {n} paper turnover events [{status}]")
    if n == 0:
        lines.append("  no turnover events recorded yet")
        return "\n".join(lines)
    pred = report["predicted_cost_bps"]
    obs = report["observed_cost_proxy_bps"]
    err = report["error_bp"]
    lines.append(f"  predicted trading cost : {pred:.2f} bp (V1 flat model)")
    lines.append(f"  observed slippage proxy: {obs:.2f} bp" if obs is not None else "  observed slippage proxy: n/a")
    lines.append(f"  error                  : {err:+.2f} bp" if err is not None else "  error                  : n/a")
    if report.get("scale_vs_v1") is not None:
        lines.append(f"  scale vs V1            : x{report['scale_vs_v1']:.2f}")
    md = report.get("mean_abs_decision_to_exec_drift_bps")
    sp = report.get("mean_quoted_spread_bps")
    if md is not None:
        lines.append(f"  mean |drift|           : {md:.2f} bp (includes signal noise)")
    if sp is not None:
        lines.append(f"  mean quoted spread     : {sp:.2f} bp (when quotes captured)")
    v2 = report["v2_proposal"]
    if report["sufficient"] and v2.get("effective_spread_plus_slippage_bps") is not None:
        lines.append(f"  V2 proposal            : fee unchanged; effective spread+slip "
                     f"{v2['effective_spread_plus_slippage_bps']:.2f} bp "
                     f"(x{v2['scale_vs_v1']:.2f} of V1)")
    else:
        lines.append(f"  V2 proposal            : deferred ({v2['status']})")
    lines.append("  assumptions: " + "; ".join(report["assumptions"]))
    return "\n".join(lines)
