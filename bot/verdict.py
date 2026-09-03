"""Strategy verdict: how much evidence actually supports this system?

The product of this repo is not "a bot with a backtest" — it is an honest
answer to "how much evidence supports this strategy?" This module grades
five independent dimensions and combines them into one verdict:

  1. HISTORICAL EVIDENCE      PSR/DSR of the headline configuration in the
                              canonical record — CAPPED by selection-bias
                              risk (a DSR computed at trial-count=1 while
                              the program searched 29+ experiments is not
                              strong evidence).
  2. WALK-FORWARD ROBUSTNESS  share of assets with positive OOS Sharpe in
                              the canonical record + fold count.
  3. SELECTION-BIAS RISK      pool size vs research-ledger search total;
                              HIGH dominates until a ledger-informed DSR
                              clears the bar.
  4. COST ROBUSTNESS          predicted-vs-observed friction error from the
                              accumulated cost tape.
  5. PROSPECTIVE FORWARD      days of frozen forward evidence, gated by
                              code verification and seal integrity.

Grades are ordered: Insufficient < Weak < Moderate < Strong.
A sixth implicit state, COMPROMISED, overrides everything when the seal is
broken or parameters changed after freezing.

OVERALL mapping (documented, deterministic):
  - any dimension COMPROMISED              -> "invalidated"
  - forward < Preliminary                  -> "promising, not validated"
     (unless historical/robustness Weak    -> "not established")
  - all core dims >= Moderate and forward >= Meaningful
                                           -> "validated (provisional)"
  - otherwise                              -> "partially supported"

Every grade ships with its numeric inputs so the verdict is auditable, not
oracular.
"""
from __future__ import annotations

GRADES = ("Insufficient", "Weak", "Moderate", "Strong")


def _at_least(grade: str, floor: str) -> bool:
    return GRADES.index(grade) >= GRADES.index(floor)


def _days_phrase(n: int) -> str:
    """'1 trading day' / '0 trading days'.

    This string is rendered verbatim in the dashboard hero, so it has to read
    like English rather than like a format string.
    """
    return f"{n} trading day" + ("" if n == 1 else "s")


# ---------------------------------------------------------------------------
# dimension graders — each returns {grade, inputs:{...}, reason:str}
# ---------------------------------------------------------------------------

def grade_historical(rule_stats: list[dict], headline_rule_substring: str, selection_risk_grade: str) -> dict:
    headline = next((r for r in rule_stats if headline_rule_substring.lower() in r["name"].lower()), None)
    if headline is None:
        return {"grade": "Insufficient", "inputs": {}, "reason": "headline rule absent from canonical record"}
    dsr, psr = float(headline["dsr"]), float(headline["psr"])
    if dsr >= 0.95 and psr >= 0.99:
        base = "Strong"
    elif dsr >= 0.90 or psr >= 0.95:
        base = "Moderate"
    elif dsr >= 0.80:
        base = "Weak"
    else:
        base = "Insufficient"
    # The cap: a trial-count-1 DSR inside a heavily-searched program is NOT
    # strong evidence, however good the number looks.
    grade = base
    reason = f"PSR {psr:.3f} / DSR {dsr:.3f} (recorded at trial-count=1)"
    if base == "Strong" and selection_risk_grade == "High":
        grade = "Moderate"
        reason += " — capped to Moderate because the search-program correction is unresolved"
    return {
        "grade": grade,
        "inputs": {"rule": headline["name"], "psr": psr, "dsr": dsr,
                   "cagr": headline.get("cagr"), "max_drawdown": headline.get("max_drawdown")},
        "reason": reason,
    }


def grade_robustness(per_asset: list[dict], n_folds: int | None) -> dict:
    if not per_asset:
        return {"grade": "Insufficient", "inputs": {}, "reason": "canonical record has no per-asset results"}
    n = len(per_asset)
    positive = sum(1 for a in per_asset if float(a.get("sharpe", 0)) > 0)
    share = positive / n
    inputs = {"assets": n, "positive_sharpe_assets": positive, "share_positive": round(share, 3),
              "folds": n_folds}
    if share >= 0.9 and (n_folds or 0) >= 4:
        grade = "Strong"
    elif share >= 0.65:
        grade = "Moderate"
    elif share >= 0.5:
        grade = "Weak"          # coin-flip: no demonstrated selection skill
    else:
        grade = "Insufficient"  # negative across most assets
    if (n_folds or 0) < 3 and grade == "Strong":
        grade = "Moderate"
    return {
        "grade": grade,
        "inputs": inputs,
        "reason": f"{positive}/{n} assets positive out-of-sample across {n_folds} folds",
    }


def grade_selection_bias(
    pool_size: int,
    ledger_search_n: int | None = None,
    ledger_informed_dsr: float | None = None,
) -> dict:
    if pool_size <= 1 and not ledger_search_n:
        return {"grade": "Low", "inputs": {}, "reason": "single pre-declared strategy, nothing searched"}
    inputs = {"candidate_pool": pool_size, "ledger_search_experiments": ledger_search_n or 0}
    if ledger_informed_dsr is not None and ledger_informed_dsr >= 0.95:
        return {"grade": "Moderate",
                "inputs": {**inputs, "ledger_informed_dsr": ledger_informed_dsr},
                "reason": "ledger-informed DSR clears 0.95 despite wide search"}
    if (ledger_search_n or 0) >= 25 or pool_size >= 50:
        return {"grade": "High", "inputs": inputs,
                "reason": f"wide search ({pool_size} candidates, {ledger_search_n} experiments) "
                          "without a search-corrected DSR clearing the bar"}
    return {"grade": "Moderate", "inputs": inputs,
            "reason": "moderate search breadth; search-corrected evidence pending"}


def grade_costs(n_turnover_events: int, mean_error_bp: float | None, sufficient: bool) -> dict:
    inputs = {"turnover_events": n_turnover_events, "mean_error_bp": mean_error_bp}
    if n_turnover_events == 0:
        return {"grade": "Insufficient", "inputs": inputs, "reason": "no paper turnover observed yet"}
    if not sufficient:
        return {"grade": "Insufficient", "inputs": inputs,
                "reason": f"only {n_turnover_events} events (<30) — V2 recalibration deferred"}
    err = abs(mean_error_bp) if mean_error_bp is not None else None
    if err is None:
        return {"grade": "Insufficient", "inputs": inputs, "reason": "no measurable drift recorded"}
    if err <= 5:
        grade = "Strong"
    elif err <= 15:
        grade = "Moderate"
    else:
        grade = "Weak"
    return {"grade": grade, "inputs": inputs,
            "reason": f"model-vs-tape error {mean_error_bp:+.2f} bp over {n_turnover_events} events"}


def grade_forward(
    days_recorded: int,
    code_verified: bool = True,
    parameter_changes: int = 0,
    outage_days: int = 0,
) -> dict:
    """Grade prospective evidence. Days are TRADING days actually logged."""
    if not code_verified:
        return {"grade": "COMPROMISED", "inputs": {"code_verified": False},
                "reason": "code identity verification failed — forward evidence void"}
    if parameter_changes != 0:
        return {"grade": "COMPROMISED", "inputs": {"parameter_changes": parameter_changes},
                "reason": "parameters changed after freeze — forward evidence void"}
    inputs = {"days_recorded": days_recorded, "outage_days": outage_days}
    outage_ratio = outage_days / days_recorded if days_recorded else 0.0
    if days_recorded < 30:
        grade = "Insufficient"
    elif days_recorded < 90:
        grade = "Weak"
    elif days_recorded < 180:
        grade = "Moderate"
    else:
        grade = "Strong"
    reason = f"{_days_phrase(days_recorded)} recorded"
    if outage_ratio > 0.2 and grade in ("Moderate", "Strong"):
        grade = "Weak"  # downgrade one level: feed reliability question
        reason += f"; {outage_days} outage days exceeds 20%"
    return {"grade": grade, "inputs": inputs, "reason": reason}


# ---------------------------------------------------------------------------
# combination
# ---------------------------------------------------------------------------

_OVERALL_MATRIX_NOTE = (
    "overall: invalidated if compromised; 'promising, not validated' while "
    "forward evidence is below Preliminary; 'validated (provisional)' only when "
    "historical/robustness/costs are all >= Moderate AND forward >= Strong "
    "(>=180 trading days); anything between is 'partially supported'"
)


def combine(hist: str, robust: str, selection: str, costs: str, forward: str) -> tuple[str, str]:
    if "COMPROMISED" in (hist, robust, selection, costs, forward):
        return "INVALIDATED", _OVERALL_MATRIX_NOTE
    core_ok = all(_at_least(g, "Moderate") for g in (hist, robust, costs))
    fwd_idx = GRADES.index(forward) if forward in GRADES else -1
    if fwd_idx < GRADES.index("Weak"):
        if _at_least(hist, "Moderate") and _at_least(robust, "Moderate"):
            return "promising, not validated", _OVERALL_MATRIX_NOTE
        return "not established", _OVERALL_MATRIX_NOTE
    if core_ok and _at_least(forward, "Strong") and selection != "High":
        return "validated (provisional)", _OVERALL_MATRIX_NOTE
    return "partially supported", _OVERALL_MATRIX_NOTE


def build_verdict(
    *,
    canonical_rule_stats: list[dict],
    canonical_per_asset: list[dict],
    canonical_n_folds: int | None,
    pool_size: int,
    ledger_search_n: int | None,
    cost_report: dict | None,
    forward: dict | None,
    headline_rule_substring: str = "banded 5% rebalance",
) -> dict:
    sel = grade_selection_bias(pool_size, ledger_search_n, ledger_informed_dsr=None)
    hist = grade_historical(canonical_rule_stats, headline_rule_substring, sel["grade"])
    robust = grade_robustness(canonical_per_asset, canonical_n_folds)

    if cost_report:
        c = grade_costs(cost_report.get("n_turnover_events", 0),
                        cost_report.get("error_bp"),
                        cost_report.get("sufficient", False))
    else:
        c = {"grade": "Insufficient", "inputs": {}, "reason": "no cost tape"}

    if forward and forward.get("available") and forward.get("started"):
        n_days = int(forward.get("n_days_recorded", 0))
        f = grade_forward(
            days_recorded=n_days,
            code_verified=bool(forward.get("code_verified")),
            parameter_changes=int(forward.get("parameter_changes", 0)),
            outage_days=int(forward.get("data_outages", 0)),
        )
        f_out = {"grade": f["grade"], "inputs": {**f["inputs"], "days_recorded": n_days},
                 "reason": f["reason"],
                 "label": f"{f['grade']} — {_days_phrase(n_days)}"}
    else:
        reason = (forward or {}).get("reason") or "no freeze/forward log"
        f_out = {"grade": "Insufficient", "inputs": {}, "reason": reason,
                 "label": "Insufficient — 0 trading days"}

    overall, note = combine(hist["grade"], robust["grade"], sel["grade"], c["grade"], f_out["grade"])

    return {
        "verdict": {
            "historical_evidence": hist["grade"],
            "walk_forward_robustness": robust["grade"],
            "selection_bias_risk": sel["grade"],
            "cost_robustness": c["grade"],
            "prospective_forward_evidence": f_out["label"],
            "overall": overall,
        },
        "details": {"historical": hist, "robustness": robust,
                    "selection_bias": sel, "costs": c, "forward": f_out},
        "note": note,
    }


def format_verdict(v: dict) -> str:
    vd = v["verdict"]
    L = [
        "=" * 62,
        "STRATEGY VERDICT",
        "=" * 62,
        f"Historical evidence         : {vd['historical_evidence']}",
        f"Walk-forward robustness     : {vd['walk_forward_robustness']}",
        f"Selection-bias risk         : {vd['selection_bias_risk']}",
        f"Cost robustness             : {vd['cost_robustness']}",
        f"Prospective forward evidence: {vd['prospective_forward_evidence']}",
        "-" * 62,
        f"OVERALL: {vd['overall']}",
        "-" * 62,
    ]
    d = v["details"]
    for key in ("historical", "robustness", "selection_bias", "costs", "forward"):
        L.append(f"[{key}] {d[key]['reason']}")
        ins = d[key].get("inputs") or {}
        if ins:
            pretty = ", ".join(f"{k}={val}" for k, val in ins.items())
            L.append(f"    inputs: {pretty}")
    L.append(v["note"])
    return "\n".join(L)
