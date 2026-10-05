"""Strategy verdict: how much evidence actually supports this system?

The product of this repo is not "a bot with a backtest" — it is an honest
answer to "how much evidence supports this strategy?" This module grades
six independent dimensions and combines them into one verdict:

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
   6. PRE-REGISTRATION        whether a sealed plan fixed the criterion
                               (metric, direction, threshold, minimum
                               evidence, arms, stopping rule) BEFORE the
                               result was seen. This dimension is upstream of
                               the rest: a rule whose success criterion was
                               picked after the numbers are known is not a
                               test, however strong the tape behind it is.
                               Missing registration CAPS the overall verdict;
                               it never cancels the evidence itself.

Grades are ordered: Insufficient < Weak < Moderate < Strong.
A sixth implicit state, COMPROMISED, overrides everything when the seal is
broken or parameters changed after freezing.
(A "sixth dimension" is pre-registration; the COMPROMISED state is orthogonal
to all six.)

OVERALL mapping (documented, deterministic):
  - any dimension COMPROMISED              -> "invalidated"
  - forward < Preliminary                  -> "promising, not validated"
     (unless historical/robustness Weak    -> "not established")
  - all core dims >= Moderate and forward >= Meaningful
                                           -> "validated (provisional)"
  - pre-registration below Weak            -> "promising, not validated"
  - otherwise                              -> "partially supported"

Every grade ships with its numeric inputs so the verdict is auditable, not
oracular.
"""
from __future__ import annotations

from .canonical_identity import PRIMARY_RULE_STAT_NAME

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
    psr = float(headline["psr"])
    # A record may legitimately carry no DSR: when the search that produced the
    # rule cannot be counted, the deflated statistic is unavailable rather than
    # reported as a number that corrects for nothing.
    raw_dsr = headline.get("dsr")
    dsr: float | None = None if raw_dsr is None else float(raw_dsr)
    if dsr is not None and not bool(headline.get("dsr_available", True)):
        dsr = None  # the record marks it unavailable; do not surface the number
    n_trials = headline.get("dsr_n_trials")

    if dsr is None:
        # Fail closed: PSR alone, uncorrected for a search we know happened,
        # cannot support Strong evidence.
        base = "Moderate" if psr >= 0.99 else ("Weak" if psr >= 0.95 else "Insufficient")
        grade = base if base != "Moderate" else "Weak"
        reason = (
            f"PSR {psr:.3f}; DSR unavailable — "
            f"{headline.get('dsr_unavailable_reason') or 'no selection history recorded'}"
        )
    else:
        if dsr >= 0.95 and psr >= 0.99:
            base = "Strong"
        elif dsr >= 0.90 or psr >= 0.95:
            base = "Moderate"
        elif dsr >= 0.80:
            base = "Weak"
        else:
            base = "Insufficient"
        grade = base
        reason = f"PSR {psr:.3f} / DSR {dsr:.3f} (deflated against N={n_trials} recorded trials)"
        if base == "Strong" and selection_risk_grade == "High":
            grade = "Moderate"
            reason += " — capped to Moderate because the search program is still wide"
    return {
        "grade": grade,
        "inputs": {"rule": headline["name"], "psr": psr, "dsr": dsr,
                   "dsr_available": dsr is not None, "dsr_n_trials": n_trials,
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
    evidence_verified: bool = True,
    methodologically_current: bool = True,
    experiment_version: str | None = None,
    superseded_by: str | None = None,
) -> dict:
    """Grade prospective evidence. Days are TRADING days actually logged.

    METHODOLOGICAL CURRENCY IS A SEPARATE GATE FROM INTEGRITY.

    A tape can be perfectly intact (config hash valid, seal present) and still
    not be evidence of the code now running: it may have been produced under an
    execution-accounting model the repo has since corrected. That is not
    tampering, so it is deliberately NOT graded COMPROMISED — calling it
    compromised would print "INVALIDATED" for a system that is merely
    un-revalidated, which is its own kind of dishonesty.

    Instead the days are NOT COUNTED. A superseded experiment is honest
    evidence of the implementation that produced it and of nothing else, so
    the honest grade for the running system is "Insufficient — zero current
    forward days", with the reason stating exactly why. Understating
    prospective evidence is the safe direction to be wrong in.

    `methodologically_current` fails CLOSED: if the caller cannot establish
    currency, the days are not counted.
    """
    if not evidence_verified:
        return {"grade": "COMPROMISED", "inputs": {"evidence_verified": False},
                "reason": "forward evidence integrity verification failed — evidence void"}
    if not code_verified:
        return {"grade": "COMPROMISED", "inputs": {"code_verified": False},
                "reason": "code identity verification failed — forward evidence void"}
    if parameter_changes != 0:
        return {"grade": "COMPROMISED", "inputs": {"parameter_changes": parameter_changes},
                "reason": "parameters changed after freeze — forward evidence void"}
    if not methodologically_current:
        version = experiment_version or "unknown"
        target = superseded_by or "a newer methodology"
        return {
            "grade": "Insufficient",
            "inputs": {
                "methodologically_current": False,
                "experiment_version": version,
                "superseded_by": target,
                "days_recorded_not_counted": days_recorded,
            },
            "reason": (
                f"experiment {version} is SUPERSEDED by {target}; its {days_recorded} forward "
                f"days are valid evidence of the implementation that produced them, "
                f"NOT of the code now running, and are not counted here"
            ),
        }
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
    "(>=180 trading days) AND a valid pre-registration covers the claim; "
    "anything between is 'partially supported'"
)


def grade_registration(registration: dict | None) -> dict:
    """Was the decision rule fixed BEFORE the result?

    WHY THIS IS A DIMENSION AND NOT A FOOTNOTE
        Every other dimension grades how strong the evidence is. This one
        grades something upstream of the evidence: whether anyone committed,
        in advance, to the criterion the evidence is now being measured
        against. A rule chosen after the numbers are known — the flattering
        metric, the arm that happened to win, the threshold that happens to be
        cleared — produces evidence that every other dimension can rate Strong
        and that still is not a test. The DSR cannot catch this, because these
        are trials of the decision procedure, not trials of the strategy.

    THE CAP, AND WHY IT IS ONLY A CAP
        Missing registration never *cancels* evidence; it withholds the
        strongest word. A strategy with 200 clean forward days and no plan is
        still genuinely interesting, and saying "INVALIDATED" would be its own
        kind of dishonesty. So an absent or unreadable plan caps the overall
        verdict at 'promising, not validated' — the same ceiling a
        below-Preliminary forward tape gets, for the same underlying reason:
        the thing being measured has not yet been pinned down.

    FAIL-CLOSED
        No section, a drifted plan (edited after sealing), or a plan that was
        only read AFTER the fact all read the same way: no valid plan covers
        the claim.
    """
    if not isinstance(registration, dict) or not registration:
        return {
            "grade": "Weak",
            "inputs": {"registered": False},
            "reason": "no pre-registered plan covers this claim; the criterion was never fixed in advance",
        }

    n_valid = int(registration.get("n_valid", 0) or 0)
    n_drifted = int(registration.get("n_drifted", 0) or 0)
    if n_valid <= 0:
        reason = (
            f"no valid pre-registered plan covers this claim ({n_drifted} plan(s) failed integrity); "
            "the criterion was never fixed in advance"
        )
        return {"grade": "Weak", "inputs": {"registered": False, "n_drifted": n_drifted}, "reason": reason}

    plan = registration.get("primary_plan") or {}
    inputs = {
        "registered": True,
        "registration_id": plan.get("registration_id"),
        "seal": plan.get("seal"),
        "criterion": plan.get("criterion"),
        "min_evidence": plan.get("min_evidence"),
        "n_arms": plan.get("n_arms"),
        "declared_trials": plan.get("declared_trials"),
        "read": bool(registration.get("any_read")),
    }

    if not registration.get("any_read"):
        # The plan exists but nobody has graded a result against it yet. That
        # is a legitimate state — a forward test accrues days before it is due
        # to be read — and it must not be treated as a failure. It simply does
        # not yet license the strongest verdict.
        return {
            "grade": "Moderate",
            "inputs": inputs,
            "reason": (
                f"a valid pre-registration covers this claim ({plan.get('criterion')} on >= "
                f"{plan.get('min_evidence')} observation(s), {plan.get('n_arms')} arm(s)); "
                "it has not been read yet"
            ),
        }

    return {
        "grade": "Strong",
        "inputs": inputs,
        "reason": (
            f"read against a sealed plan: {plan.get('criterion')} on >= {plan.get('min_evidence')} "
            f"observation(s), {plan.get('n_arms')} declared arm(s) "
            f"({'family-wise corrected' if int(plan.get('n_arms') or 0) > 1 else 'single arm'})"
        ),
    }


def combine(
    hist: str,
    robust: str,
    selection: str,
    costs: str,
    forward: str,
    registration: str = "Moderate",
) -> tuple[str, str]:
    if "COMPROMISED" in (hist, robust, selection, costs, forward):
        return "INVALIDATED", _OVERALL_MATRIX_NOTE
    core_ok = all(_at_least(g, "Moderate") for g in (hist, robust, costs))
    fwd_idx = GRADES.index(forward) if forward in GRADES else -1
    if fwd_idx < GRADES.index("Weak"):
        if _at_least(hist, "Moderate") and _at_least(robust, "Moderate"):
            return "promising, not validated", _OVERALL_MATRIX_NOTE
        return "not established", _OVERALL_MATRIX_NOTE
    # A claim whose decision rule was never fixed in advance cannot reach the
    # strongest word, however strong the tape behind it is. This is a ceiling,
    # not a refutation.
    #
    # The floor is Moderate, not Weak: `grade_registration` returns exactly
    # "Weak" for an absent, unreadable or drifted plan, so testing against Weak
    # would let the strongest verdict through precisely when there is no plan.
    if not _at_least(registration, "Moderate"):
        return "promising, not validated", _OVERALL_MATRIX_NOTE
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
    registration: dict | None = None,
    headline_rule_substring: str = PRIMARY_RULE_STAT_NAME,
) -> dict:
    sel = grade_selection_bias(pool_size, ledger_search_n, ledger_informed_dsr=None)
    hist = grade_historical(canonical_rule_stats, headline_rule_substring, sel["grade"])
    robust = grade_robustness(canonical_per_asset, canonical_n_folds)
    reg_grade = grade_registration(registration)

    if cost_report and cost_report.get("integrity_verified") is False:
        c = {
            "grade": "COMPROMISED",
            "inputs": {"integrity_verified": False},
            "reason": cost_report.get("integrity_reason") or "cost evidence integrity verification failed",
        }
    elif cost_report:
        c = grade_costs(cost_report.get("n_turnover_events", 0),
                        cost_report.get("error_bp"),
                        cost_report.get("sufficient", False))
    else:
        c = {"grade": "Insufficient", "inputs": {}, "reason": "no cost tape"}

    if forward and forward.get("available") and forward.get("evidence_verified") is False:
        f = grade_forward(
            days_recorded=0,
            code_verified=bool(forward.get("code_verified", True)),
            evidence_verified=False,
            parameter_changes=int(forward.get("parameter_changes", 0)),
            outage_days=0,
        )
        f_out = {
            "grade": f["grade"],
            "inputs": {**f["inputs"], "days_recorded": 0},
            "reason": forward.get("evidence_reason") or f["reason"],
            "label": f"{f['grade']} — 0 trading days",
        }
    elif forward and forward.get("available") and forward.get("started"):
        n_days = int(forward.get("n_days_recorded", 0))
        # Methodological currency fails CLOSED: if the payload does not carry
        # the field at all, we cannot establish that the tape speaks for the
        # running code, so its days are not counted. The honest response to
        # "we cannot tell" is not to award prospective evidence.
        currency_known = "methodologically_current" in forward
        f = grade_forward(
            days_recorded=n_days,
            code_verified=bool(forward.get("code_verified")),
            evidence_verified=bool(forward.get("evidence_verified", True)),
            parameter_changes=int(forward.get("parameter_changes", 0)),
            outage_days=int(forward.get("data_outages", 0)),
            methodologically_current=currency_known and bool(forward.get("methodologically_current")),
            experiment_version=forward.get("experiment_version"),
            superseded_by=forward.get("superseded_by"),
        )
        f_out = {"grade": f["grade"], "inputs": {**f["inputs"], "days_recorded": n_days},
                 "reason": f["reason"],
                 # The label is rendered as the hero, so it must not advertise
                 # days the grade just declined to count.
                 "label": f"{f['grade']} — {_days_phrase(0 if not currency_known or not forward.get('methodologically_current') else n_days)}"}
    else:
        reason = (forward or {}).get("reason") or "no freeze/forward log"
        f_out = {"grade": "Insufficient", "inputs": {}, "reason": reason,
                 "label": "Insufficient — 0 trading days"}

    overall, note = combine(
        hist["grade"], robust["grade"], sel["grade"], str(c["grade"]), f_out["grade"], reg_grade["grade"]
    )

    return {
        "verdict": {
            "historical_evidence": hist["grade"],
            "walk_forward_robustness": robust["grade"],
            "selection_bias_risk": sel["grade"],
            "cost_robustness": c["grade"],
            "prospective_forward_evidence": f_out["label"],
            "pre_registration": reg_grade["grade"],
            "overall": overall,
        },
        "details": {"historical": hist, "robustness": robust,
                    "selection_bias": sel, "costs": c, "forward": f_out,
                    "registration": reg_grade},
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
        f"Pre-registration            : {vd.get('pre_registration', 'Weak')}",
        "-" * 62,
        f"OVERALL: {vd['overall']}",
        "-" * 62,
    ]
    d = v["details"]
    for key in ("historical", "robustness", "selection_bias", "costs", "forward", "registration"):
        L.append(f"[{key}] {d[key]['reason']}")
        ins = d[key].get("inputs") or {}
        if ins:
            pretty = ", ".join(f"{k}={val}" for k, val in ins.items())
            L.append(f"    inputs: {pretty}")
    L.append(v["note"])
    return "\n".join(L)
