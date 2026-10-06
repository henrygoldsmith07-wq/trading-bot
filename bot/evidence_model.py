"""The single source of truth for every public claim this repository makes.

WHY THIS EXISTS
    The README, the dashboard, the CLI, the API and the generated reports each
    used to reconstruct "what does the evidence say" from whichever numbers
    were handy. A stale derived copy could therefore disagree with the
    canonical rule: the equal-weight comparator once carried the headline while
    the frozen strategy under test trailed the index. That class of bug is
    eliminated structurally here.

    One object — the canonical verdict object — is generated from the PRIMARY
    rule's metrics and the BENCHMARK's metrics and nothing else. Every surface
    renders from it: README headline, README tables, CLI output and exit code,
    API summary, dashboard, generated reports, and canonical reproduction
    output. There is no second formatting of a performance sentence anywhere
    in the product.

    Comparators are structurally incapable of touching the claim: the verdict
    builders never accept them. A comparator appears only in an explicitly
    labelled context block (`comparators`), and invariant tests assert that a
    comparator with spectacular returns cannot move the verdict text, the
    headline numbers, the dashboard status, the API verdict, or the exit code.

Paper trading only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .canonical_identity import (
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
)
from .experiment_manifest import BENCHMARK_ID, BENCHMARK_LABEL

# The exit code contract. One place. The CLI reads it from the verdict object.
EXIT_BEATS = 0
EXIT_TRAILS = 1


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, fixed separators, no NaN/Infinity.

    Two identical evidence objects serialize to identical bytes on any
    machine, which is what makes "two generated reports from identical
    evidence are byte-identical" testable rather than aspirational.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _f(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


def render_verdict_sentence(verdict: dict[str, Any]) -> str:
    """THE performance sentence. Rendered here and nowhere else.

    Every surface that states whether the strategy beats the benchmark must
    use this function on the canonical verdict object. Hand-written variants
    of this sentence are a bug even when they happen to be right.
    """
    m = verdict["metrics"]
    b = verdict["benchmark_metrics"]
    cmp_ = verdict["comparison"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    return (
        f"primary rule [{verdict['primary_rule_id']}] OOS CAGR "
        f"{'BEATS' if cmp_['beats_cagr'] else 'TRAILS'} {verdict['benchmark_label']} "
        f"({pct(m['cagr'])} vs {pct(b['cagr'])}); "
        f"Sharpe {'beats' if cmp_['beats_sharpe'] else 'trails'} "
        f"({num(m['sharpe'])} vs {num(b['sharpe'])}); "
        f"max drawdown {'better' if cmp_['beats_max_drawdown'] else 'worse'} "
        f"({pct(m['max_drawdown'])} vs {pct(b['max_drawdown'])})"
    )


def build_canonical_verdict(
    *,
    primary_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    generated_from_run: str,
    evidence_state: str | None = None,
    generated_at: str | None = None,
    primary_rule_id: str = PRIMARY_RULE_ID,
    benchmark_id: str = BENCHMARK_ID,
    benchmark_label: str = BENCHMARK_LABEL,
) -> dict[str, Any]:
    """Build the canonical verdict object from the primary rule and benchmark.

    NOTE THE SIGNATURE: there is no comparators parameter, by design. Nothing
    about any other strategy can reach this computation, so no downstream
    rendering can accidentally claim a comparator's result. Callers that have
    comparators attach them via `attach_comparators`, which touches only the
    labelled context block and never the verdict fields.
    """
    # The three claim metrics are REQUIRED; the wider set is carried when the
    # record has it so context tables render from the same blocks.
    required = ("cagr", "sharpe", "max_drawdown")
    optional = ("vol", "sortino", "calmar", "es95", "final")
    for name, block in (("primary", primary_metrics), ("benchmark", benchmark_metrics)):
        missing = [k for k in required if k not in block]
        if missing:
            raise ValueError(f"{name} metrics missing required fields: {missing}")

    m = {k: _f(primary_metrics.get(k)) for k in required}
    m.update({k: _f(primary_metrics.get(k)) for k in optional if primary_metrics.get(k) is not None})
    b = {k: _f(benchmark_metrics.get(k)) for k in required}
    b.update({k: _f(benchmark_metrics.get(k)) for k in optional if benchmark_metrics.get(k) is not None})
    if any(v is None for v in m.values()) or any(v is None for v in b.values()):
        raise ValueError("metrics must carry numeric cagr/sharpe/max_drawdown")

    # The required trio is non-None after the check above; bind it as floats so
    # the comparisons below are arithmetic on numbers, not on optionals.
    p_cagr, p_sharpe, p_mdd = m["cagr"], m["sharpe"], m["max_drawdown"]
    b_cagr, b_sharpe, b_mdd = b["cagr"], b["sharpe"], b["max_drawdown"]
    assert p_cagr is not None and p_sharpe is not None and p_mdd is not None
    assert b_cagr is not None and b_sharpe is not None and b_mdd is not None
    comparison = {
        "beats_cagr": p_cagr > b_cagr,
        "beats_sharpe": p_sharpe > b_sharpe,
        "beats_max_drawdown": p_mdd > b_mdd,
        "cagr_delta": p_cagr - b_cagr,
        "sharpe_delta": p_sharpe - b_sharpe,
        "max_drawdown_delta": p_mdd - b_mdd,
    }

    verdict: dict[str, Any] = {
        "primary_rule_id": primary_rule_id,
        "primary_rule_description": PRIMARY_RULE_DESCRIPTION,
        "benchmark_id": benchmark_id,
        "benchmark_label": benchmark_label,
        "metrics": m,
        "benchmark_metrics": b,
        "comparison": comparison,
        "verdict_code": "beats_on_cagr_and_sharpe" if (comparison["beats_cagr"] and comparison["beats_sharpe"]) else "does_not_beat",
        "evidence_state": evidence_state,
        "generated_from_run": generated_from_run,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
    }
    # The sentence is a FUNCTION of the object, never a stored claim.
    verdict["verdict"] = render_verdict_sentence(verdict)
    # Exit code follows the primary rule, computed in exactly one place.
    verdict["exit_code"] = EXIT_BEATS if verdict["verdict_code"] == "beats_on_cagr_and_sharpe" else EXIT_TRAILS
    return verdict


def attach_comparators(verdict: dict[str, Any], comparators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attach comparator context without touching any verdict field.

    Returns a copy. The returned object's claim fields are byte-identical to
    the input's; comparators live only under `comparators`, each entry labelled
    as context. This is the ONLY sanctioned way comparators enter the product.
    """
    out = json.loads(canonical_json(verdict))
    out["comparators"] = {
        name: {
            "metrics": {k: _f(block.get(k)) for k in ("cagr", "sharpe", "max_drawdown", "vol", "sortino", "calmar", "es95", "final")},
            "label": "comparator — context only, never the claim",
        }
        for name, block in sorted(comparators.items())
    }
    return out


def normalize_rule_stats(rules: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Force every statistical row onto the current trial-accounting contract.

    The canonical-v2 record predates trial accounting: its rows carry a bare
    `dsr` with no `dsr_n_trials` and no `dsr_available` flag. That stored value
    was computed at trial-count-1, which applies NO multiple-testing
    correction, so it is arithmetically a DSR and evidentially not one.
    Rendering it as a number would be the exact dishonesty this repository
    exists to prevent.

    So a row that cannot evidence its trial count gets `dsr` withheld — the
    number is replaced by None and a reason is attached. Every claim surface
    (verdict grade, README table, API) normalizes through HERE, so the
    treatment cannot differ between them.
    """
    out: list[dict[str, Any]] = []
    for row in rules or []:
        r = dict(row)
        has_accounting = "dsr_available" in r or "dsr_n_trials" in r
        available = bool(r.get("dsr_available", False)) if has_accounting else False
        if not has_accounting and r.get("dsr") is not None:
            r["dsr_unavailable_reason"] = (
                "record row predates trial accounting; the stored value corrected for "
                "nothing and is withheld"
            )
        if not available:
            r["dsr"] = None
            r["dsr_available"] = False
            r.setdefault(
                "dsr_unavailable_reason",
                "no selection history establishes a trial count for this row",
            )
        out.append(r)
    return out


def _portable_registration(registration: Any) -> Any:
    """Strip local read-state from the registration section for fingerprinting.

    A sealed plan is committed and portable; whether it has been READ on this
    particular checkout is not. The read ledger lives in a gitignored
    local file, so folding `any_read` / `n_reads` / `latest_read_state` into
    the fingerprint made the fingerprint a function of the machine: CI (no
    ledger) and a developer who has read a plan (ledger present) disagreed on
    the evidence identity, and the committed README block could not match both.
    Same class as `git_now` -- local state masquerading as evidence.

    What survives is the committed, content-addressed part: which plans exist,
    whether each still verifies against its own seal, and the criterion each
    one fixes. Those genuinely are properties of the evidence.
    """
    if not isinstance(registration, dict):
        return registration
    local_keys = ("read", "n_reads", "latest_read_state")
    out = {k: v for k, v in registration.items() if k not in ("any_read",)}
    if "primary_plan" in out:
        pp = out["primary_plan"]
        if isinstance(pp, dict):
            out["primary_plan"] = {k: v for k, v in pp.items() if k not in local_keys}
    plans = []
    for p in registration.get("plans") or []:
        if not isinstance(p, dict):
            plans.append(p)
            continue
        plans.append({k: v for k, v in p.items() if k not in local_keys})
    if "plans" in out:
        out["plans"] = plans
    return out


def evidence_fingerprint(doc: dict[str, Any]) -> str:
    """sha256 over the evidence document minus its non-evidential fields.

    Two builds from identical evidence share this fingerprint; any drift in
    the evidence itself changes it. Reports render deterministically from a
    fixed document, so identical fingerprints imply byte-identical reports.

    Three inputs are excluded because they describe the BUILD, not the evidence:

    * `generated_at` — wall-clock time;
    * `provenance.git_now` — the live checkout's HEAD commit and dirty-path
      list, which says where the build ran rather than what it found;
    * the registration section's LOCAL read-state (`any_read` and the
      per-plan `read`/`n_reads`/`latest_read_state`), which lives in a
      gitignored per-checkout ledger.

    Including any of these made the fingerprint a function of the machine:
    checking out at a different commit, with an unrelated file open, or after
    recording a plan read, produced a different "evidence" fingerprint. That
    broke the invariant this function exists to provide AND made the README
    canonical block unsatisfiable -- the block embeds the fingerprint, so
    committing it, or reading a plan locally, moved a value the block itself
    records.

    These fields stay IN the document; `verify-evidence`, the verdict and the
    verdict sentence all still read them. They are simply not part of the
    evidence's identity.
    """
    stable = {k: v for k, v in doc.items() if k != "generated_at"}
    provenance = stable.get("provenance")
    if isinstance(provenance, dict):
        stable["provenance"] = {k: v for k, v in provenance.items() if k != "git_now"}
    if "registration" in stable:
        stable["registration"] = _portable_registration(stable["registration"])
    # The verdict's pre_registration GRADE is a downstream consequence of
    # read-state: reading a refuted plan moves it Moderate -> Weak. That grade
    # is live process status, not evidence, so it is normalised out of the
    # fingerprint along with the ledger fields it is derived from. The grade
    # itself is untouched in the document; only the hashed copy drops it.
    verdict = stable.get("verdict")
    if isinstance(verdict, dict) and isinstance(verdict.get("grades"), dict):
        stable["verdict"] = {**verdict,
                             "grades": {k: v for k, v in verdict["grades"].items()
                                        if k != "pre_registration"}}
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def build_evidence_document(
    *,
    experiment: dict[str, Any],
    status: dict[str, Any],
    primary_strategy: dict[str, Any],
    benchmark: dict[str, Any],
    historical: dict[str, Any],
    walk_forward: dict[str, Any],
    selection_bias: dict[str, Any],
    forward: dict[str, Any],
    data_quality: dict[str, Any],
    execution_quality: dict[str, Any],
    provenance: dict[str, Any],
    verdict: dict[str, Any],
    comparators: dict[str, dict[str, Any]] | None = None,
    rule_stats: list[dict[str, Any]] | None = None,
    timeline: list[dict[str, Any]] | None = None,
    registration: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble evidence.json — the central user-facing evidence object.

    Every section is REQUIRED. A surface that needs one section consumes this
    object; no surface re-derives research meaning from raw tapes. The verdict
    section is the canonical verdict object built by `build_canonical_verdict`,
    so the claim in evidence.json and the claim on every screen are the same
    bytes.

    Deterministic given inputs: sections arrive complete, and ordering
    everywhere is sorted or explicitly sequenced.
    """
    if verdict.get("primary_rule_id") != experiment.get("primary_rule_id"):
        raise ValueError(
            "evidence cannot be assembled: the verdict grades "
            f"{verdict.get('primary_rule_id')!r} but the experiment is "
            f"{experiment.get('primary_rule_id')!r}"
        )
    doc: dict[str, Any] = {
        "schema": "evidence/1",
        "experiment": experiment,
        "status": status,
        "primary_strategy": primary_strategy,
        "benchmark": benchmark,
        "historical": historical,
        "walk_forward": walk_forward,
        "selection_bias": selection_bias,
        "forward": forward,
        "data_quality": data_quality,
        "execution_quality": execution_quality,
        "provenance": provenance,
        "verdict": verdict,
    }
    if comparators:
        doc["comparators"] = {
            name: {
                "metrics": {k: _f(block.get(k)) for k in ("cagr", "sharpe", "max_drawdown", "vol", "sortino", "calmar", "es95", "final")},
                "label": "comparator — context only, never the claim",
            }
            for name, block in sorted(comparators.items())
        }
    if rule_stats:
        doc["rule_stats"] = list(rule_stats)
    if timeline:
        doc["timeline"] = list(timeline)
    # Always present, and always explicit about being absent. An omitted
    # section would be indistinguishable from an older document, and "was this
    # decision rule fixed in advance?" must have a visible answer on every
    # surface rather than a missing key.
    doc["registration"] = registration or {"registered": False, "n_plans": 0, "n_valid": 0, "n_drifted": 0, "any_read": False, "plans": []}
    doc["generated_at"] = generated_at or datetime.now(UTC).isoformat()
    doc["evidence_fingerprint"] = evidence_fingerprint(doc)
    return doc


def headline_block(evidence: dict[str, Any]) -> dict[str, Any]:
    """The five facts the first screen must answer, derived from evidence.json.

    This is the contract `python -m bot evidence` and the dashboard hero both
    render: what is being tested, is there a current valid freeze, how many
    forward days exist, does it beat the benchmark, and the biggest caveat.
    Nothing here is formatted anywhere else.
    """
    verdict = evidence["verdict"]
    status = evidence["status"]
    forward = evidence.get("forward", {})
    clean_days = int(forward.get("clean_forward_days", 0) or 0)

    caveats: list[str] = []
    if not status.get("has_valid_freeze"):
        caveats.append("no current valid freeze — forward evidence is unavailable")
    if clean_days < 30:
        caveats.append(f"only {clean_days} clean forward day(s) — historical evidence dominates and is subject to selection")
    if not evidence.get("selection_bias", {}).get("dsr_available", True):
        caveats.append("no selection-corrected statistic is available for the searched rule")
    if not evidence.get("registration", {}).get("registered"):
        caveats.append("no pre-registered plan fixes the success criterion in advance")
    caveat = caveats[0] if caveats else (
        "forward sample is still small; do not treat a short run as validation"
    )

    return {
        "strategy": f"{verdict['primary_rule_id']} — {verdict.get('primary_rule_description', '')}",
        "benchmark_id": verdict["benchmark_id"],
        "beats_benchmark": bool(verdict["comparison"]["beats_cagr"] and verdict["comparison"]["beats_sharpe"]),
        "verdict_sentence": verdict["verdict"],
        "exit_code": verdict["exit_code"],
        "forward_clean_days": clean_days,
        "biggest_caveat": caveat,
    }
