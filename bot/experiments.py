"""Experiment versioning: which freeze generated which tape, and is it current?

A forward tape is only prospective evidence of a *specific implementation*.
The 2026-09-06 freeze (v1) generated its tape under an execution-accounting
model that was later found to UNDERSTATE returns across overnight gaps: the
session return was the sum of two simple legs rather than the exact evolution
of portfolio wealth. The corrected model is `exact-wealth-v2`.

That does not make the v1 tape worthless — it is real, immutable, honestly
generated evidence of the v1 implementation. But it is NOT prospective
validation OF THE CORRECTED IMPLEMENTATION, and presenting it as such would
be exactly the kind of evidence inflation this repo exists to prevent. So the
system must be able to answer, mechanically:

    which code/config generated this tape?    -> code_sha256 + config_sha256
    is that methodology still current?       -> accounting_model vs CURRENT
    has it been superseded?                   -> superseded_by
    which experiment is the verdict grading?  -> the current freeze

v1 is PRESERVED, never rewritten or deleted. Superseding is expressed as a
statement about the CURRENT methodology, never as an edit to history.
"""
from __future__ import annotations

# The execution-accounting model a tape was produced under.
ACCOUNTING_MODEL_ADDITIVE = "additive-legs-v1"
ACCOUNTING_MODEL_EXACT = "exact-wealth-v2"

# The model the CURRENT code implements. Anything else is, by construction,
# not methodologically current.
CURRENT_ACCOUNTING_MODEL = ACCOUNTING_MODEL_EXACT

SUPERSEDED_BY = {ACCOUNTING_MODEL_ADDITIVE: ACCOUNTING_MODEL_EXACT}

STATUS_CURRENT = "current"
STATUS_SUPERSEDED = "superseded"

# Human-readable statement of what changed, surfaced verbatim in reports.
SUPERSEDE_REASON = (
    "Session returns were computed as the additive sum of the overnight and "
    "intraday legs, which understates the return whenever a position is held "
    "through an overnight gap (it drops the cross term). The corrected model "
    "evolves portfolio wealth exactly, so close-to-close returns reproduce to "
    "machine precision. Numbers recorded under the old model are NOT "
    "comparable with numbers recorded under the new one."
)


def infer_accounting_model(manifest: dict) -> str:
    """Which accounting model produced this freeze's tape?

    A manifest predating the correction carries no field, so absence is itself
    the evidence: it was sealed by code that could only do the additive thing.
    New freezes stamp the field explicitly and are never guessed at.
    """
    stamped = manifest.get("accounting_model")
    if isinstance(stamped, str) and stamped:
        return stamped
    if manifest.get("experiment_version") in ("v2", "v3"):
        # A post-v2 experiment without an explicit model can only be current.
        return ACCOUNTING_MODEL_EXACT
    return ACCOUNTING_MODEL_ADDITIVE


def describe_freeze(manifest: dict) -> dict:
    """Full experiment-provenance record for a freeze manifest."""
    model = infer_accounting_model(manifest)
    current = model == CURRENT_ACCOUNTING_MODEL
    return {
        "experiment_version": manifest.get("experiment_version") or "v1",
        "accounting_model": model,
        "current_accounting_model": CURRENT_ACCOUNTING_MODEL,
        "methodologically_current": current,
        "status": STATUS_CURRENT if current else STATUS_SUPERSEDED,
        "superseded_by": None if current else SUPERSEDED_BY.get(model, CURRENT_ACCOUNTING_MODEL),
        "supersede_reason": None if current else SUPERSEDE_REASON,
        "frozen_at": manifest.get("frozen_at"),
        "git_commit_at_freeze": manifest.get("git_commit_at_freeze"),
        "code_sha256": manifest.get("code_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "freeze_id": f"freeze/{manifest.get('frozen_at_date')}" if manifest.get("frozen_at_date") else None,
    }


def current_freeze_status(manifest: dict) -> str:
    """One-line statement suitable for a report or dashboard caption."""
    d = describe_freeze(manifest)
    if d["methodologically_current"]:
        return (
            f"experiment {d['experiment_version']} (accounting model "
            f"{d['accounting_model']}) is methodologically current"
        )
    return (
        f"experiment {d['experiment_version']} (accounting model {d['accounting_model']}) "
        f"is SUPERSEDED by {d['superseded_by']}; its tape is valid evidence of the "
        f"implementation that produced it, NOT of the corrected implementation"
    )
