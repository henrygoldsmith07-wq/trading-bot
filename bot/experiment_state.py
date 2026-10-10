"""Experiment state: one deterministic answer to "where is this experiment?".

WHY THIS EXISTS
    Users used to infer status from several files: does a canonical record
    exist, is the freeze current, how many forward rows are stamped, is the
    source dirty. Each surface re-derived that story and they could disagree.
    This module derives ONE state from the evidence itself, deterministically,
    so the CLI, API, dashboard, and README all print the same word.

STATES (in lifecycle order)
    RESEARCH_ONLY        no canonical evidence record yet
    CANONICAL_READY      canonical evidence exists, nothing frozen
    READY_TO_FREEZE      a freeze exists but its source seal is broken
    FORWARD_PRELIMINARY  a valid current freeze with fewer than 30 clean days
    FORWARD_MEANINGFUL   >= 30 clean forward days from the CURRENT experiment
    SUPERSEDED           the freeze is intact but its methodology is no longer
                         current; its tape is evidence of what produced it, not
                         of the code now running
    INVALIDATED          provenance or the experiment seal is broken; no claim
                         may stand on this evidence

The state is derived, never assigned. A superseded experiment NEVER
contributes forward days to the current experiment's count, and a state
transition is a fact about the evidence, not a mutable label.

Paper trading only.
"""
from __future__ import annotations

from typing import Any

RESEARCH_ONLY = "RESEARCH_ONLY"
CANONICAL_READY = "CANONICAL_READY"
READY_TO_FREEZE = "READY_TO_FREEZE"
FORWARD_PRELIMINARY = "FORWARD_PRELIMINARY"
FORWARD_MEANINGFUL = "FORWARD_MEANINGFUL"
SUPERSEDED = "SUPERSEDED"
INVALIDATED = "INVALIDATED"

ALL_STATES = (
    RESEARCH_ONLY,
    CANONICAL_READY,
    READY_TO_FREEZE,
    FORWARD_PRELIMINARY,
    FORWARD_MEANINGFUL,
    SUPERSEDED,
    INVALIDATED,
)

# The threshold at which forward evidence stops being a sanity check and
# becomes evidence worth grading. Defined ONCE; the verdict layer and the
# checkpoint table both read it from here.
MEANINGFUL_FORWARD_DAYS = 30

# How long the prospective tape may go without a new day before the run that
# is supposed to produce one is treated as broken rather than merely quiet.
#
# A frozen experiment accrues one day per calendar day, so a gap this large is
# not a holiday: it means the scheduled run stopped advancing evidence. The
# slack exists so weekends, exchange closures, feed outages and a backlogged
# runner queue are never mistaken for a dead experiment.
#
# This bounds a real failure that went undetected here for two weeks: the tape
# last advanced on 2026-09-22 while every scheduled run reported success, and
# nothing in the evidence said the number had stopped moving.
MAX_FORWARD_TAPE_GAP_DAYS = 7

# Forward checkpoints (clean days) and what each one licenses.
CHECKPOINTS = (
    (10, "sanity check only"),
    (30, "preliminary evidence"),
    (60, "early evaluation"),
    (90, "meaningful first review"),
    (180, "stronger assessment"),
    (365, "full annual regime sample"),
)


class ExperimentStateError(RuntimeError):
    """Evidence is internally contradictory and no state can be derived."""


def next_checkpoint(clean_days: int) -> dict[str, Any] | None:
    """The next checkpoint, how many clean days remain, and what it licenses."""
    for days, meaning in CHECKPOINTS:
        if clean_days < days:
            return {
                "clean_days_required": days,
                "clean_days_remaining": days - clean_days,
                "meaning": meaning,
                "label": f"{days} clean forward days",
            }
    return None


def checkpoint_table(clean_days: int) -> list[dict[str, Any]]:
    """Every checkpoint with its status against the current clean-day count."""
    out = []
    for days, meaning in CHECKPOINTS:
        out.append({
            "clean_days_required": days,
            "label": f"{days} clean forward days",
            "meaning": meaning,
            "reached": clean_days >= days,
        })
    return out


def derive_state(
    *,
    has_canonical_record: bool,
    has_freeze: bool,
    freeze_intact: bool,
    methodologically_current: bool,
    clean_forward_days: int,
) -> str:
    """Derive the experiment state from evidence facts.

    Deterministic and total: every combination of inputs maps to exactly one
    state, with the priority order INVALIDATED > SUPERSEDED > lifecycle order.
    Fail closed: when the caller cannot establish integrity, the evidence is
    INVALIDATED rather than merely unverified.
    """
    if clean_forward_days < 0:
        raise ExperimentStateError(f"clean_forward_days cannot be negative (got {clean_forward_days})")

    if has_freeze and not freeze_intact:
        return INVALIDATED
    if has_freeze and not methodologically_current:
        return SUPERSEDED
    if not has_canonical_record:
        return RESEARCH_ONLY
    if not has_freeze:
        return CANONICAL_READY
    # A freeze whose seal verifies but that has no forward rows yet sits at the
    # very first rung of forward evidence, not at CANONICAL_READY: the
    # experiment is live, just unobserved.
    if clean_forward_days >= MEANINGFUL_FORWARD_DAYS:
        return FORWARD_MEANINGFUL
    return FORWARD_PRELIMINARY


def state_explanation(state: str, clean_forward_days: int) -> str:
    """One plain sentence a dashboard or README can render verbatim."""
    if state == RESEARCH_ONLY:
        return "No canonical evidence record exists yet, so nothing is being claimed."
    if state == CANONICAL_READY:
        return "Canonical research evidence exists but no experiment has been frozen."
    if state == READY_TO_FREEZE:
        return "A freeze exists but its source seal does not verify; it cannot be traded or graded."
    if state == SUPERSEDED:
        return (
            "The frozen experiment is intact but its methodology is no longer current; "
            "its tape is evidence of the implementation that produced it, not of the code now running."
        )
    if state == INVALIDATED:
        return "Provenance or the experiment seal is broken; no claim may stand on this evidence."
    if state == FORWARD_MEANINGFUL:
        return (
            f"{clean_forward_days} clean forward days from the current experiment "
            f"(>= {MEANINGFUL_FORWARD_DAYS}); prospective evidence is now the headline."
        )
    return (
        f"{clean_forward_days} clean forward day(s) from the current experiment; "
        f"below the {MEANINGFUL_FORWARD_DAYS}-day threshold, this is a sanity check, not validation."
    )
