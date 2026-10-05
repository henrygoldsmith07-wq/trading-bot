"""How much searching produced the rule under test — one place, one number.

WHY THIS EXISTS
    The repository used to carry several different counts of "how wide was the
    strategy search": the candidate pool length, the number of distinct
    strategies walk-forward selection actually picked, the research-ledger
    experiment total, and the clustering estimate of effective independent
    trials. Different surfaces quoted different ones, and a DSR computed at
    N=1 while the program had searched dozens of experiments was once the
    headline. That number was arithmetically true and evidentially empty.

    SearchAccounting is the single record of what was searched, and the only
    authority for the trial count a multiple-testing correction may use. The
    canonical run record and the statistical-validation pipeline both consume
    it; neither recomputes anything.

DISCIPLINE
    * every count is EXPLICIT input; nothing here reads files or guesses.
    * a rule produced by selection over a pool is never "N=1" merely because
      the final rule happens to be fixed; `prespecified` must be affirmatively
      established by the caller (frozen before any data was seen), and
      `n_trials_reasoning()` records the method used.
    * when the trial count cannot be established, the honest output is
      "DSR unavailable" plus a reason, never a weakened correction.

Paper trading only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Why the effective count is what it is. Stable strings, reported verbatim.
METHOD_MAX_OF_SOURCES = "max_of_recorded_sources"
METHOD_PRE_SPECIFIED = "pre_specified_single_trial"
METHOD_UNESTABLISHED = "unestablished"

SOURCE_CANDIDATE_POOL = "candidate_pool"
SOURCE_RESEARCH_LEDGER = "research_ledger"
SOURCE_OVERLAY_SELECTION = "portfolio_overlay_selection"
SOURCE_WALK_FORWARD_SELECTION = "walk_forward_selection"
SOURCE_COMPARATOR_ONLY = "comparator_only"


@dataclass(frozen=True)
class TrialSource:
    """One stream of selection that consumed degrees of freedom."""

    type: str
    count: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "count": self.count}
        if self.note:
            d["note"] = self.note
        return d


@dataclass(frozen=True)
class SearchAccounting:
    """The complete search record for one experiment.

    Counts are what the program DID, not what it can afford to admit:
    `candidate_count_generated` >= `candidate_count_evaluated`, and rejected
    research-ledger entries count toward the search because the search was
    real even when the idea failed.
    """

    candidate_pool_version: str
    candidate_count_generated: int
    candidate_count_evaluated: int
    walk_forward_selections: int
    research_ledger_experiments: int
    overlay_selections: int
    prespecified: bool
    dsr_available: bool
    dsr_unavailable_reason: str | None
    sources: tuple[TrialSource, ...]

    def __post_init__(self) -> None:
        if self.candidate_count_generated < 0 or self.candidate_count_evaluated < 0:
            raise ValueError("candidate counts must be non-negative")
        if self.candidate_count_evaluated > self.candidate_count_generated:
            raise ValueError(
                f"evaluated ({self.candidate_count_evaluated}) cannot exceed "
                f"generated ({self.candidate_count_generated})"
            )
        if self.walk_forward_selections < 0 or self.research_ledger_experiments < 0 or self.overlay_selections < 0:
            raise ValueError("selection counts must be non-negative")
        if self.prespecified and self.effective_trials() not in (None, 1):
            raise ValueError("a pre-specified rule cannot carry a search-derived trial count")

    def effective_trials(self) -> int | None:
        """The trial count a DSR correction may use, or None if unestablished.

        The honest count is the LARGEST search that actually happened, not the
        smallest one that is convenient. Pre-specified rules are the sole
        exception: their count is one because nothing was searched, and that is
        an affirmative claim recorded in `prespecified`.
        """
        if self.prespecified:
            return 1
        counts = [s.count for s in self.sources if s.type != SOURCE_COMPARATOR_ONLY]
        counts = [c for c in counts if c > 0]
        return max(counts) if counts else None

    def n_trials_reasoning(self) -> dict[str, Any]:
        """The structured reasoning behind `effective_trials()`.

        This object is what the evidence record publishes in place of a bare
        integer, so a reader can see WHO says the search was this wide and why.
        """
        trials = self.effective_trials()
        return {
            "effective_trials": trials,
            "sources": [s.to_dict() for s in self.sources],
            "method": (
                METHOD_PRE_SPECIFIED if self.prespecified
                else (METHOD_MAX_OF_SOURCES if trials is not None else METHOD_UNESTABLISHED)
            ),
            "prespecified": self.prespecified,
            "dsr_available": self.dsr_available and trials is not None and trials >= 1,
            "dsr_unavailable_reason": self.dsr_unavailable_reason,
            "candidate_pool_version": self.candidate_pool_version,
            "candidate_count_generated": self.candidate_count_generated,
            "candidate_count_evaluated": self.candidate_count_evaluated,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.n_trials_reasoning() | {
            "walk_forward_selections": self.walk_forward_selections,
            "research_ledger_experiments": self.research_ledger_experiments,
            "overlay_selections": self.overlay_selections,
        } | {"breadth": self.breadth_disclosure()}

    def breadth_disclosure(self) -> dict[str, Any]:
        """How the recorded sources COMPOUND, not just which one is largest.

        WHY THIS EXISTS
            `effective_trials()` takes the MAX of the recorded sources, on the
            stated principle that the honest count is the largest search that
            actually happened. That principle is right about which single
            number to correct against, but it is silent about something a
            reader of a DSR deserves to see: these sources are not competing
            estimates of one quantity, they are *dimensions of the same search*.

            A per-fold, per-asset walk-forward selection re-runs the candidate
            pool on each of F folds for each of A assets. The pool (85) and the
            number of picks (F*A) therefore do not compete — they multiply. A
            cross-sectional Sharpe chosen from 85 candidates, then re-chosen
            78 times across folds and sleeves, and then compared across several
            overlay rules, is a far wider search than any one of those factors
            alone, and the DSR benchmark scales with the width.

        WHAT IT DOES AND DOES NOT DO
            This is a disclosure, not a recomputation. It changes no DSR, moves
            no threshold and alters no published number, because the exact
            effective trial count is a judgement about correlation structure
            (overlapping folds, near-duplicate families) that cannot be settled
            by arithmetic — the repo already reports an `effective_trials`
            estimate from clustering for exactly that reason. What it does is
            stop the single `max` from looking like the whole search when a
            reader can see the factors laid out.
        """
        pool = max(self.candidate_count_generated, self.candidate_count_evaluated)
        dimensions = {
            "candidate_pool": pool,
            "walk_forward_selections": self.walk_forward_selections,
            "research_ledger_experiments": self.research_ledger_experiments,
            "overlay_selections": self.overlay_selections,
        }
        active = {k: v for k, v in dimensions.items() if v > 0}
        # The naive product is the most conservative reading of "all of these
        # compounded". It is deliberately NOT used as the DSR trial count; it
        # is published so the reader sees how much of the search a single
        # max-of-sources number is leaving implicit.
        product = 1
        for v in active.values():
            product *= v
        return {
            "dimensions": dimensions,
            "active_dimensions": sorted(active),
            "naive_compounded_trials": product if active else 0,
            "effective_trials_used": self.effective_trials(),
            "note": (
                "effective_trials_used is the max of these dimensions and is what a "
                "deflation corrects against. The dimensions compound rather than "
                "compete, so naive_compounded_trials is a disclosed upper reading, "
                "not the applied correction — the applied count is a judgement about "
                "correlation between folds and families, and the repo publishes its "
                "clustering-based effective-trial estimate for that."
            ),
        }


def build_search_accounting(
    *,
    candidate_pool_version: str,
    candidate_count_generated: int,
    candidate_count_evaluated: int,
    walk_forward_selections: int = 0,
    research_ledger_experiments: int = 0,
    overlay_selections: int = 0,
    prespecified: bool = False,
    dsr_unavailable_reason: str | None = None,
) -> SearchAccounting:
    """Assemble the accounting from explicit counts.

    `dsr_available` is derived, never asserted: a correction is computable only
    when a trial count exists AND someone searched. A pre-specified rule still
    gets its DSR (which equals its PSR at N=1); a selected rule with no
    establishable search history gets `dsr_available=False` and a reason.
    """
    sources: list[TrialSource] = []
    if candidate_count_generated > 0:
        sources.append(TrialSource(
            SOURCE_CANDIDATE_POOL,
            candidate_count_generated,
            f"pool {candidate_pool_version or 'unknown'}; {candidate_count_evaluated} evaluated",
        ))
    if research_ledger_experiments > 0:
        sources.append(TrialSource(
            SOURCE_RESEARCH_LEDGER,
            research_ledger_experiments,
            "every recorded search experiment, accepted or rejected",
        ))
    if walk_forward_selections > 0:
        sources.append(TrialSource(
            SOURCE_WALK_FORWARD_SELECTION,
            walk_forward_selections,
            "fold-by-fold strategy selections performed during validation",
        ))
    if overlay_selections > 0:
        sources.append(TrialSource(
            SOURCE_OVERLAY_SELECTION,
            overlay_selections,
            "portfolio overlays chosen during research",
        ))

    accounting = SearchAccounting(
        candidate_pool_version=candidate_pool_version,
        candidate_count_generated=candidate_count_generated,
        candidate_count_evaluated=candidate_count_evaluated,
        walk_forward_selections=walk_forward_selections,
        research_ledger_experiments=research_ledger_experiments,
        overlay_selections=overlay_selections,
        prespecified=prespecified,
        dsr_available=False,
        dsr_unavailable_reason=None,
        sources=tuple(sources),
    )

    trials = accounting.effective_trials()
    reason: str | None = dsr_unavailable_reason
    available = False
    if prespecified:
        available = True
        reason = None
    elif trials is None:
        reason = reason or (
            "no search history establishes a trial count; a trial-count-1 DSR would "
            "apply no correction at all and would not be evidence"
        )
    elif trials <= 1:
        reason = reason or (
            "the recorded search establishes a single trial only; reporting a "
            "selection-corrected DSR would overstate the correction applied"
        )
    else:
        available = True
        reason = None

    return SearchAccounting(
        candidate_pool_version=candidate_pool_version,
        candidate_count_generated=candidate_count_generated,
        candidate_count_evaluated=candidate_count_evaluated,
        walk_forward_selections=walk_forward_selections,
        research_ledger_experiments=research_ledger_experiments,
        overlay_selections=overlay_selections,
        prespecified=prespecified,
        dsr_available=available,
        dsr_unavailable_reason=reason,
        sources=tuple(sources),
    )
