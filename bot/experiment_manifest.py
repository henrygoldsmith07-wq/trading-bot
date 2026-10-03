"""Experiment identity: one explicit manifest every piece of evidence names.

WHY THIS EXISTS
    Experiment metadata used to be spread across freeze.json, the canonical run
    record, the forward log's row stamps, and helper constants in three modules.
    Each surface reconstructed "which experiment is this?" from whichever fields
    happened to be nearby, and two pieces of evidence could silently be combined
    even when different code, different config, or different accounting produced
    them. That is exactly the class of mistake this repository exists to prevent.

    One frozen dataclass now carries the identity. Every forward observation,
    canonical record, and evidence artifact names a manifest fingerprint. When
    two fingerprints differ, `assert_same_experiment` refuses the combination
    instead of averaging across implementations.

The manifest is EXPLICIT by construction: nothing here infers experiment
identity from unrelated fields. Legacy artifacts (a freeze predating the
manifest schema) are projected into a manifest with their unknowns stated as
unknown, never guessed.

Paper trading only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .experiments import CURRENT_ACCOUNTING_MODEL, infer_accounting_model

# The benchmark the canonical claim is graded against. One definition, here,
# because "which benchmark" must never be re-decided per surface.
BENCHMARK_ID = "spx"
BENCHMARK_LABEL = "S&P 500"

# Methodology identity: the accounting model plus the execution convention.
# Any change to either produces a NEW methodology version and therefore a new
# experiment; it can never silently extend the old one.
METHODOLOGY_VERSION = f"{CURRENT_ACCOUNTING_MODEL}+next_open"


class MixedExperimentError(RuntimeError):
    """Two artifacts describe different experiments and cannot be combined."""


@dataclass(frozen=True)
class ExperimentManifest:
    """The frozen identity of one paper-trading experiment.

    Every field that can change a return or an inference has exactly one owner:
    the freeze that created this manifest. `fingerprint()` is the value every
    other artifact stores, and the value `assert_same_experiment` compares.
    """

    experiment_id: str
    version: str
    primary_rule_id: str

    source_commit: str
    source_fingerprint: str
    config_fingerprint: str

    candidate_pool_version: str
    candidate_count_generated: int
    candidate_count_evaluated: int
    effective_trial_count: int | None

    accounting_model: str
    execution_model: str

    universe_method: str
    universe_snapshot_id: str | None

    benchmark_id: str

    frozen_at: str
    methodology_version: str

    forward_log_id: str | None

    # Anything a legacy artifact could not establish. Present in the dict so a
    # reader can see what the identity does NOT pin; never used to infer.
    unknowns: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unknowns"] = list(self.unknowns)
        d["fingerprint"] = self.fingerprint()
        return d

    def fingerprint(self) -> str:
        """sha256 over the canonical JSON of the identity fields.

        Deterministic across processes: keys sorted, no timestamps beyond the
        recorded ones, floats not involved. Two manifests agree iff every
        identity field agrees.
        """
        d = asdict(self)
        d["unknowns"] = list(self.unknowns)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def short_id(self) -> str:
        return f"{self.version}:{self.experiment_id[:12]}"


def assert_same_experiment(*manifests: ExperimentManifest | dict[str, Any] | None) -> str:
    """Refuse to combine evidence produced under different experiments.

    Accepts manifests or their dict forms (the dict must carry `fingerprint`,
    as produced by `to_dict`). Returns the shared fingerprint. Raises
    MixedExperimentError naming every disagreeing field, so the failure message
    tells the reader exactly what differs instead of "mismatch".
    """
    seen: list[tuple[str, dict[str, Any]]] = []
    for m in manifests:
        if m is None:
            continue
        if isinstance(m, ExperimentManifest):
            seen.append((m.fingerprint(), m.to_dict()))
        elif isinstance(m, dict):
            fp = m.get("fingerprint")
            if not fp:
                raise MixedExperimentError("experiment dict carries no fingerprint; rebuild it from a manifest")
            seen.append((str(fp), m))
        else:
            raise MixedExperimentError(f"cannot interpret experiment identity from {type(m).__name__}")
    if not seen:
        raise MixedExperimentError("no experiment identity supplied; evidence without a manifest cannot be combined")
    first_fp, first = seen[0]
    for fp, other in seen[1:]:
        if fp == first_fp:
            continue
        diffs = sorted(
            k for k in set(first) | set(other)
            if k != "fingerprint" and k != "unknowns" and first.get(k) != other.get(k)
        )
        raise MixedExperimentError(
            "refusing to combine evidence from different experiments "
            f"({first.get('experiment_id')} vs {other.get('experiment_id')}); "
            f"differing fields: {', '.join(diffs) or 'fingerprint only'}"
        )
    return first_fp


def manifest_from_freeze(
    freeze: dict[str, Any],
    *,
    primary_rule_id: str,
    candidate_pool_version: str,
    candidate_count_generated: int,
    candidate_count_evaluated: int,
    effective_trial_count: int | None,
    universe_snapshot_id: str | None = None,
    forward_log_id: str | None = None,
    methodology_version: str | None = None,
) -> ExperimentManifest:
    """Project a freeze manifest into the explicit experiment identity.

    A freeze predating this schema carries no manifest block, so the identity
    is built from what the freeze DOES seal (commit, config hash, code hash,
    accounting model) and the counts the caller establishes from the canonical
    record and the search-accounting layer. Fields that no artifact establishes
    are recorded in `unknowns` instead of defaulting to something plausible.
    """
    config_raw = freeze.get("config")
    config = config_raw if isinstance(config_raw, dict) else {}
    frictions_raw = config.get("frictions")
    frictions = frictions_raw if isinstance(frictions_raw, dict) else {}
    algorithm_raw = config.get("algorithm")
    algorithm = algorithm_raw if isinstance(algorithm_raw, dict) else {}

    unknown: list[str] = []
    commit = freeze.get("git_commit_at_freeze")
    if not commit:
        unknown.append("source_commit")
    code_sha = freeze.get("code_sha256")
    if not code_sha:
        unknown.append("source_fingerprint")
    config_sha = freeze.get("config_sha256")
    if not config_sha:
        unknown.append("config_fingerprint")
    frozen_at = freeze.get("frozen_at") or freeze.get("frozen_at_date") or ""
    if not frozen_at:
        unknown.append("frozen_at")

    accounting = infer_accounting_model(freeze)
    execution = str(frictions.get("execution") or "")
    if not execution:
        unknown.append("execution_model")

    pool_version = str(algorithm.get("candidate_pool_version") or candidate_pool_version or "")
    if not pool_version:
        unknown.append("candidate_pool_version")

    version = str(freeze.get("experiment_version") or "v1")
    frozen_date = str(freeze.get("frozen_at_date") or "")
    experiment_id = f"freeze/{frozen_date}" if frozen_date else f"freeze/{config_sha or 'unknown'}"

    if candidate_count_evaluated <= 0:
        unknown.append("candidate_count_evaluated")
    if effective_trial_count is None:
        unknown.append("effective_trial_count")

    return ExperimentManifest(
        experiment_id=experiment_id,
        version=version,
        primary_rule_id=primary_rule_id,
        source_commit=str(commit or ""),
        source_fingerprint=str(code_sha or ""),
        config_fingerprint=str(config_sha or ""),
        candidate_pool_version=pool_version,
        candidate_count_generated=int(candidate_count_generated),
        candidate_count_evaluated=int(candidate_count_evaluated),
        effective_trial_count=effective_trial_count,
        accounting_model=accounting,
        execution_model=execution or "unknown",
        universe_method=str(algorithm.get("universe_selection_rule") or "") or "unknown",
        universe_snapshot_id=universe_snapshot_id,
        benchmark_id=BENCHMARK_ID,
        frozen_at=str(frozen_at),
        methodology_version=methodology_version or METHODOLOGY_VERSION,
        forward_log_id=forward_log_id,
        unknowns=tuple(unknown),
    )


def manifest_block(manifest: ExperimentManifest) -> dict[str, Any]:
    """The `experiment` block of evidence.json, in one canonical shape."""
    d = manifest.to_dict()
    d["methodologically_current"] = manifest.accounting_model == CURRENT_ACCOUNTING_MODEL
    d["benchmark_label"] = BENCHMARK_LABEL
    return d
