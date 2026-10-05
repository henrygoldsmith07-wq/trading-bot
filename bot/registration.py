"""Pre-analysis plans: deciding what counts as a result BEFORE seeing the result.

WHY THIS EXISTS
    Every other control in this repository is *retrospective*. The research
    ledger records experiments after they ran. The freeze pins identity, not a
    decision rule. The deflated Sharpe corrects for the number of trials once
    the winner is known. All three are real, and all three are downstream of
    the same unprotected decision: **who decided, in advance, what would count
    as success, and which arm would be the claim?**

    That gap is the largest remaining source of false discovery here, because
    every freedom it protects is exercised unconsciously and invisibly:

    * picking the flattering metric once several are available
      (Sharpe, CAGR, Calmar and ES rank the portfolio rules differently);
    * picking the threshold once the number is known ("30 clean days", and not
      60, and not 12);
    * picking which arm is "primary" after seeing which arm wins — this repo
      grades five portfolio rules and one of them is named primary;
    * reading a forward tape repeatedly until one reading looks good, which is
      the multiple-comparisons problem applied to *time* rather than to arms;
    * stopping a losing forward run early and restarting, which converts one
      falsifiable prediction into an unbounded sequence of them.

    The Deflated Sharpe cannot see any of this: those are not trials of a
    strategy, they are trials of the *decision procedure*, and they happen
    after the trial count has been fixed. This module closes them by fixing the
    decision procedure first and refusing to grade against a plan that moved.

THE DISCIPLINE, STATED SO IT CAN BE CHECKED
    1. A registration declares the hypothesis, ONE primary metric, the
       direction, the numeric threshold, the minimum evidence required before
       the test may be read, every competing arm, the expected search breadth,
       and the stopping rule.
    2. It is content-addressed. Editing any field changes the seal, and a
       result may only be graded against a seal that still matches.
    3. A plan is read at most once. `read_result(..., event_log=...)` returns
       ALREADY_READ if a prior read exists, and `record_read` appends the read
       to a hash-chained log, so a second look is detectable rather than
       invisible and cannot be quietly performed.
    4. Absent sufficient evidence the answer is INCONCLUSIVE. It is never a
       pass, and it is never a failure — the repo's existing rule that
       `allPass=false` and `INSUFFICIENT DATA` are honest outputs applies here
       without exception.
    5. With more than one declared arm, the read applies Holm-Bonferroni, so
       the declared arm count actually costs something. Declaring five arms
       and then reading the best one is a search, and this module charges for it.

WHAT THIS DOES NOT CLAIM
    A registration cannot manufacture evidence. A well-formed plan read against
    a losing tape still returns NOT_CONFIRMED. Its value is entirely in making
    the reader's decision rule inspectable *in advance*, so that a later
    "well, we said we'd look at Sharpe" is a visible drift rather than a quiet
    one.

Paper trading only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REGISTRATION_DIR = "registrations"
DEFAULT_EVENT_LOG = "registration_events.jsonl"

# The read-out vocabulary. Deliberately closed: every state a reader can reach
# is one of these, so "is this validated?" has an exhaustive answer.
READ_CONFIRMED = "CONFIRMED"
READ_NOT_CONFIRMED = "NOT_CONFIRMED"
READ_INCONCLUSIVE = "INCONCLUSIVE"
READ_UNREGISTERED = "UNREGISTERED"
READ_DRIFTED = "DRIFTED"
READ_ALREADY_READ = "ALREADY_READ"

READ_STATES = (
    READ_CONFIRMED,
    READ_NOT_CONFIRMED,
    READ_INCONCLUSIVE,
    READ_UNREGISTERED,
    READ_DRIFTED,
    READ_ALREADY_READ,
)

# Directions a primary metric may be required to move in. Every threshold in
# this repo is two-sided in prose ("beat 0.95"); encoding it explicitly is what
# stops the direction being chosen after the fact.
DIRECTIONS = ("gt", "gte", "lt", "lte")


class RegistrationError(ValueError):
    """A registration is malformed, and the honest response is to refuse."""


class RegistrationDrift(RegistrationError):
    """The plan moved after it was sealed, so the result is a new study."""


# ---------------------------------------------------------------------------
# Canonical serialization + sealing
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, fixed separators, no NaN/Infinity.

    Mirrors `evidence_model.canonical_json` so that "byte-identical from
    identical inputs" holds for registrations too, and so a seal is stable
    across machines and Python versions.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _finite(value: float, field_name: str) -> float:
    """Reject non-finite numbers at the boundary.

    A threshold of NaN silently compares False against everything, which turns
    a plan into one that can never be confirmed and never be refuted. Refusing
    is the only honest option.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistrationError(f"{field_name} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise RegistrationError(f"{field_name} must be finite, got {value!r}")
    return out


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistrationError(f"{field_name} must be a non-empty string")
    return value.strip()


# ---------------------------------------------------------------------------
# Arms — competing hypotheses, declared together
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One competing hypothesis, declared before any of them is graded.

    Declaring arms is what makes "which one is the claim?" a question the plan
    answers rather than the tape. Exactly one arm carries `primary=True`, and
    it is fixed at registration time — choosing it later is the single most
    common way a multi-arm comparison turns into a single lucky result.
    """

    arm_id: str
    description: str
    primary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _require_text(self.arm_id, "arm_id"))
        object.__setattr__(self, "description", _require_text(self.description, "description"))

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "description": self.description, "primary": self.primary}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Arm:
        if not isinstance(raw, dict):
            raise RegistrationError("each arm must be a JSON object")
        unknown = set(raw) - {"arm_id", "description", "primary"}
        if unknown:
            raise RegistrationError(f"unknown arm fields {sorted(unknown)}; a typo must not become a default")
        # `__post_init__` runs `_require_text` on both strings, so a missing or
        # non-string value is rejected there; the explicit `is None` guards just
        # keep the types honest without weakening that validation.
        arm_id = raw.get("arm_id")
        description = raw.get("description")
        if arm_id is None or description is None:
            raise RegistrationError("each arm must carry both arm_id and description")
        return cls(
            arm_id=arm_id,
            description=description,
            primary=bool(raw.get("primary", False)),
        )


# ---------------------------------------------------------------------------
# The registration itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Registration:
    """A pre-analysis plan, sealed before any result is observed."""

    registration_id: str
    version: int
    title: str
    hypothesis: str
    primary_metric: str
    direction: str
    threshold: float
    min_evidence: int
    alpha: float
    arms: tuple[Arm, ...]
    stopping_rule: str
    declared_trials: int
    data_window: str
    created_at: str
    code_fingerprint: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "registration_id", _require_text(self.registration_id, "registration_id"))
        object.__setattr__(self, "title", _require_text(self.title, "title"))
        object.__setattr__(self, "hypothesis", _require_text(self.hypothesis, "hypothesis"))
        object.__setattr__(self, "primary_metric", _require_text(self.primary_metric, "primary_metric"))
        object.__setattr__(self, "stopping_rule", _require_text(self.stopping_rule, "stopping_rule"))
        object.__setattr__(self, "data_window", _require_text(self.data_window, "data_window"))
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))

        if self.direction not in DIRECTIONS:
            raise RegistrationError(f"direction must be one of {list(DIRECTIONS)}, got {self.direction!r}")
        object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))

        if isinstance(self.min_evidence, bool) or not isinstance(self.min_evidence, int) or self.min_evidence < 0:
            raise RegistrationError(f"min_evidence must be a non-negative integer, got {self.min_evidence!r}")

        alpha = _finite(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise RegistrationError(f"alpha must lie strictly between 0 and 1, got {self.alpha!r}")
        object.__setattr__(self, "alpha", alpha)

        if isinstance(self.declared_trials, bool) or not isinstance(self.declared_trials, int) or self.declared_trials < 1:
            raise RegistrationError(f"declared_trials must be a positive integer, got {self.declared_trials!r}")

        if not isinstance(self.arms, tuple) or not self.arms:
            raise RegistrationError("a registration must declare at least one arm")
        arm_ids = [a.arm_id for a in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise RegistrationError(f"duplicate arm ids {sorted(arm_ids)}")
        primaries = [a for a in self.arms if a.primary]
        if len(primaries) != 1:
            raise RegistrationError(
                f"exactly one arm must be primary, found {len(primaries)}; the primary arm is "
                "fixed here precisely so it cannot be chosen after seeing which arm wins"
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise RegistrationError("version must be a positive integer")

    # -- identity ---------------------------------------------------------

    @property
    def primary_arm(self) -> Arm:
        return next(a for a in self.arms if a.primary)

    def payload(self) -> dict[str, Any]:
        """The sealed content. `seal` is deliberately excluded.

        Everything that could change the conclusion is inside; nothing that is
        derived from it is, so the seal covers exactly the decision procedure.
        """
        return {
            "registration_id": self.registration_id,
            "version": self.version,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "threshold": self.threshold,
            "min_evidence": self.min_evidence,
            "alpha": self.alpha,
            "arms": [a.to_dict() for a in self.arms],
            "stopping_rule": self.stopping_rule,
            "declared_trials": self.declared_trials,
            "data_window": self.data_window,
            "created_at": self.created_at,
            "code_fingerprint": self.code_fingerprint,
            "notes": self.notes,
        }

    def seal(self) -> str:
        """Content address of the decision procedure (sha256, first 16 hex)."""
        blob = canonical_json(self.payload()).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def document(self) -> dict[str, Any]:
        """On-disk form: the payload plus its own seal."""
        return self.payload() | {"seal": self.seal()}

    # -- construction ----------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Registration:
        if not isinstance(raw, dict):
            raise RegistrationError("a registration must be a JSON object")
        known = {
            "registration_id", "version", "title", "hypothesis", "primary_metric",
            "direction", "threshold", "min_evidence", "alpha", "arms", "stopping_rule",
            "declared_trials", "data_window", "created_at", "code_fingerprint", "notes", "seal",
        }
        unknown = set(raw) - known
        if unknown:
            raise RegistrationError(f"unknown registration fields {sorted(unknown)}; a typo must not become a default")
        missing = known - set(raw) - {"seal", "code_fingerprint", "notes"}
        if missing:
            raise RegistrationError(f"missing registration fields {sorted(missing)}")

        reg = cls(
            registration_id=raw["registration_id"],
            version=raw["version"],
            title=raw["title"],
            hypothesis=raw["hypothesis"],
            primary_metric=raw["primary_metric"],
            direction=raw["direction"],
            threshold=raw["threshold"],
            min_evidence=raw["min_evidence"],
            alpha=raw["alpha"],
            arms=tuple(Arm.from_dict(a) for a in raw["arms"]),
            stopping_rule=raw["stopping_rule"],
            declared_trials=raw["declared_trials"],
            data_window=raw["data_window"],
            created_at=raw["created_at"],
            code_fingerprint=raw.get("code_fingerprint"),
            notes=raw.get("notes", ""),
        )
        declared = raw.get("seal")
        if declared is not None and declared != reg.seal():
            raise RegistrationDrift(
                f"registration {reg.registration_id!r} was edited after sealing: "
                f"declared seal {declared} but the content hashes to {reg.seal()}"
            )
        return reg


def create_registration(
    *,
    registration_id: str,
    title: str,
    hypothesis: str,
    primary_metric: str,
    threshold: float,
    direction: str = "gt",
    min_evidence: int = 1,
    alpha: float = 0.05,
    arms: Iterable[Arm | dict[str, Any]],
    stopping_rule: str,
    declared_trials: int = 1,
    data_window: str = "unspecified",
    code_fingerprint: str | None = None,
    notes: str = "",
    version: int = 1,
    created_at: str | None = None,
) -> Registration:
    """Assemble and seal a plan. `created_at` defaults to now, UTC."""
    stamp = created_at or datetime.now(UTC).isoformat(timespec="seconds")
    return Registration(
        registration_id=registration_id,
        version=version,
        title=title,
        hypothesis=hypothesis,
        primary_metric=primary_metric,
        direction=direction,
        threshold=threshold,
        min_evidence=min_evidence,
        alpha=alpha,
        arms=tuple(a if isinstance(a, Arm) else Arm.from_dict(a) for a in arms),
        stopping_rule=stopping_rule,
        declared_trials=declared_trials,
        data_window=data_window,
        created_at=stamp,
        code_fingerprint=code_fingerprint,
        notes=notes,
    )


def write_registration(reg: Registration, directory: str | Path = DEFAULT_REGISTRATION_DIR) -> Path:
    """Write `<registration_id>.json` atomically, refusing to overwrite.

    A plan is written once. Overwriting is the edit-after-sealing case that the
    seal exists to catch, and making it impossible at the filesystem removes a
    whole class of silent drift; changing a plan means registering a NEW
    version, which is visible as a new file and a new event.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{reg.registration_id}.json"
    if path.exists():
        raise RegistrationError(
            f"{path} already exists; a registered plan is immutable — register a new version instead"
        )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg.document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_registration(path: str | Path) -> Registration:
    """Read and integrity-check one plan. Any mismatch raises."""
    p = Path(path)
    if not p.exists():
        raise RegistrationError(f"no registration at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return Registration.from_dict(raw)


def registration_path(registration_id: str, directory: str | Path = DEFAULT_REGISTRATION_DIR) -> Path:
    return Path(directory) / f"{registration_id}.json"


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def direction_holds(value: float, direction: str, threshold: float) -> bool:
    """Does `value` clear `threshold` in the declared direction?"""
    if direction == "gt":
        return value > threshold
    if direction == "gte":
        return value >= threshold
    if direction == "lt":
        return value < threshold
    if direction == "lte":
        return value <= threshold
    raise RegistrationError(f"unknown direction {direction!r}")


def holm_bonferroni(p_values: dict[str, float], alpha: float) -> dict[str, Any]:
    """Holm-Bonferroni step-down FWER control across the declared arms.

    Declaring several arms and then reading the best-looking one is a multiple
    comparisons problem. Charging for it is what makes `declared_trials` in a
    plan a commitment rather than a decoration.

    Returns per-arm rejected flags plus the per-arm threshold actually applied,
    so the correction is visible rather than folded into a single verdict.
    """
    for name, p in p_values.items():
        value = _finite(p, f"p-value for arm {name!r}")
        if not 0.0 <= value <= 1.0:
            raise RegistrationError(f"p-value for arm {name!r} must lie in [0, 1], got {value}")
    if not p_values:
        raise RegistrationError("holm_bonferroni requires at least one p-value")

    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    rejected: dict[str, bool] = {}
    thresholds: dict[str, float] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        # step-down: the k-th smallest p is tested at alpha / (m - k)
        per_arm_alpha = alpha / (m - rank)
        thresholds[name] = per_arm_alpha
        if still_rejecting and p <= per_arm_alpha:
            rejected[name] = True
        else:
            # Holm stops at the first failure; nothing later may reject.
            still_rejecting = False
            rejected[name] = False
    return {"rejected": rejected, "per_arm_alpha": thresholds, "n_arms": m, "family_alpha": alpha}


# ---------------------------------------------------------------------------
# Read-out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """What was measured for one arm, when the plan is read.

    `evidence_count` is the sample size behind the number (forward days, folds,
    periods). `p_value` is optional: a plan may declare a hard threshold rather
    than a significance test, and in that case the threshold alone decides.
    """

    arm_id: str
    value: float
    evidence_count: int
    p_value: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise RegistrationError("observation arm_id must be a non-empty string")
        object.__setattr__(self, "value", _finite(self.value, "observed value"))
        if isinstance(self.evidence_count, bool) or not isinstance(self.evidence_count, int) or self.evidence_count < 0:
            raise RegistrationError(
                f"evidence_count for arm {self.observation_arm_id()!r} must be a non-negative integer"
            )
        if self.p_value is not None:
            p = _finite(self.p_value, "p_value")
            if not 0.0 <= p <= 1.0:
                raise RegistrationError(f"p_value must lie in [0, 1], got {self.p_value!r}")
            object.__setattr__(self, "p_value", p)

    def observation_arm_id(self) -> str:
        return self.arm_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "value": self.value,
            "evidence_count": self.evidence_count,
            "p_value": self.p_value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReadResult:
    """The answer to "does this result count?" — exhaustive, never invented."""

    state: str
    registration_id: str
    registration_seal: str
    reason: str
    primary_arm: str
    primary_value: float | None
    threshold: float
    direction: str
    evidence_count: int | None
    min_evidence: int
    per_arm: dict[str, Any] = field(default_factory=dict)

    @property
    def counts_as_evidence(self) -> bool:
        """Only CONFIRMED is evidence. INCONCLUSIVE and UNREGISTERED are not.

        Deliberately narrow: every other state is either a refutation or the
        absence of one, and treating any of them as support is precisely the
        error this module exists to prevent.
        """
        return self.state == READ_CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "counts_as_evidence": self.counts_as_evidence,
            "registration_id": self.registration_id,
            "registration_seal": self.registration_seal,
            "reason": self.reason,
            "primary_arm": self.primary_arm,
            "primary_value": self.primary_value,
            "threshold": self.threshold,
            "direction": self.direction,
            "evidence_count": self.evidence_count,
            "min_evidence": self.min_evidence,
            "per_arm": self.per_arm,
        }


def read_result(
    registration: Registration | None,
    observations: Sequence[Observation],
    *,
    expected_seal: str | None = None,
    event_log: str | Path | None = None,
) -> ReadResult:
    """Grade observations against a sealed plan. Fails closed at every step.

    The order of checks is the argument of this function:

    1. no plan            -> UNREGISTERED (exploratory only; never evidence)
    2. plan/code drift    -> DRIFTED       (a new study, not a result)
    3. already read       -> ALREADY_READ  (a repeated look is a search over time)
    4. sample too small   -> INCONCLUSIVE  (honest, and NOT a pass)
    5. family-wise test   -> CONFIRMED / NOT_CONFIRMED

    Check 4 sits deliberately before any statistical test. A significant result
    computed on three observations is exactly the false discovery this repo
    exists to avoid publishing, and "the test passed" must never be reachable
    ahead of "there was enough data to test".

    `expected_seal` is what the caller believes it is grading against; a
    mismatch means the plan was replaced underneath the reader. `event_log`
    enables the repeated-look check.
    """
    if registration is None:
        return ReadResult(
            state=READ_UNREGISTERED,
            registration_id="",
            registration_seal="",
            reason=(
                "no pre-registered plan covers this result; it is exploratory "
                "and cannot be cited as confirmation of anything"
            ),
            primary_arm="",
            primary_value=None,
            threshold=0.0,
            direction="gt",
            evidence_count=None,
            min_evidence=0,
        )

    reg = registration
    seal = reg.seal()
    primary = reg.primary_arm

    if expected_seal is not None and expected_seal != seal:
        return ReadResult(
            state=READ_DRIFTED,
            registration_id=reg.registration_id,
            registration_seal=seal,
            reason=(
                f"the plan now seals to {seal} but this read was set up against {expected_seal}; "
                "the decision procedure changed, so the result belongs to a new study"
            ),
            primary_arm=primary.arm_id,
            primary_value=None,
            threshold=reg.threshold,
            direction=reg.direction,
            evidence_count=None,
            min_evidence=reg.min_evidence,
        )

    if event_log is not None and already_read(event_log, reg.registration_id):
        return ReadResult(
            state=READ_ALREADY_READ,
            registration_id=reg.registration_id,
            registration_seal=seal,
            reason=(
                "this plan has already been read against evidence; a second read of the same "
                "plan is a search over readings, and the registered stopping rule governs when "
                "the result may be taken"
            ),
            primary_arm=primary.arm_id,
            primary_value=None,
            threshold=reg.threshold,
            direction=reg.direction,
            evidence_count=None,
            min_evidence=reg.min_evidence,
        )

    declared = {a.arm_id for a in reg.arms}
    seen: dict[str, Observation] = {}
    for obs in observations:
        if obs.arm_id not in declared:
            raise RegistrationError(
                f"observation for undeclared arm {obs.arm_id!r}; declared arms are {sorted(declared)}"
            )
        if obs.arm_id in seen:
            raise RegistrationError(f"two observations for arm {obs.arm_id!r}")
        seen[obs.arm_id] = obs

    primary_obs = seen.get(primary.arm_id)
    if primary_obs is None:
        return ReadResult(
            state=READ_INCONCLUSIVE,
            registration_id=reg.registration_id,
            registration_seal=seal,
            reason=f"no observation was recorded for the primary arm {primary.arm_id!r}",
            primary_arm=primary.arm_id,
            primary_value=None,
            threshold=reg.threshold,
            direction=reg.direction,
            evidence_count=None,
            min_evidence=reg.min_evidence,
        )

    if primary_obs.evidence_count < reg.min_evidence:
        return ReadResult(
            state=READ_INCONCLUSIVE,
            registration_id=reg.registration_id,
            registration_seal=seal,
            reason=(
                f"{primary_obs.evidence_count} observation(s) against a registered minimum of "
                f"{reg.min_evidence}; the plan is not yet readable, and an unread plan is "
                "neither confirmed nor refuted"
            ),
            primary_arm=primary.arm_id,
            primary_value=primary_obs.value,
            threshold=reg.threshold,
            direction=reg.direction,
            evidence_count=primary_obs.evidence_count,
            min_evidence=reg.min_evidence,
        )

    # Family-wise correction across every declared arm, when p-values exist.
    pvals = {arm_id: obs.p_value for arm_id, obs in seen.items() if obs.p_value is not None}
    per_arm: dict[str, Any] = {}
    family_rejected: bool | None = None
    if pvals:
        holm = holm_bonferroni(pvals, reg.alpha)
        per_arm["family_wise"] = holm
        family_rejected = holm["rejected"].get(primary.arm_id)
        for arm_id, obs in seen.items():
            per_arm[arm_id] = {
                "value": obs.value,
                "clears_threshold": direction_holds(obs.value, reg.direction, reg.threshold),
                "p_value": obs.p_value,
                "rejected_after_correction": holm["rejected"].get(arm_id),
            }
    else:
        for arm_id, obs in seen.items():
            per_arm[arm_id] = {
                "value": obs.value,
                "clears_threshold": direction_holds(obs.value, reg.direction, reg.threshold),
                "p_value": None,
                "rejected_after_correction": None,
            }

    clears = direction_holds(primary_obs.value, reg.direction, reg.threshold)
    if family_rejected is False:
        state = READ_NOT_CONFIRMED
        reason = (
            f"primary arm {primary.arm_id!r} cleared the registered threshold "
            f"({primary_obs.value:.4f} vs {reg.direction} {reg.threshold}) but did not survive "
            f"Holm-Bonferroni across the {len(pvals)} declared arm(s) at alpha={reg.alpha}"
        )
    elif clears:
        state = READ_CONFIRMED
        reason = (
            f"primary arm {primary.arm_id!r} met the registered criterion "
            f"({primary_obs.value:.4f} vs {reg.direction} {reg.threshold}) on "
            f"{primary_obs.evidence_count} observation(s), minimum {reg.min_evidence}"
        )
    else:
        state = READ_NOT_CONFIRMED
        reason = (
            f"primary arm {primary.arm_id!r} did not meet the registered criterion "
            f"({primary_obs.value:.4f} vs {reg.direction} {reg.threshold}) on "
            f"{primary_obs.evidence_count} observation(s)"
        )

    return ReadResult(
        state=state,
        registration_id=reg.registration_id,
        registration_seal=seal,
        reason=reason,
        primary_arm=primary.arm_id,
        primary_value=primary_obs.value,
        threshold=reg.threshold,
        direction=reg.direction,
        evidence_count=primary_obs.evidence_count,
        min_evidence=reg.min_evidence,
        per_arm=per_arm,
    )


# ---------------------------------------------------------------------------
# Append-only event log
# ---------------------------------------------------------------------------

_EVENT_HASH_FIELD = "event_hash"
_EVENT_PREV_FIELD = "prev_hash"
_GENESIS = "0" * 16


def _event_body(event: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in event.items() if k not in (_EVENT_HASH_FIELD, _EVENT_PREV_FIELD)}


def _hash_event(event: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(_event_body(event)).encode("utf-8")).hexdigest()[:16]


def append_event(
    path: str | Path,
    *,
    event_type: str,
    registration_id: str,
    registration_seal: str,
    payload: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Append one registration event, hash-chained to its predecessor.

    The chain exists so that "when was this plan first read?" has an answer
    that a later edit cannot quietly change. Deleting an inconvenient read
    breaks the chain; rewriting one breaks the chain; both are detectable.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    entries = load_events(p, strict=False)
    prev = entries[-1][_EVENT_HASH_FIELD] if entries else _GENESIS

    event: dict[str, Any] = {
        "event_type": event_type,
        "registration_id": registration_id,
        "registration_seal": registration_seal,
        "payload": payload or {},
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        _EVENT_PREV_FIELD: prev,
    }
    event[_EVENT_HASH_FIELD] = _hash_event(event)

    with p.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(event) + "\n")
    return event


def load_events(path: str | Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """Read the event chain. Interior corruption is fatal when `strict`.

    A missing file is an empty chain, not an error: no registration activity is
    a legitimate state. A malformed interior record never is.
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if not strict:
                break
            raise RegistrationError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(event, dict):
            if not strict:
                break
            raise RegistrationError(f"{p}:{lineno} is not a JSON object")
        events.append(event)
    if strict:
        verify_event_chain(events)
    return events


def verify_event_chain(events: Sequence[dict[str, Any]]) -> None:
    """Raise unless every event hashes as recorded and links to its predecessor."""
    prev = _GENESIS
    for i, event in enumerate(events):
        missing = {k for k in (_EVENT_HASH_FIELD, _EVENT_PREV_FIELD) if k not in event}
        if missing:
            raise RegistrationError(f"event {i} is missing {sorted(missing)}")
        if event[_EVENT_PREV_FIELD] != prev:
            raise RegistrationError(
                f"event {i} does not link to its predecessor: declares "
                f"{event[_EVENT_PREV_FIELD]} but chain is at {prev}"
            )
        expected = _hash_event(event)
        if event[_EVENT_HASH_FIELD] != expected:
            raise RegistrationError(
                f"event {i} was altered after it was written: recorded {event[_EVENT_HASH_FIELD]}, "
                f"content hashes to {expected}"
            )
        prev = event[_EVENT_HASH_FIELD]


def already_read(path: str | Path, registration_id: str, arm_id: str | None = None) -> bool:
    """Has this plan (or this arm) already been read against evidence?

    Used to catch the repeated-look problem: a tape examined until one reading
    looks good is a search over time, and this is what makes it visible.

    Scope note: this is a *detection* helper, not the rule itself. The
    once-only guarantee is enforced by `read_result(..., event_log=...)`,
    which returns `ALREADY_READ`; the caller then records the read with
    `record_read`. Detection and enforcement live in one place so they cannot
    drift apart.
    """
    for event in load_events(path, strict=False):
        if event.get("event_type") != "read":
            continue
        if event.get("registration_id") != registration_id:
            continue
        payload = event.get("payload") or {}
        if arm_id is not None and payload.get("arm_id") not in (None, arm_id):
            continue
        return True
    return False


def record_read(
    path: str | Path,
    result: ReadResult,
    *,
    arm_id: str | None = None,
) -> dict[str, Any]:
    """Append the read to the hash-chained log and return the appended event.

    Call this exactly once per accepted read, immediately after `read_result`
    returns CONFIRMED or NOT_CONFIRMED. A caller that grades a plan and forgets
    to record the read has not broken any invariant — the next read will simply
    still be permitted — so this is the one step that depends on discipline,
    and the surrounding code is arranged to make it the obvious one.
    """
    return append_event(
        path,
        event_type="read",
        registration_id=result.registration_id,
        registration_seal=result.registration_seal,
        payload={
            "state": result.state,
            "arm_id": arm_id if arm_id is not None else result.primary_arm,
            "primary_value": result.primary_value,
            "threshold": result.threshold,
            "direction": result.direction,
            "evidence_count": result.evidence_count,
            "min_evidence": result.min_evidence,
        },
    )
