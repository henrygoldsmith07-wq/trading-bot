"""Explicit, tested transfer of forward evidence from the frozen tree to main.

The scheduled paper run computes one forward day by executing the FROZEN
implementation in an isolated git worktree, then has to land that evidence on
main, which is a different tree and usually has newer commits. That transfer
used to be inline Python in a GitHub Actions heredoc, and it failed in a way
that quietly destroyed evidence:

    git checkout -B main origin/main
    # error: Your local changes to the following files would be overwritten
    # by checkout: forward_log.jsonl, freeze.json, universe_log.jsonl

The forward day had ALREADY been computed at that point, so the run died
having silently discarded it. The failure was never about evidence integrity;
it was a dirty working tree colliding with a branch switch.

This module is the fix, and it is testable on its own:

* the frozen tree is created with "git worktree add" so main is never dirtied
  by the forward run (see .github/workflows/scheduled-paper.yml);
* the merge lives here, in importable Python with unit tests, rather than in an
  untestable heredoc;
* merging is keyed on the evidence DATE. An existing row is never rewritten,
  so a concurrent commit on main can delay evidence but never overwrite it;
* the merge is idempotent, so re-running it after a push conflict converges;
* the freeze identity is rechecked against main before anything is written, so
  evidence produced under one freeze is never attached to another;
* malformed input fails CLOSED. A truncated JSONL line is a corrupt evidence
  tape, not a line to skip past.

Paper trading only. This module moves records; it never places orders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORWARD_LOG = "forward_log.jsonl"
UNIVERSE_LOG = "universe_log.jsonl"
COST_LOG = "cost_observations.jsonl"
FREEZE_FILE = "freeze.json"
STAGING_MANIFEST = "evidence_staging.json"

__all__ = [
    "COST_LOG",
    "EvidenceIntegrityError",
    "FORWARD_LOG",
    "FreezeChangedError",
    "FreezeIdentity",
    "MergeReport",
    "STAGING_MANIFEST",
    "UNIVERSE_LOG",
    "freeze_identity",
    "merge_cost_observations",
    "merge_forward_entries",
    "merge_universe_snapshots",
    "read_jsonl",
    "transfer_evidence",
    "validate_log",
]


class EvidenceIntegrityError(RuntimeError):
    """Evidence on disk is malformed, inconsistent or self-contradictory."""


class FreezeChangedError(EvidenceIntegrityError):
    """The active freeze moved while evidence was being produced."""


@dataclass(frozen=True)
class FreezeIdentity:
    """The subset of a freeze manifest that decides evidence admissibility."""

    commit: str
    code_sha256: str
    config_sha256: str
    frozen_at_date: str

    def as_dict(self) -> dict[str, str]:
        return {
            "commit": self.commit,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "frozen_at_date": self.frozen_at_date,
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> FreezeIdentity:
        return cls(
            commit=str(manifest.get("git_commit_at_freeze") or ""),
            code_sha256=str(manifest.get("code_sha256") or ""),
            config_sha256=str(manifest.get("config_sha256") or ""),
            frozen_at_date=str(manifest.get("frozen_at_date") or ""),
        )


@dataclass
class MergeReport:
    """What a merge actually did. Every field is auditable; none are silent."""

    forward_added: list[str] = field(default_factory=list)
    forward_duplicate: list[str] = field(default_factory=list)
    forward_pre_freeze_dropped: list[str] = field(default_factory=list)
    universe_added: list[str] = field(default_factory=list)
    universe_duplicate: list[str] = field(default_factory=list)
    cost_added: int = 0
    cost_duplicate: int = 0
    written: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.forward_added or self.universe_added or self.cost_added)

    def as_dict(self) -> dict[str, Any]:
        return {
            "forward_added": list(self.forward_added),
            "forward_duplicate": list(self.forward_duplicate),
            "forward_pre_freeze_dropped": list(self.forward_pre_freeze_dropped),
            "universe_added": list(self.universe_added),
            "universe_duplicate": list(self.universe_duplicate),
            "cost_added": self.cost_added,
            "cost_duplicate": self.cost_duplicate,
            "written": list(self.written),
            "changed": self.changed,
        }

    def render(self) -> str:
        fwd = ", ".join(self.forward_added)
        uni = ", ".join(self.universe_added)
        lines = [
            f"forward rows added        : {len(self.forward_added)}" + (f" ({fwd})" if fwd else ""),
            f"forward rows already there: {len(self.forward_duplicate)}",
            f"forward rows pre-freeze   : {len(self.forward_pre_freeze_dropped)} dropped",
            f"universe snapshots added  : {len(self.universe_added)}" + (f" ({uni})" if uni else ""),
            f"universe snapshots dupes  : {len(self.universe_duplicate)}",
            f"cost observations added   : {self.cost_added}",
            f"cost observations dupes   : {self.cost_duplicate}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading: fail closed on anything malformed
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL evidence tape. One bad line voids the whole file.

    Skipping a corrupt line would silently shrink a prospective evidence tape,
    which is exactly the quiet loss this system exists to prevent. We refuse
    instead, and say which line broke.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:  # an unreadable tape is not an empty tape
        raise EvidenceIntegrityError(f"cannot read evidence file {p}: {exc}") from exc
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceIntegrityError(
                f"{p} line {lineno} is not valid JSON ({exc}); refusing to merge a "
                f"partially-readable evidence tape"
            ) from exc
        if not isinstance(entry, dict):
            raise EvidenceIntegrityError(f"{p} line {lineno} is not a JSON object")
        # Every tape this module moves is a DATED evidence tape. A row with no
        # usable date cannot be de-duplicated, ordered, or checked against the
        # freeze boundary, so it is corruption rather than an odd-but-valid row.
        # Checking here means every consumer fails closed, not just the merge.
        _entry_date(entry, kind="evidence", where=f"{p} line {lineno}")
        out.append(entry)
    return out


def _entry_date(entry: dict[str, Any], *, kind: str, where: str) -> str:
    date = entry.get("date")
    if not isinstance(date, str) or not date.strip():
        raise EvidenceIntegrityError(
            f"{where}: {kind} entry has no usable date field "
            f"(keys: {sorted(entry)}); evidence cannot be de-duplicated without it"
        )
    return date.strip()


def validate_log(entries: list[dict[str, Any]], *, kind: str = "forward") -> dict[str, Any]:
    """Report duplicate dates and ordering problems without mutating anything.

    Used by tests and by "forward-evidence verify" so a tape's integrity can be
    asserted independently of the merge that writes it.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    dates: list[str] = []
    for entry in entries:
        date = _entry_date(entry, kind=kind, where=f"{kind} log")
        dates.append(date)
        if date in seen:
            duplicates.append(date)
        seen.add(date)
    return {
        "kind": kind,
        "rows": len(dates),
        "unique_dates": len(seen),
        "duplicate_dates": sorted(set(duplicates)),
        "sorted": dates == sorted(dates),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
    }


# ---------------------------------------------------------------------------
# merging: keyed on date, existing rows immutable
# ---------------------------------------------------------------------------

def merge_forward_entries(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    cutoff_date: str | None = None,
) -> tuple[list[dict[str, Any]], MergeReport]:
    """Merge forward-day records. Existing rows are NEVER rewritten.

    cutoff_date is the freeze's frozen_at_date. Nothing downstream filters the
    tape by freeze date, so a row dated on or before the freeze would read as
    forward evidence for a rule that had not been sealed yet. The boundary is
    enforced here on every run rather than once by hand.
    """
    report = MergeReport()
    merged = [dict(e) for e in existing]
    taken = {_entry_date(e, kind="forward", where="existing forward log") for e in merged}
    for entry in incoming:
        date = _entry_date(entry, kind="forward", where="incoming forward log")
        if date in taken:
            # A concurrent commit on main already recorded this day. The row
            # already committed is the row of record; appending a second row for
            # the same date would double-count the day.
            report.forward_duplicate.append(date)
            continue
        if cutoff_date and date <= cutoff_date:
            report.forward_pre_freeze_dropped.append(date)
            continue
        merged.append(dict(entry))
        taken.add(date)
        report.forward_added.append(date)
    merged.sort(key=lambda e: _entry_date(e, kind="forward", where="merged forward log"))
    return merged, report


def _snapshot_body(entry: dict[str, Any]) -> str:
    """The membership claim itself, ignoring when the snapshot was generated."""
    return json.dumps(entry.get("universe"), sort_keys=True)


def merge_universe_snapshots(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], MergeReport]:
    """Merge dated universe snapshots. Same date and same universe is a no-op.

    Same date but a DIFFERENT universe is a contradiction: two observations
    claiming one date saw different membership. That fails closed rather than
    picking a winner, because silently keeping either one would put a fabricated
    membership record into first-class evidence.
    """
    report = MergeReport()
    by_date: dict[str, dict[str, Any]] = {}
    for entry in existing:
        by_date[_entry_date(entry, kind="universe", where="existing universe log")] = dict(entry)
    for entry in incoming:
        date = _entry_date(entry, kind="universe", where="incoming universe log")
        prior = by_date.get(date)
        if prior is not None:
            if _snapshot_body(prior) != _snapshot_body(entry):
                raise EvidenceIntegrityError(
                    f"universe snapshot for {date} conflicts with the committed record: "
                    f"committed={_snapshot_body(prior)!r} incoming={_snapshot_body(entry)!r}. "
                    "Two different membership observations cannot both be evidence for one date."
                )
            report.universe_duplicate.append(date)
            continue
        by_date[date] = dict(entry)
        report.universe_added.append(date)
    return [by_date[d] for d in sorted(by_date)], report


def merge_cost_observations(existing_text: str, incoming_text: str) -> tuple[str, int, int]:
    """Append-only cost tape, de-duplicated by exact line content."""
    existing_lines = [ln for ln in existing_text.splitlines() if ln.strip()]
    seen = set(existing_lines)
    out = list(existing_lines)
    added = 0
    dupes = 0
    for line in incoming_text.splitlines():
        if not line.strip():
            continue
        if line in seen:
            dupes += 1
            continue
        seen.add(line)
        out.append(line)
        added += 1
    return ("\n".join(out) + "\n") if out else "", added, dupes


# ---------------------------------------------------------------------------
# freeze identity
# ---------------------------------------------------------------------------

def freeze_identity(manifest: dict[str, Any]) -> FreezeIdentity:
    return FreezeIdentity.from_manifest(manifest)


def read_staging_manifest(staging: Path) -> dict[str, Any]:
    p = staging / STAGING_MANIFEST
    if not p.exists():
        raise EvidenceIntegrityError(
            f"staging directory {staging} has no {STAGING_MANIFEST}; the frozen run did "
            f"not declare what it produced, so nothing may be merged from it"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceIntegrityError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceIntegrityError(f"{p} must contain a JSON object")
    return data


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file plus replace, so a crash cannot truncate the tape."""
    tmp = path.with_name(path.name + ".tmp-transfer")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _dump(entries: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)


# ---------------------------------------------------------------------------
# the transfer itself
# ---------------------------------------------------------------------------

def transfer_evidence(
    staging: str | Path,
    repo: str | Path,
    *,
    require_freeze_match: bool = True,
) -> MergeReport:
    """Merge staged evidence into the repository's persistent logs.

    Returns a MergeReport describing exactly what changed. Raises rather than
    partially applying when the evidence is malformed, or when the freeze moved
    underneath the run.
    """
    staging = Path(staging)
    repo = Path(repo)
    if not staging.is_dir():
        raise EvidenceIntegrityError(f"staging directory does not exist: {staging}")

    staged = read_staging_manifest(staging)
    repo_freeze_path = repo / FREEZE_FILE
    if not repo_freeze_path.exists():
        raise EvidenceIntegrityError(
            f"{repo_freeze_path} is missing; there is no active freeze to attach evidence to"
        )
    try:
        repo_manifest = json.loads(repo_freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceIntegrityError(f"{repo_freeze_path} is not valid JSON: {exc}") from exc
    repo_identity = freeze_identity(repo_manifest)

    produced = staged.get("produced_under_freeze")
    if produced is None:
        raise EvidenceIntegrityError(
            f"{staging / STAGING_MANIFEST} does not record produced_under_freeze; evidence "
            f"without a declared freeze identity cannot be admitted"
        )
    run_identity = freeze_identity(produced)
    if require_freeze_match and run_identity != repo_identity:
        raise FreezeChangedError(
            "the active freeze changed while this forward run was executing, so its evidence "
            "belongs to a different experiment and must not be attached to the current one.\n"
            f"  produced under : {run_identity.as_dict()}\n"
            f"  main now has   : {repo_identity.as_dict()}\n"
            "Staged evidence has been preserved; re-run the forward step under the new freeze."
        )

    cutoff = repo_identity.frozen_at_date or None
    report = MergeReport()

    staged_forward = staging / FORWARD_LOG
    if staged_forward.exists():
        incoming = read_jsonl(staged_forward)
        existing = read_jsonl(repo / FORWARD_LOG)
        merged, fwd = merge_forward_entries(existing, incoming, cutoff_date=cutoff)
        report.forward_added = fwd.forward_added
        report.forward_duplicate = fwd.forward_duplicate
        report.forward_pre_freeze_dropped = fwd.forward_pre_freeze_dropped
        _atomic_write(repo / FORWARD_LOG, _dump(merged))
        report.written.append(FORWARD_LOG)

    staged_universe = staging / UNIVERSE_LOG
    if staged_universe.exists():
        incoming = read_jsonl(staged_universe)
        existing = read_jsonl(repo / UNIVERSE_LOG)
        merged, uni = merge_universe_snapshots(existing, incoming)
        report.universe_added = uni.universe_added
        report.universe_duplicate = uni.universe_duplicate
        _atomic_write(repo / UNIVERSE_LOG, _dump(merged))
        report.written.append(UNIVERSE_LOG)

    staged_cost = staging / COST_LOG
    if staged_cost.exists():
        repo_cost = repo / COST_LOG
        existing_text = repo_cost.read_text(encoding="utf-8") if repo_cost.exists() else ""
        incoming_text = staged_cost.read_text(encoding="utf-8")
        text, added, dupes = merge_cost_observations(existing_text, incoming_text)
        report.cost_added = added
        report.cost_duplicate = dupes
        if added:
            _atomic_write(repo_cost, text)
            report.written.append(COST_LOG)

    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bot forward-evidence",
        description="Merge forward evidence produced by the frozen tree into the repository logs.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="merge staged evidence into the repository logs")
    m.add_argument("--staging", required=True, help="directory holding the frozen run's evidence")
    m.add_argument("--repo", default=".", help="repository root (default: cwd)")
    m.add_argument(
        "--allow-freeze-change",
        action="store_true",
        help="merge even if the active freeze changed (for quarantining, not for scoring)",
    )
    v = sub.add_parser("verify", help="validate the repository evidence tapes")
    v.add_argument("--repo", default=".", help="repository root (default: cwd)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "merge":
        try:
            report = transfer_evidence(
                args.staging, args.repo, require_freeze_match=not args.allow_freeze_change
            )
        except FreezeChangedError as exc:
            # A re-freeze racing a scheduled run is a normal, safe outcome: the
            # evidence belongs to a different experiment, so it is NOT attached
            # and NOT destroyed. It is still uploaded as an artifact. Succeeding
            # here keeps the signal readable; the notice carries the reason.
            print(f"::notice::{exc}")
            return 0
        except EvidenceIntegrityError as exc:
            print(f"::error::{exc}")
            return 1
        print(report.render())
        return 0
    if args.command == "verify":
        repo = Path(args.repo)
        for name, kind in ((FORWARD_LOG, "forward"), (UNIVERSE_LOG, "universe")):
            try:
                print(json.dumps(validate_log(read_jsonl(repo / name), kind=kind), sort_keys=True))
            except EvidenceIntegrityError as exc:
                print(f"::error::{exc}")
                return 1
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
