"""Provenance: what produced a canonical record, and can we prove it?

A canonical run record is evidence. If it names a commit, that commit must be
the source that produced it, and the tree it ran in must be the tree that is
committed. Otherwise the record is a story about code nobody can retrieve, and
every reproduction attempt is guaranteed to fail or, worse, to "succeed"
against different inputs.

This module makes both properties explicit and checkable:

* the working tree state at the moment of the run (clean or dirty, with the
  offending paths);
* HEAD at the moment of the run;
* agreement between the source fingerprint recorded in a record and the source
  fingerprint of the tree it claims to come from;
* a refusal path for canonical runs and freezes executed on a dirty tree,
  with an explicit, loudly-named non-canonical override.

Fail closed by default. An override exists because exploratory runs are
legitimate; they just must never be labelled canonical.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DIRTY_REFUSAL_HINT = (
    "Canonical evidence must be produced from a clean, committed tree. "
    "Commit or stash the change, or re-run with --allow-dirty-tree and treat "
    "the result as NON-canonical."
)


class DirtyTreeError(RuntimeError):
    """A canonical artefact was requested from a dirty working tree."""


class ProvenanceMismatch(RuntimeError):
    """A record claims a source that does not correspond to the code."""


def _run_git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    # NOTE: do not strip the payload here. `git status --porcelain` pads each
    # line with a two-character status column, and a global .strip() would eat
    # the first character of the first path.
    return out.stdout


@dataclass(frozen=True)
class GitState:
    """The repository state a piece of evidence was produced from."""

    root: Path
    head_commit: str | None
    dirty: bool
    dirty_paths: tuple[str, ...] = ()
    is_repo: bool = True

    def as_dict(self) -> dict:
        return {
            "head_commit": self.head_commit,
            "dirty": self.dirty,
            "dirty_paths": list(self.dirty_paths),
            "is_repo": self.is_repo,
        }

    def describe(self) -> str:
        if not self.is_repo:
            return "not a git repository"
        head = (self.head_commit or "unknown")[:12]
        if not self.dirty:
            return f"clean at {head}"
        shown = ", ".join(self.dirty_paths[:5])
        if len(self.dirty_paths) > 5:
            shown += f", +{len(self.dirty_paths) - 5} more"
        return f"DIRTY at {head}: {shown}"


def git_state(root: str | Path | None = None) -> GitState:
    """Read HEAD and cleanliness. Never raises: absence of git is data too."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    head = (_run_git(base, "rev-parse", "HEAD") or "").strip() or None
    if head is None:
        return GitState(root=base, head_commit=None, dirty=False, is_repo=False)
    status = _run_git(base, "status", "--porcelain")
    paths: list[str] = []
    if status:
        for line in status.splitlines():
            if not line.strip():
                continue
            # porcelain lines look like "XY path"; the path is what matters
            entry = line[3:].strip() if len(line) > 3 else line.strip()
            # a rename reads "old -> new"; the new path is the one in the tree
            if " -> " in entry:
                entry = entry.rsplit(" -> ", 1)[-1]
            paths.append(entry)
    return GitState(root=base, head_commit=head, dirty=bool(paths), dirty_paths=tuple(paths))


def require_clean_tree(
    root: str | Path | None = None,
    *,
    allow_dirty: bool = False,
    what: str = "canonical run",
) -> GitState:
    """Return the git state, refusing a dirty tree unless explicitly overridden."""
    state = git_state(root)
    if state.dirty and not allow_dirty:
        raise DirtyTreeError(
            f"refusing to create a {what} from a dirty working tree.\n"
            f"  {state.describe()}\n"
            f"  {DIRTY_REFUSAL_HINT}"
        )
    return state


@dataclass
class SourceFingerprint:
    """Everything needed to decide whether a record matches a source tree."""

    code_sha256: str
    module_hashes: dict[str, str] = field(default_factory=dict)
    clean: bool = True
    head_commit: str | None = None

    def as_dict(self) -> dict:
        return {
            "algo": "sha256-lf-v1",
            "sha256": self.code_sha256,
            "module_hashes": dict(self.module_hashes),
            "clean": self.clean,
            "head_commit": self.head_commit,
        }


SEALED_MODULES = (
    "bot/strategy.py",
    "bot/portfolio_rules.py",
    "bot/universe.py",
    "bot/universe_pit.py",
    "bot/verdict.py",
    "bot/stats_validation.py",
    "bot/engine.py",
    "bot/paper.py",
    "bot/prospective.py",
    "bot/canonical_identity.py",
)


def source_fingerprint(root: str | Path | None = None) -> SourceFingerprint:
    """Fingerprint the running source tree, including per-module hashes."""
    from .identity import code_fingerprint, source_file_fingerprint

    state = git_state(root)
    return SourceFingerprint(
        code_sha256=code_fingerprint(root),
        module_hashes=source_file_fingerprint(list(SEALED_MODULES), root)["files"],
        clean=not state.dirty,
        head_commit=state.head_commit,
    )


def record_environment(root: str | Path | None = None) -> dict:
    """The environment block a canonical run record should carry."""
    import platform

    from .canonical_identity import PRIMARY_RULE_ID

    fp = source_fingerprint(root)
    return {
        "python": platform.python_version(),
        "git_commit": fp.head_commit,
        "code_fingerprint": {"algo": "sha256-lf-v1", "sha256": fp.code_sha256},
        "module_hashes": fp.module_hashes,
        "source_clean": fp.clean,
        "primary_rule_id": PRIMARY_RULE_ID,
    }


def assert_record_matches_source(record: dict, root: str | Path | None = None) -> None:
    """A canonical record must not claim a commit/fingerprint it cannot have produced.

    Checks, in order of severity:
      * the record names a commit and the tree here is at that commit;
      * the record's code fingerprint equals the running source's fingerprint;
      * the record was produced from a clean tree (unless it says otherwise).
    """
    if not isinstance(record, dict):
        raise ProvenanceMismatch("record is not an object")
    env = record.get("results", record).get("environment") if isinstance(record.get("results", record), dict) else None
    if not isinstance(env, dict):
        raise ProvenanceMismatch("record carries no environment block; provenance cannot be checked")
    state = git_state(root)
    claimed_commit = env.get("git_commit")
    claimed_fp = (env.get("code_fingerprint") or {}).get("sha256")

    if not claimed_commit:
        raise ProvenanceMismatch(
            "record records no git_commit. A canonical record that cannot name the "
            "source that produced it is not reproducible evidence."
        )
    if state.is_repo and state.head_commit and claimed_commit != state.head_commit:
        raise ProvenanceMismatch(
            f"record claims commit {claimed_commit[:12]} but the tree is at "
            f"{state.head_commit[:12]}.\n"
            "The record was produced by different source than this checkout; "
            "regenerate it here or check out the commit it names.",
        )
    if claimed_fp:
        from .identity import code_fingerprint

        actual = code_fingerprint(root)
        if actual != claimed_fp:
            raise ProvenanceMismatch(
                f"record claims source fingerprint {claimed_fp[:12]} but this tree hashes "
                f"to {actual[:12]}.\n"
                "The code that produced the record is not the code in front of you.",
            )
    if env.get("source_clean") is False:
        raise ProvenanceMismatch(
            "record was produced from a DIRTY working tree and is therefore not "
            "canonical evidence. Re-run from a clean, committed checkout.",
        )
