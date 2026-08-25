"""Code identity: a deterministic fingerprint of the implementation.

A freeze that pins configuration but not implementation is not a frozen
experiment: `strategy.py` could change the day after freezing while
`freeze.json` keeps claiming the old behaviour ran. This module gives every
freeze a second seal — a hash over the *source* itself — so the forward
runner can refuse to execute anything except the exact code that was frozen.

Determinism contract (algorithm `sha256-lf-v1`):
- Files included: every `bot/*.py` plus `pyproject.toml`, walked in sorted
  relative-path order.
- Content is decoded as UTF-8 and normalised to LF before hashing, so a
  Windows working tree and a Linux CI checkout of the same commit produce
  the same digest regardless of `core.autocrlf`.
- The manifest records which algorithm produced the digest; a runner that
  does not recognise the algorithm must refuse rather than guess.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CODE_FINGERPRINT_ALGO = "sha256-lf-v1"

# Paths hashed, relative to the repo root. Sorted for determinism.
_CODE_PATHS = sorted(["pyproject.toml", *[f"bot/{p.name}" for p in Path(__file__).parent.glob("*.py")]])


def _normalise(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def code_fingerprint(root: str | Path | None = None) -> str:
    """sha256 over the frozen source set, LF-normalised. Stable across OSes."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    seen_any = False
    for rel in _CODE_PATHS:
        p = base / rel
        if not p.exists():
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        try:
            content = _normalise(p.read_bytes())
        except OSError:
            content = b"<unreadable>"
        h.update(content)
        h.update(b"\n")
        seen_any = True
    if not seen_any:
        raise RuntimeError("no source files found to fingerprint")
    return h.hexdigest()


def current_git_commit(repo: str | Path | None = None) -> str | None:
    """HEAD commit id, or None when unavailable / not a repo."""
    import subprocess

    root = Path(repo) if repo is not None else Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def verify_freeze_code(manifest: dict, root: str | Path | None = None) -> None:
    """Raise unless the running source matches the manifest's frozen code.

    Refuses when the manifest predates code sealing (no fingerprint recorded)
    as well as on an actual mismatch: both mean we cannot prove what would run.
    """
    algo = manifest.get("code_fingerprint_algo")
    expected = manifest.get("code_sha256")
    if not algo or not expected:
        raise ValueError(
            "freeze manifest has no code fingerprint (code_sha256 missing) — "
            "it was created by a pre-sealing version and cannot be proven to "
            "run the frozen implementation. Re-freeze with 'python -m bot freeze'."
        )
    if algo != CODE_FINGERPRINT_ALGO:
        raise ValueError(
            f"freeze uses unknown code-fingerprint algorithm {algo!r}; this runner "
            f"implements {CODE_FINGERPRINT_ALGO!r}. Refusing to execute."
        )
    actual = code_fingerprint(root)
    if actual != expected:
        raise ValueError(
            "CODE MISMATCH: the running implementation does not match the freeze.\n"
            f"  frozen under : {manifest.get('frozen_at')}\n"
            f"  frozen commit: {manifest.get('git_commit_at_freeze')}\n"
            f"  expected sha : {expected}\n"
            f"  running sha  : {actual}\n"
            "The forward period must be traded on the frozen code only. Check out "
            "the frozen commit (or rebuild its image) instead of editing the manifest."
        )


def source_file_fingerprint(rel_paths: list[str], root: str | Path | None = None) -> dict:
    """Per-file sha256 (LF-normalised) for a group of source files plus a
    combined hash — used to seal strategy definitions / portfolio rules /
    universe modules individually inside run records."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for rel in sorted(rel_paths):
        p = base / rel
        try:
            digest = hashlib.sha256(_normalise(p.read_bytes())).hexdigest()
        except OSError:
            digest = "missing"
        files[rel] = digest
        combined.update(digest.encode())
        combined.update(b"\n")
    return {"files": files, "combined": combined.hexdigest()}


if __name__ == "__main__":  # tiny debug helper: python -m bot.identity
    print(f"{CODE_FINGERPRINT_ALGO} {code_fingerprint()}", file=sys.stderr)
