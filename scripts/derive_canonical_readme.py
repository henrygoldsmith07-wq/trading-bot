"""Derive the README canonical block from the evidence document.

Usage:
  python scripts/derive_canonical_readme.py            # rewrite README block
  python scripts/derive_canonical_readme.py --check    # fail if out of sync

The evidence document (bot/reporting.build_evidence) is the single source of
truth: the block between the CANONICAL markers in README.md is rendered from
it, and the performance sentence inside the block is the canonical verdict
object's sentence — never a hand-written variant. Hand edits inside the block
are overwritten, and --check fails in CI when the README has drifted.

Deterministic: the same evidence document renders byte-identical markdown, so
`--check` is a pure consistency gate and never a race with the wall clock.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow importing bot/ from repo root regardless of how this script is invoked.
sys.path.insert(0, str(ROOT))

from bot.reporting import (  # noqa: E402
    CANONICAL_RUN_ID,
    build_evidence,
    render_readme_block,
)

RECORD = ROOT / "runs" / CANONICAL_RUN_ID / "run.json"
README = ROOT / "README.md"
BEGIN = f"<!-- CANONICAL:BEGIN — generated from the {CANONICAL_RUN_ID} evidence document; do not edit by hand -->"
END = "<!-- CANONICAL:END -->"


def render(record: dict | None = None, *, artifacts: dict | None = None) -> str:
    """Render the canonical block.

    `record` is accepted for compatibility and ignored: the block renders from
    the evidence document, so no caller can feed the formatter a record the
    evidence layer would refuse (a stale record, a comparator block, or a
    record from a different run). Passing one does not change a single byte of
    the output. `artifacts` lets tests inject a synthetic evidence world
    without touching the committed tapes.
    """
    del record
    doc = build_evidence(ROOT, artifacts=artifacts)
    return render_readme_block(doc)


def main() -> int:
    check = "--check" in sys.argv
    try:
        rendered = render()
    except Exception as exc:
        print(f"FAIL: cannot render the canonical block from the evidence document: {exc}")
        return 2
    md = README.read_text(encoding="utf-8")
    if BEGIN not in md or END not in md:
        print("markers missing from README.md — insert them around the headline block first")
        return 2
    pre = md.split(BEGIN, 1)[0]
    post = md.split(END, 1)[1]
    new = pre + BEGIN + "\n" + rendered + "\n\n" + END + post
    if new == md:
        print("README already in sync with the evidence document")
        return 0
    if check:
        print(f"FAIL: README canonical block is out of sync with the {CANONICAL_RUN_ID} evidence document")
        return 1
    README.write_text(new, encoding="utf-8")
    print("README canonical block regenerated from the evidence document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
