"""Read the freeze pointer for CI. Emits GitHub Actions output lines.

The scheduled paper run must execute the FROZEN commit, not whatever is on
main today (see README "Code identity"). This script reads only the pointer
fields from freeze.json on the checked-out main tip and prints them as
$GITHUB_OUTPUT key=value lines; the workflow then checks out that ref and
verifies the code fingerprint before trading a single forward day.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FREEZE_FILE = Path("freeze.json")


def main() -> int:
    if not FREEZE_FILE.exists():
        print(f"::notice::no {FREEZE_FILE} committed yet — create one with 'python -m bot freeze'")
        return 0
    try:
        manifest = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"::error::freeze.json is not valid JSON ({e})")
        return 1
    commit = manifest.get("git_commit_at_freeze")
    if not commit or not isinstance(commit, str) or len(commit) < 7:
        print("::error::freeze.json has no usable git_commit_at_freeze — re-freeze with a current version")
        return 1
    gh = __import__("os").environ.get("GITHUB_OUTPUT")
    lines = [
        f"commit={commit}",
        f"code_sha256={manifest.get('code_sha256', '')}",
        f"config_sha256={manifest.get('config_sha256', '')}",
        f"frozen_at_date={manifest.get('frozen_at_date', '')}",
    ]
    tag = manifest.get("git_tag")
    if tag:
        lines.append(f"tag={tag}")
    digest = manifest.get("image_digest")
    if digest:
        lines.append(f"image_digest={digest}")
    out = "\n".join(lines) + "\n"
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
