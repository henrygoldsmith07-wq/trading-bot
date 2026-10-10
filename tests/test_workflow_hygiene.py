"""Guards on GitHub Actions workflow files themselves.

A workflow file is code that only runs in CI, so nothing in the local test
suite executes it. That made an invalid workflow invisible: GitHub rejects the
file at registration time and reports a zero-job "workflow file issue"
failure, which looks like flaky infrastructure rather than a bug in the repo.

This module validates the files against the rules GitHub actually enforces, so
the failure surfaces in `pytest` instead of on the next push.

Why this exists
---------------
`scheduled-paper.yml` declared, at WORKFLOW level:

    env:
      FROZEN_TREE: ${{ runner.temp }}/frozen
      STAGING:    ${{ runner.temp }}/staging

The `runner` context is not available in the workflow-level `env:` key --
GitHub permits only `github`, `secrets`, `inputs` and `vars` there. The file
was therefore invalid, and the failure was silent in the worst possible way:

  * every push produced a 0s, 0-job failed run (easy to dismiss as noise);
  * every SCHEDULED run stopped advancing the forward tape entirely.

The prospective evidence tape stopped on 2026-09-22 and nobody noticed,
because the failure never appeared inside a job where it could be logged. The
whole point of this repo is prospective validation, so a silently stopped
forward test is the most expensive possible bug: the paper record simply stops
growing while every retrospective number still looks healthy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))

# From GitHub's context-availability table. Each entry maps a workflow key to
# the contexts that may be used in an expression at that key. Only the keys
# this repo uses are listed; anything unlisted is treated as unconstrained.
ALLOWED_CONTEXTS: dict[str, set[str]] = {
    "workflow.env": {"github", "secrets", "inputs", "vars"},
    "workflow.concurrency": {"github", "inputs", "vars"},
    "job.env": {"github", "needs", "strategy", "matrix", "vars", "secrets", "inputs"},
    "job.if": {"github", "needs", "vars", "inputs"},
    "job.runs-on": {"github", "needs", "strategy", "matrix", "vars", "inputs"},
    "step.env": {
        "github", "needs", "strategy", "matrix", "job", "runner",
        "env", "vars", "secrets", "steps", "inputs",
    },
    "step.run": {
        "github", "needs", "strategy", "matrix", "job", "runner",
        "env", "vars", "secrets", "steps", "inputs",
    },
    "step.with": {
        "github", "needs", "strategy", "matrix", "job", "runner",
        "env", "vars", "secrets", "steps", "inputs",
    },
    "step.working-directory": {
        "github", "needs", "strategy", "matrix", "job", "runner",
        "env", "vars", "secrets", "steps", "inputs",
    },
}

# `on` parses to the boolean True under YAML 1.1, so a workflow dict's trigger
# key is literally True rather than the string "on".
_EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)

# The context is the FIRST identifier of a dotted chain. In
# `steps.pointer.outputs.commit` only `steps` is a context; `pointer` and
# `outputs` are properties of it. Anchoring on a non-identifier boundary
# (start, or a character that cannot continue an identifier) avoids matching
# the inner segments.
_IDENT = re.compile(r"(?:^|[^A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*\.")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _contexts_in(value: object) -> set[str]:
    """Every context name referenced inside ${{ ... }} in a scalar.

    Recurses into mappings and sequences, because the keys this module guards
    are frequently mappings (`env:`, `concurrency:`, `with:`) rather than plain
    strings. Returning early on a non-string would skip exactly the case that
    caused the outage: `env:` is a mapping, so a string-only implementation
    silently checked nothing and reported the invalid file as clean.
    """
    if isinstance(value, str):
        found: set[str] = set()
        for expr in _EXPR.findall(value):
            for name in _IDENT.findall(expr):
                found.add(name)
        return found
    if isinstance(value, dict):
        found = set()
        for v in value.values():
            found |= _contexts_in(v)
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for v in value:
            found |= _contexts_in(v)
        return found
    return set()


def _check(where: str, key: str, value: object, problems: list[str]) -> None:
    allowed = ALLOWED_CONTEXTS.get(key)
    if allowed is None:
        return
    used = _contexts_in(value)
    illegal = used - allowed
    if illegal:
        problems.append(
            f"{where}: {key} uses context(s) {sorted(illegal)}; "
            f"only {sorted(allowed)} are permitted there"
        )


def _walk_workflow(path: Path) -> list[str]:
    doc = _load(path)
    problems: list[str] = []
    label = path.name

    _check(label, "workflow.env", doc.get("env"), problems)
    _check(label, "workflow.concurrency", doc.get("concurrency"), problems)

    for job_id, job in (doc.get("jobs") or {}).items():
        where = f"{label}:{job_id}"
        _check(where, "job.env", job.get("env"), problems)
        _check(where, "job.if", job.get("if"), problems)
        _check(where, "job.runs-on", job.get("runs-on"), problems)

        for i, step in enumerate(job.get("steps") or []):
            sname = step.get("name") or step.get("uses") or f"step[{i}]"
            swhere = f"{where}:{sname}"
            _check(swhere, "step.env", step.get("env"), problems)
            _check(swhere, "step.run", step.get("run"), problems)
            _check(swhere, "step.with", step.get("with"), problems)
            _check(swhere, "step.working-directory", step.get("working-directory"), problems)

    return problems


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_as_yaml(path: Path) -> None:
    doc = _load(path)
    assert isinstance(doc, dict), f"{path.name} does not parse to a mapping"
    assert doc.get("jobs"), f"{path.name} declares no jobs"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_uses_only_permitted_contexts(path: Path) -> None:
    """The regression: `runner` is illegal in workflow-level `env`.

    That single mistake invalidated the whole file, so GitHub ran zero jobs and
    the prospective forward test stopped advancing without ever logging why.
    """
    problems = _walk_workflow(path)
    assert not problems, "\n".join(problems)


def test_runner_context_is_never_used_at_workflow_level_env() -> None:
    """Direct, named guard for the exact defect that cost the forward tape.

    Kept separate from the table-driven test so a future refactor of the table
    cannot quietly stop covering the case that actually broke.
    """
    offenders: list[str] = []
    for path in WORKFLOWS:
        env = _load(path).get("env")
        if isinstance(env, dict) and "runner" in _contexts_in(env):
            offenders.append(path.name)
    assert not offenders, (
        "`runner` is not available in workflow-level `env`; declaring it there "
        f"makes the workflow invalid and every run a zero-job failure: {offenders}"
    )


def test_scheduled_workflow_can_actually_reach_its_forward_step() -> None:
    """A schedule-only workflow must still be a *valid* file.

    Guards the blast radius: when the file is invalid, GitHub silently stops
    running the schedule, which is precisely the outcome that leaves the
    prospective tape frozen while the repo looks healthy.
    """
    path = WORKFLOW_DIR / "scheduled-paper.yml"
    doc = _load(path)
    assert doc.get("jobs"), "scheduled-paper.yml declares no jobs"

    steps = doc["jobs"]["forward-step"]["steps"]
    names = [s.get("name") for s in steps]
    assert "Advance the prospective paper portfolio one day" in names

    # Every step that references one of these shell variables must declare it
    # at step level. They are NOT available from workflow-level `env` (see
    # above), so a step that reads one without declaring it would run with the
    # variable unset -- and every one of these scripts runs under `set -u`,
    # which turns that into an immediate abort.
    for step in steps:
        blob = " ".join(str(step.get(k, "")) for k in ("run", "working-directory", "with"))
        for var in ("FROZEN_TREE", "STAGING"):
            if f"${var}" not in blob and f"${{{var}}}" not in blob:
                continue
            assert step.get("env"), (
                f"{step.get('name')} references ${var} but declares no step-level "
                "env; the variable would be unset and `set -u` would abort the step"
            )
            assert var in step["env"], (
                f"{step.get('name')} references ${var} but does not declare it"
            )
