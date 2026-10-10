"""Independent tests for the forward-evidence transfer.

These assert behaviour, not implementation shape. In particular
"test_legacy_checkout_B_fails_on_dirty_evidence_tree" actually runs git and
proves the failure that lost forward days, so the root cause stays
documented even after the workflow is rewritten.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bot.forward_evidence import (
    COST_LOG,
    FORWARD_LOG,
    STAGING_MANIFEST,
    UNIVERSE_LOG,
    EvidenceIntegrityError,
    FreezeChangedError,
    merge_cost_observations,
    merge_forward_entries,
    merge_universe_snapshots,
    read_jsonl,
    transfer_evidence,
    validate_log,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _manifest(commit: str, code: str = "code0", date: str = "2026-01-02") -> dict:
    return {
        "git_commit_at_freeze": commit,
        "code_sha256": code,
        "config_sha256": "cfg0",
        "frozen_at_date": date,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo holding freeze.json + evidence tapes, committed on main."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "checkout", "-B", "main")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "user.name", "test")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "freeze.json").write_text(json.dumps(_manifest("aaaaaaa")), encoding="utf-8")
    _write_jsonl(r / FORWARD_LOG, [{"date": "2026-01-03", "port_ret": 0.001}])
    _write_jsonl(r / UNIVERSE_LOG, [{"date": "2026-01-03", "universe": ["AAA"]}])
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")
    return r


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    s = tmp_path / "staging"
    s.mkdir()
    return s


def _stage(staging: Path, *, manifest: dict, forward: list[dict] | None = None,
          universe: list[dict] | None = None, costs: str | None = None) -> None:
    (staging / STAGING_MANIFEST).write_text(json.dumps({"produced_under_freeze": manifest}), encoding="utf-8")
    if forward is not None:
        _write_jsonl(staging / FORWARD_LOG, forward)
    if universe is not None:
        _write_jsonl(staging / UNIVERSE_LOG, universe)
    if costs is not None:
        (staging / COST_LOG).write_text(costs, encoding="utf-8")


# ---------------------------------------------------------------------------
# the regression that lost forward days
# ---------------------------------------------------------------------------

def test_legacy_checkout_B_fails_on_dirty_evidence_tree(tmp_path: Path) -> None:
    """Reproduce the scheduled run's real failure.

    The old workflow executed the frozen commit in the SAME tree, produced
    evidence there, then tried to return to main with
    "git checkout -B main origin/main". Git refuses when the working tree
    has local edits to files that differ between the two refs, so the
    already-computed paper day was thrown away.
    """
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "checkout", "-B", "main")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "user.name", "test")
    _git(r, "config", "commit.gpgsign", "false")
    _write_jsonl(r / FORWARD_LOG, [{"date": "2026-01-01"}])
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "A")
    frozen = _git(r, "rev-parse", "HEAD").stdout.strip()
    # main advances while the frozen run executes
    _write_jsonl(r / FORWARD_LOG, [{"date": "2026-01-01"}, {"date": "2026-01-02"}])
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "B")
    # the runner switches to the frozen commit and produces evidence in-tree
    assert _git(r, "checkout", "-q", frozen).returncode == 0
    _write_jsonl(r / FORWARD_LOG, [{"date": "2026-01-01"}, {"date": "2026-01-05"}])
    (r / "freeze.json").write_text(json.dumps(_manifest(frozen)), encoding="utf-8")
    _write_jsonl(r / UNIVERSE_LOG, [{"date": "2026-01-05", "universe": ["AAA"]}])

    dirty = _git(r, "status", "--porcelain")
    assert dirty.stdout.strip(), "expected a dirty tree before the return-to-main step"
    res = _git(r, "checkout", "-B", "main", "main")
    assert res.returncode != 0, "expected the legacy return-to-main step to fail"
    assert "forward_log.jsonl" in (res.stdout + res.stderr)

def test_isolated_worktree_keeps_main_clean_so_the_switch_cannot_fail(tmp_path: Path) -> None:
    """The architectural fix: frozen code runs in its own worktree."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "checkout", "-B", "main")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "user.name", "test")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "freeze.json").write_text(json.dumps(_manifest("aaaaaaa")), encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "A")
    frozen = _git(r, "rev-parse", "HEAD").stdout.strip()
    _write_jsonl(r / FORWARD_LOG, [{"date": "2026-01-02"}])
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "B")

    wt = tmp_path / "frozen"
    assert _git(r, "worktree", "add", "-q", "--detach", str(wt), frozen).returncode == 0
    # the frozen run produces evidence in the worktree only
    _write_jsonl(wt / FORWARD_LOG, [{"date": "2026-01-09"}])
    (wt / "freeze.json").write_text(json.dumps(_manifest(frozen)), encoding="utf-8")

    assert _git(r, "status", "--porcelain").stdout.strip() == "", "main must stay clean"
    switched = _git(r, "checkout", "-B", "main", "main")
    assert switched.returncode == 0, "returning to main must not fail on a clean tree"
    _git(r, "worktree", "remove", "--force", str(wt))


# ---------------------------------------------------------------------------
# merge semantics
# ---------------------------------------------------------------------------

def test_duplicate_date_is_not_appended_twice() -> None:
    merged, report = merge_forward_entries(
        [{"date": "2026-02-01", "port_ret": 0.01}],
        [{"date": "2026-02-01", "port_ret": 0.99}],
    )
    assert report.forward_added == []
    assert report.forward_duplicate == ["2026-02-01"]
    assert len(merged) == 1
    assert merged[0]["port_ret"] == 0.01, "the committed row is the row of record"


def test_existing_rows_are_never_rewritten() -> None:
    existing = [{"date": "2026-02-01", "port_ret": 0.01}]
    merged, _ = merge_forward_entries(existing, [{"date": "2026-02-02"}])
    assert merged[0] == existing[0]


def test_rows_on_or_before_the_freeze_date_are_dropped() -> None:
    merged, report = merge_forward_entries(
        [],
        [{"date": "2026-01-01"}, {"date": "2026-01-02"}, {"date": "2026-01-03"}],
        cutoff_date="2026-01-02",
    )
    assert [e["date"] for e in merged] == ["2026-01-03"]
    assert report.forward_pre_freeze_dropped == ["2026-01-01", "2026-01-02"]


def test_merge_is_idempotent() -> None:
    incoming = [{"date": "2026-02-02"}]
    once, r1 = merge_forward_entries([{"date": "2026-02-01"}], incoming)
    twice, r2 = merge_forward_entries(once, incoming)
    assert twice == once
    assert r2.forward_added == []
    assert r2.forward_duplicate == ["2026-02-02"]


def test_merge_output_is_sorted_by_date() -> None:
    merged, _ = merge_forward_entries([], [{"date": "2026-02-03"}, {"date": "2026-02-01"}])
    assert [e["date"] for e in merged] == ["2026-02-01", "2026-02-03"]


def test_universe_snapshot_duplicate_same_content_is_a_noop() -> None:
    merged, report = merge_universe_snapshots(
        [{"date": "2026-02-01", "universe": ["A"], "generated_at": "t1"}],
        [{"date": "2026-02-01", "universe": ["A"], "generated_at": "t2"}],
    )
    assert report.universe_duplicate == ["2026-02-01"]
    assert len(merged) == 1


def test_universe_snapshot_conflict_fails_closed() -> None:
    """Two different membership claims for one date cannot both be evidence."""
    with pytest.raises(EvidenceIntegrityError) as exc:
        merge_universe_snapshots(
            [{"date": "2026-02-01", "universe": ["A"]}],
            [{"date": "2026-02-01", "universe": ["B"]}],
        )
    assert "2026-02-01" in str(exc.value)


def test_cost_tape_appends_only_new_lines() -> None:
    text, added, dupes = merge_cost_observations("{\"a\":1}\n", "{\"a\":1}\n{\"b\":2}\n")
    assert added == 1
    assert dupes == 1
    assert text == "{\"a\":1}\n{\"b\":2}\n"


def test_malformed_jsonl_fails_closed_rather_than_skipping(tmp_path: Path) -> None:
    bad = tmp_path / "tape.jsonl"
    bad.write_text('{"date": "2026-01-01"}\n{"date": "2026-01-02"\n', encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError) as exc:
        read_jsonl(bad)
    assert "line 2" in str(exc.value)


def test_entry_without_date_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "tape.jsonl"
    bad.write_text(json.dumps({"port_ret": 0.1}) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError):
        read_jsonl(bad)


def test_validate_log_reports_duplicates() -> None:
    report = validate_log([{"date": "b"}, {"date": "a"}, {"date": "b"}], kind="forward")
    assert report["duplicate_dates"] == ["b"]
    assert report["rows"] == 3
    assert report["unique_dates"] == 2
    assert report["sorted"] is False


# ---------------------------------------------------------------------------
# end-to-end transfer against a real repo
# ---------------------------------------------------------------------------

def test_transfer_lands_evidence_and_leaves_main_clean(repo: Path, staging: Path) -> None:
    _stage(staging, manifest=_manifest("aaaaaaa"), forward=[{"date": "2026-01-06"}])
    report = transfer_evidence(staging, repo)
    assert report.forward_added == ["2026-01-06"]
    assert [e["date"] for e in read_jsonl(repo / FORWARD_LOG)] == ["2026-01-03", "2026-01-06"]


def test_transfer_never_overwrites_a_concurrent_main_update(repo: Path, staging: Path) -> None:
    """A commit landing on main during the run must survive the merge."""
    _stage(staging, manifest=_manifest("aaaaaaa"), forward=[{"date": "2026-01-06"}])
    # someone else commits while the frozen step was running
    _write_jsonl(repo / FORWARD_LOG, [{"date": "2026-01-03"}, {"date": "2026-01-07"}])
    report = transfer_evidence(staging, repo)
    assert report.forward_added == ["2026-01-06"]
    dates = [e["date"] for e in read_jsonl(repo / FORWARD_LOG)]
    assert dates == ["2026-01-03", "2026-01-06", "2026-01-07"]


def test_transfer_is_idempotent_across_runs(repo: Path, staging: Path) -> None:
    _stage(staging, manifest=_manifest("aaaaaaa"), forward=[{"date": "2026-01-06"}])
    first = transfer_evidence(staging, repo)
    second = transfer_evidence(staging, repo)
    assert first.forward_added == ["2026-01-06"]
    assert second.forward_added == []
    assert second.forward_duplicate == ["2026-01-06"]
    assert len(read_jsonl(repo / FORWARD_LOG)) == 2


def test_transfer_refuses_when_the_freeze_changed_mid_run(repo: Path, staging: Path) -> None:
    _stage(staging, manifest=_manifest("bbbbbbb"), forward=[{"date": "2026-01-06"}])
    before = (repo / FORWARD_LOG).read_text(encoding="utf-8")
    with pytest.raises(FreezeChangedError) as exc:
        transfer_evidence(staging, repo)
    assert "freeze changed" in str(exc.value)
    assert (repo / FORWARD_LOG).read_text(encoding="utf-8") == before, "must not partially apply"


def test_interrupted_run_keeps_staged_evidence_for_retry(repo: Path, staging: Path) -> None:
    """A computed paper day must survive a crash before the merge."""
    _stage(staging, manifest=_manifest("aaaaaaa"), forward=[{"date": "2026-01-06"}])
    assert staging.joinpath(FORWARD_LOG).exists()
    # the merge never happened; the staged row is still there to be merged
    assert [e["date"] for e in read_jsonl(repo / FORWARD_LOG)] == ["2026-01-03"]
    report = transfer_evidence(staging, repo)
    assert report.forward_added == ["2026-01-06"]


def test_transfer_requires_a_declared_freeze_identity(repo: Path, staging: Path) -> None:
    _write_jsonl(staging / FORWARD_LOG, [{"date": "2026-01-06"}])
    with pytest.raises(EvidenceIntegrityError):
        transfer_evidence(staging, repo)


def test_transfer_preserves_universe_and_cost_tapes(repo: Path, staging: Path) -> None:
    _stage(
        staging,
        manifest=_manifest("aaaaaaa"),
        forward=[{"date": "2026-01-06"}],
        universe=[{"date": "2026-01-06", "universe": ["AAA", "BBB"]}],
        costs=json.dumps({"k": 1}) + "\n",
    )
    report = transfer_evidence(staging, repo)
    assert report.universe_added == ["2026-01-06"]
    assert report.cost_added == 1
    assert [e["date"] for e in read_jsonl(repo / UNIVERSE_LOG)] == ["2026-01-03", "2026-01-06"]
    assert (repo / COST_LOG).exists()


def test_transfer_clamps_the_freeze_boundary_against_main(repo: Path, staging: Path) -> None:
    _stage(staging, manifest=_manifest("aaaaaaa"), forward=[{"date": "2026-01-01"}, {"date": "2026-01-06"}])
    report = transfer_evidence(staging, repo)
    assert report.forward_pre_freeze_dropped == ["2026-01-01"]
    assert report.forward_added == ["2026-01-06"]


def test_missing_staging_directory_is_refused(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(EvidenceIntegrityError):
        transfer_evidence(tmp_path / "nope", repo)


# ---------------------------------------------------------------------------
# the workflow itself must keep the architecture, not just the module
# ---------------------------------------------------------------------------

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "scheduled-paper.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    import yaml

    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(workflow: dict) -> list[dict]:
    return workflow["jobs"]["forward-step"]["steps"]


def _step(workflow: dict, name: str) -> dict:
    for s in _steps(workflow):
        if s.get("name") == name:
            return s
    raise AssertionError(f"workflow has no step named {name!r}")


def _frozen_tree(workflow: dict) -> str:
    """Where the isolated frozen tree lives, read from the step that creates it.

    Read from the workflow rather than hard-coded: if the directory is renamed,
    these guards must follow it instead of silently passing against a path that
    nothing uses any more.
    """
    return _step(workflow, "Create isolated worktree at the frozen commit")["env"]["FROZEN_TREE"]


def _staging(workflow: dict) -> str:
    """Where staged evidence lands, read from the step that writes it."""
    return _step(workflow, "Stage evidence for transfer to main")["env"]["STAGING"]


def test_frozen_code_runs_in_an_isolated_worktree(workflow: dict) -> None:
    worktree = _step(workflow, "Create isolated worktree at the frozen commit")
    assert "git worktree add" in worktree["run"]
    forward = _step(workflow, "Advance the prospective paper portfolio one day")
    # the forward day must run in the worktree, never in the main checkout
    assert forward.get("working-directory") == _frozen_tree(workflow)


def test_anything_writing_the_evidence_tapes_runs_in_the_worktree(workflow: dict) -> None:
    """The main checkout must never execute a command that writes the tapes.

    This is the invariant whose violation caused the original failure: the
    forward step used to run in the main tree, which is exactly what made the
    later "checkout -B main" impossible.
    """
    writers = ("forward --step", "universe-snapshot")
    seen = 0
    for s in _steps(workflow):
        run = str(s.get("run", ""))
        if not any(w in run for w in writers):
            continue
        seen += 1
        assert s.get("working-directory") == _frozen_tree(workflow), (
            f"{s.get('name')} writes evidence outside the isolated worktree"
        )
    assert seen >= 2, "expected both the forward step and the universe snapshot to be guarded"


def test_evidence_writers_never_target_the_main_checkout(workflow: dict) -> None:
    for s in _steps(workflow):
        run = str(s.get("run", ""))
        if "forward --step" in run or "universe-snapshot" in run:
            assert "--log forward_log.jsonl" not in run.replace('"$LOG_FILE"', "")


def test_evidence_is_staged_before_any_merge(workflow: dict) -> None:
    names = [s.get("name") for s in _steps(workflow)]
    assert names.index("Stage evidence for transfer to main") < names.index("Merge staged evidence into main and push")


def test_merge_goes_through_the_tested_module_not_a_heredoc(workflow: dict) -> None:
    merge = _step(workflow, "Merge staged evidence into main and push")
    assert "python -m bot forward-evidence merge" in merge["run"]
    assert "<<" not in merge["run"], "evidence merging must not be inline shell heredoc"


def test_merge_retries_onto_a_moving_main(workflow: dict) -> None:
    merge = _step(workflow, "Merge staged evidence into main and push")
    assert "for attempt in" in merge["run"]
    assert "git fetch" in merge["run"]
    # never force: overwriting main would destroy concurrent evidence
    assert "push --force" not in merge["run"]


def test_staged_evidence_is_uploaded_even_when_later_steps_fail(workflow: dict) -> None:
    upload = _step(workflow, "Upload evidence artifacts")
    assert upload.get("if") == "always()"
    # must upload the SAME directory the staging step writes to, not merely a
    # path that happens to contain the substring "STAGING"
    assert _staging(workflow) in str(upload["with"]["path"])


def test_freeze_is_verified_before_any_forward_day_is_traded(workflow: dict) -> None:
    names = [s.get("name") for s in _steps(workflow)]
    assert names.index("Verify running code matches the freeze") < names.index(
        "Advance the prospective paper portfolio one day"
    )


def test_a_run_that_does_not_trade_never_claims_success_silently(workflow: dict) -> None:
    """`trading=false` must be a visible, non-green outcome.

    The tape stopped advancing while runs kept returning success: the resolver
    fell through to the legacy branch (the frozen code predates `--as-of-date`),
    emitted `trading=false`, skipped every evidence step, and exited 0. A green
    run that produces no evidence is worse than a red one, because nothing
    prompts anyone to look.
    """
    resolve = _step(workflow, "Resolve the market session date")
    run = resolve["run"]

    # The decision must be surfaced, not swallowed.
    assert "::notice::" in run, "skipping a day must emit a visible notice"

    # And the forward step must be gated on it, so a non-trading run cannot
    # silently fall through into the evidence-producing steps.
    forward = _step(workflow, "Advance the prospective paper portfolio one day")
    assert "steps.session.outputs.trading == 'true'" in str(forward.get("if"))


def test_the_two_schedules_cannot_both_be_no_ops(workflow: dict) -> None:
    """At least one cadence must actually trade.

    The dual-schedule resolver exists so legacy and new frozen code each get a
    working cadence. If neither branch can reach `trading=true` for the
    schedules this workflow declares, the experiment can never advance -- which
    is the state the repo sat in while reporting success.
    """
    resolve = _step(workflow, "Resolve the market session date")
    run = resolve["run"]

    # The schedules the resolver branches on must match the declared crons.
    declared = {c["cron"] for c in workflow[True]["schedule"]}
    for cron in ("30 0 * * *", "45 21 * * 1-5"):
        assert cron in declared, f"resolver branches on {cron}, which is not scheduled"
        assert cron in run, f"scheduled cron {cron} is not handled by the resolver"

    # Each branch must be able to set trading=true.
    assert run.count("trading=true") >= 2, "both cadences must be able to trade"

