"""Provenance: canonical evidence must name the source that produced it.

These assert the guarantees, not the implementation: a dirty tree is refused,
an override exists and is explicit, and a record cannot claim a commit or
fingerprint that the tree does not have.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from bot.provenance import (
    DirtyTreeError,
    ProvenanceMismatch,
    assert_record_matches_source,
    git_state,
    require_clean_tree,
    source_fingerprint,
)

REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "checkout", "-B", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "a.txt").write_text("one", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "first")
    return r


def _args(**kw) -> argparse.Namespace:
    base = {"run_id": "scratch-1", "allow_dirty_tree": False, "canonical": False}
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# dirty-tree refusal
# ---------------------------------------------------------------------------

def test_clean_tree_passes(repo: Path) -> None:
    state = require_clean_tree(repo)
    assert state.dirty is False
    assert state.head_commit


def test_dirty_tree_is_refused_for_canonical_work(repo: Path) -> None:
    (repo / "a.txt").write_text("changed but uncommitted", encoding="utf-8")
    with pytest.raises(DirtyTreeError) as exc:
        require_clean_tree(repo, what="canonical run")
    message = str(exc.value)
    assert "DIRTY" in message
    assert "a.txt" in message
    assert "--allow-dirty-tree" in message


def test_untracked_file_also_counts_as_dirty(repo: Path) -> None:
    (repo / "brand_new.txt").write_text("x", encoding="utf-8")
    state = git_state(repo)
    assert state.dirty is True
    assert "brand_new.txt" in state.dirty_paths


def test_staged_but_uncommitted_counts_as_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("staged", encoding="utf-8")
    _git(repo, "add", "-A")
    with pytest.raises(DirtyTreeError):
        require_clean_tree(repo)


def test_explicit_override_allows_a_non_canonical_run(repo: Path) -> None:
    """Exploratory runs are legitimate; they just must not be canonical."""
    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    state = require_clean_tree(repo, allow_dirty=True, what="canonical run")
    assert state.dirty is True
    assert "a.txt" in state.dirty_paths


def test_non_git_directory_is_reported_not_crashed(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    state = git_state(plain)
    assert state.is_repo is False
    assert state.head_commit is None
    assert "not a git repository" in state.describe()


# ---------------------------------------------------------------------------
# commit / source agreement
# ---------------------------------------------------------------------------

def _record(root: Path, **env_overrides) -> dict:
    """A record that honestly describes `root`, with optional disagreements."""
    fp = source_fingerprint(root)
    env = {
        "git_commit": git_state(root).head_commit,
        "code_fingerprint": {"algo": "sha256-lf-v1", "sha256": fp.code_sha256},
        "source_clean": True,
    }
    env.update(env_overrides)
    return {"results": {"environment": env}}


def test_record_agreeing_with_the_tree_is_accepted() -> None:
    assert_record_matches_source(_record(REPO), REPO)


def test_record_claiming_a_different_commit_is_rejected() -> None:
    rec = _record(REPO, git_commit="0" * 40)
    with pytest.raises(ProvenanceMismatch) as exc:
        assert_record_matches_source(rec, REPO)
    assert "commit" in str(exc.value)


def test_record_claiming_a_different_source_fingerprint_is_rejected() -> None:
    rec = _record(REPO, code_fingerprint={"algo": "sha256-lf-v1", "sha256": "f" * 64})
    with pytest.raises(ProvenanceMismatch) as exc:
        assert_record_matches_source(rec, REPO)
    assert "fingerprint" in str(exc.value)


def test_record_with_no_commit_is_rejected() -> None:
    rec = _record(REPO, git_commit=None)
    with pytest.raises(ProvenanceMismatch) as exc:
        assert_record_matches_source(rec, REPO)
    assert "git_commit" in str(exc.value)


def test_record_from_a_dirty_tree_is_not_canonical_evidence() -> None:
    rec = _record(REPO, source_clean=False)
    with pytest.raises(ProvenanceMismatch) as exc:
        assert_record_matches_source(rec, REPO)
    assert "DIRTY" in str(exc.value)


def test_record_with_no_environment_block_is_rejected() -> None:
    with pytest.raises(ProvenanceMismatch):
        assert_record_matches_source({"results": {}}, REPO)


# ---------------------------------------------------------------------------
# the CLI-level guarantee: canonical runs refuse a dirty tree, others do not
# ---------------------------------------------------------------------------


def test_canonical_run_id_is_refused_on_a_dirty_tree(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.__main__ import _environment_block

    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    monkeypatch.chdir(repo)
    with pytest.raises(DirtyTreeError):
        _environment_block(_args(run_id="canonical-v3"))


def test_explicit_canonical_flag_is_also_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.__main__ import _environment_block

    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    monkeypatch.chdir(repo)
    with pytest.raises(DirtyTreeError):
        _environment_block(_args(run_id="scratch-1", canonical=True))


def test_non_canonical_run_is_allowed_and_says_it_was_dirty(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.__main__ import _environment_block

    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    monkeypatch.chdir(repo)
    env = _environment_block(_args(run_id="scratch-1"))
    assert env["source_clean"] is False
    assert env["is_canonical"] is False
    assert "a.txt" in env["dirty_paths"]
    assert env["git_commit"] == git_state(repo).head_commit


def test_override_lets_a_canonical_run_proceed_but_marks_it_not_clean(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch exists, and the record never pretends otherwise."""
    from bot.__main__ import _environment_block

    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    monkeypatch.chdir(repo)
    env = _environment_block(_args(run_id="canonical-v3", allow_dirty_tree=True))
    assert env["source_clean"] is False
    assert env["is_canonical"] is True


# ---------------------------------------------------------------------------
# fingerprints and the committed freeze
# ---------------------------------------------------------------------------


def test_fingerprint_records_head_and_module_hashes() -> None:
    fp = source_fingerprint(REPO)
    assert len(fp.code_sha256) == 64
    assert "bot/strategy.py" in fp.module_hashes
    assert "bot/canonical_identity.py" in fp.module_hashes
    assert fp.as_dict()["algo"] == "sha256-lf-v1"


def test_committed_freeze_names_a_real_commit() -> None:
    import json

    manifest = json.loads((REPO / "freeze.json").read_text(encoding="utf-8"))
    commit = manifest["git_commit_at_freeze"]
    assert commit and len(commit) == 40
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, f"frozen commit {commit} is not present in this repository"
