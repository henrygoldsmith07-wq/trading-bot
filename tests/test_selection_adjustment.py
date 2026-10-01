"""Multiple-testing treatment must reflect the search that really happened.

The canonical pipeline used to compute every rule DSR at n_trials=1, which
corrects for nothing, then print values like 0.999 beside a research ledger
holding 35 recorded experiments. These tests pin the corrected behaviour.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bot.research_ledger import append_entry, load_entries
from bot.stats_validation import format_dsr, selection_adjusted_stats

ROOT = Path(__file__).resolve().parents[1]
REAL_LEDGER = ROOT / "research_ledger.jsonl"


def _returns(n: int = 730, drift: float = 0.0006, vol: float = 0.012, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(drift, vol) for _ in range(n)]


def _write_ledger(
    path: Path,
    n_entries: int,
    sharpes: list[float] | None = None,
    metric: str = "OOS Sharpe",
) -> None:
    """Build a valid hash-chained ledger of n_entries search experiments."""
    path.write_text("", encoding="utf-8")
    for i in range(n_entries):
        result = sharpes[i] if sharpes and i < len(sharpes) else 0.4
        append_entry(
            path,
            category="portfolio",
            hypothesis=f"overlay variant {i} improves risk-adjusted return",
            configuration={"variant": i},
            primaryMetric=metric,
            result=result,
            accepted=(i == 0),
            source_commit="0" * 40,
        )


# ---------------------------------------------------------------------------
# the trial count must propagate
# ---------------------------------------------------------------------------

def test_selected_row_deflates_against_the_ledger_trial_count(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 40, sharpes=[0.2, 0.5, 0.9, 1.4])
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), pool_size=12)
    assert stats["dsr_available"] is True
    assert stats["n_trials"] == 40, "the ledger experiment count must be the trial count"
    assert stats["dsr"] is not None


def test_trial_count_never_falls_below_the_candidate_pool(tmp_path: Path) -> None:
    """A wide pool searched in this run counts even with a small ledger."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 3, sharpes=[0.3, 0.8])
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), pool_size=85)
    assert stats["n_trials"] == 85


def test_dsr_is_lower_than_psr_once_a_real_correction_applies(tmp_path: Path) -> None:
    """Deflation must pull the number down, not merely relabel it."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 60, sharpes=[0.2, 0.4, 0.7, 1.1, 1.6])
    rets = _returns()
    stats = selection_adjusted_stats(rets, ledger_path=str(ledger), pool_size=20)
    assert stats["dsr_available"] is True
    assert stats["dsr"] < stats["psr"]


def test_a_wider_search_never_produces_a_better_dsr(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 50, sharpes=[0.3, 0.7, 1.2])
    rets = _returns()
    narrow = selection_adjusted_stats(rets, ledger_path=str(ledger), pool_size=1)
    wide = selection_adjusted_stats(rets, ledger_path=str(ledger), pool_size=900)
    assert wide["n_trials"] > narrow["n_trials"]
    assert wide["dsr"] < narrow["dsr"]


def test_trial_count_sources_are_reported(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 9, sharpes=[0.4, 1.0])
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), pool_size=17)
    joined = " ".join(stats["trial_count_sources"])
    assert "research ledger" in joined
    assert "candidate pool" in joined


# ---------------------------------------------------------------------------
# pre-specified rows stay distinguishable from selected rows
# ---------------------------------------------------------------------------

def test_prespecified_row_uses_a_trial_count_of_one(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 50, sharpes=[0.3, 0.9])
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), prespecified=True)
    assert stats["prespecified"] is True
    assert stats["n_trials"] == 1
    assert stats["dsr_available"] is True


def test_selected_and_prespecified_rows_are_not_the_same_claim(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 50, sharpes=[0.3, 0.9])
    rets = _returns()
    selected = selection_adjusted_stats(rets, ledger_path=str(ledger), prespecified=False)
    pre = selection_adjusted_stats(rets, ledger_path=str(ledger), prespecified=True)
    assert selected["dsr"] < pre["dsr"], "a selected row must never score above a pre-specified one"


# ---------------------------------------------------------------------------
# the misleading one-trial case must be refused, not approximated
# ---------------------------------------------------------------------------

def test_no_search_history_means_dsr_is_unavailable(tmp_path: Path) -> None:
    stats = selection_adjusted_stats(_returns(), ledger_path=str(tmp_path / "absent.jsonl"), pool_size=0)
    assert stats["dsr_available"] is False
    assert stats["dsr"] is None, "an uncorrected value must never be passed off as a DSR"
    assert stats["dsr_unavailable_reason"]


def test_ledger_without_sharpe_dispersion_is_unavailable(tmp_path: Path) -> None:
    """One Sharpe-valued experiment cannot estimate cross-trial dispersion."""
    ledger = tmp_path / "ledger.jsonl"
    # five real experiments, none of them reporting a Sharpe
    _write_ledger(ledger, 5, sharpes=[0.5], metric="Max drawdown")
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), pool_size=0)
    assert stats["dsr_available"] is False
    assert stats["dsr"] is None
    assert "Sharpe-valued" in str(stats["dsr_unavailable_reason"])


def test_corrupt_ledger_fails_closed_rather_than_defaulting_to_one(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    entries = json.loads(json.dumps(load_entries(REAL_LEDGER)))
    entries[0]["result"] = 99.0  # tampered: breaks the hash chain
    ledger.write_text("".join(json.dumps(e, sort_keys=True) + chr(10) for e in entries), encoding="utf-8")
    stats = selection_adjusted_stats(_returns(), ledger_path=str(ledger), pool_size=0)
    assert stats["dsr_available"] is False
    assert stats["dsr"] is None


def test_format_dsr_never_prints_a_number_when_unavailable(tmp_path: Path) -> None:
    stats = selection_adjusted_stats(_returns(), ledger_path=str(tmp_path / "absent.jsonl"))
    rendered = format_dsr(stats)
    assert rendered.startswith("DSR n/a")
    assert "0.9" not in rendered


def test_format_dsr_shows_the_trial_count_when_available(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, 22, sharpes=[0.2, 0.9])
    rendered = format_dsr(selection_adjusted_stats(_returns(), ledger_path=str(ledger)))
    assert "N=22" in rendered
    assert "n/a" not in rendered


# ---------------------------------------------------------------------------
# the real committed ledger
# ---------------------------------------------------------------------------

def test_committed_ledger_supports_a_real_correction() -> None:
    stats = selection_adjusted_stats(_returns(), ledger_path=str(REAL_LEDGER), pool_size=18)
    assert stats["dsr_available"] is True
    assert stats["n_trials"] is not None and stats["n_trials"] > 1
