"""Research ledger: the append-only record of EVERY experiment ever run.

Multiple-testing corrections are only defensible if N counts the real search.
An 85-candidate pool understates it when the project also explored equal vs
inverse-vol weighting, tilt on/off, crisis thresholds, throttle variants,
band widths, execution models, cost models... Each of those is a draw from
the same multiple-testing lottery, whether or not it survived into the
headline configuration.

Rules enforced by design:
- Append-only JSONL. There is no update or delete API — failed experiments
  stay in the file forever. Tampering with history breaks id monotonicity
  and the recorded running hash chain.
- Every entry records: hypothesis, full configuration, primary metric,
  numeric result, accepted/rejected, and provenance (git commit).
- Backfilled entries (from git history/README tables, before the ledger
  existed) carry "backfilled": true and their original commit — clearly
  distinguished from prospectively logged work.

The deflated-Sharpe bridge (`deflated_sharpe_against_ledger`) counts ALL
strategy/portfolio/execution entries as trials, so corrections reflect the
whole research program rather than one final selection grid.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LEDGER = Path("research_ledger.jsonl")

# Categories that constitute SEARCH for the headline configuration. These are
# what multiple-testing corrections must count. ("methodology" entries — new
# measurement tools — are recorded but excluded: building a ruler is not a
# draw from the alpha lottery.)
SEARCH_CATEGORIES = ("strategy", "portfolio", "execution", "universe")
ALL_CATEGORIES = SEARCH_CATEGORIES + ("methodology",)

_REQUIRED_FIELDS = (
    "id", "timestamp", "category", "hypothesis",
    "configuration", "primaryMetric", "result", "accepted",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def entry_hash(entry: dict) -> str:
    blob = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_entry(entry: dict) -> None:
    missing = [f for f in _REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"ledger entry missing field(s): {sorted(missing)}")
    if entry["category"] not in ALL_CATEGORIES:
        raise ValueError(f"unknown category {entry['category']!r}; use one of {list(ALL_CATEGORIES)}")
    if not isinstance(entry["accepted"], bool):
        raise ValueError("accepted must be a boolean")
    if not isinstance(entry["result"], (int, float)):
        raise ValueError("result must be numeric")


def load_entries(path: str | Path = DEFAULT_LEDGER) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line after a crash
    return out


def verify_chain(entries: list[dict]) -> None:
    """Ids strictly increasing by 1, timestamps non-decreasing, every entry's
    content hash matches a recomputation (so ANY field edit is detected), and
    each `prev_hash` links to the prior entry's stored hash."""
    prev_hash = None
    prev_id = 0
    prev_ts = ""
    for e in entries:
        if e["id"] != prev_id + 1:
            raise ValueError(f"ledger gap/duplicate at id {e['id']} (expected {prev_id + 1}) — history was altered")
        if e["timestamp"] < prev_ts:
            raise ValueError(f"ledger timestamp regression at id {e['id']}")
        stored = e.get("hash")
        recomputed = entry_hash({k: v for k, v in e.items() if k != "hash"})
        if stored != recomputed:
            raise ValueError(
                f"ledger content hash mismatch at id {e['id']} — an entry was edited after writing"
            )
        if e.get("prev_hash") not in (None, prev_hash):
            raise ValueError(f"ledger hash-chain break at id {e['id']} — an entry was removed")
        prev_id = e["id"]
        prev_ts = e["timestamp"]
        prev_hash = stored


def append_entry(
    path: str | Path,
    *,
    category: str,
    hypothesis: str,
    configuration: dict | str,
    primaryMetric: str,
    result: float,
    accepted: bool,
    source_commit: str | None = None,
    backfilled: bool = False,
    timestamp: str | None = None,
) -> dict:
    """Append one experiment. Ids continue from the existing file; each entry
    carries the hash of its predecessor so edits/deletions are detectable.

    `timestamp` overrides the wall clock — used ONLY by the historical
    backfill seeder, and always applied BEFORE the content hash is computed
    so the seal stays valid."""
    entries = load_entries(path)
    entry = {
        "id": (entries[-1]["id"] + 1) if entries else 1,
        "timestamp": timestamp or _now(),
        "category": category,
        "hypothesis": hypothesis,
        "configuration": configuration,
        "primaryMetric": primaryMetric,
        "result": float(result),
        "accepted": accepted,
    }
    if backfilled:
        entry["backfilled"] = True
    if source_commit:
        entry["source_commit"] = source_commit
    validate_entry(entry)
    if entries:
        entry["prev_hash"] = entries[-1].get("hash")
    entry["hash"] = entry_hash({k: v for k, v in entry.items()})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


# ---------------------------------------------------------------------------
# Reporting & the multiple-testing bridge
# ---------------------------------------------------------------------------

def summarize(entries: list[dict]) -> dict:
    by_cat: dict[str, int] = {c: 0 for c in ALL_CATEGORIES}
    accepted_by_cat: dict[str, int] = {c: 0 for c in ALL_CATEGORIES}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        if e["accepted"]:
            accepted_by_cat[e["category"]] = accepted_by_cat.get(e["category"], 0) + 1
    search_total = sum(by_cat[c] for c in SEARCH_CATEGORIES)
    sharpe_trials = [
        float(e["result"]) for e in entries
        if e["category"] in SEARCH_CATEGORIES and e["primaryMetric"].lower().replace("-", " ") in
        ("oos sharpe", "sharpe", "annualized sharpe")
    ]
    return {
        "total_entries": len(entries),
        "by_category": by_cat,
        "accepted_by_category": accepted_by_cat,
        "rejected": len(entries) - sum(accepted_by_cat.values()),
        "search_categories": list(SEARCH_CATEGORIES),
        "recommended_trial_count": max(search_total, 1),
        "sharpe_valued_trials": len(sharpe_trials),
        "trial_sharpes": sharpe_trials,
    }


def recommended_trial_count(path: str | Path = DEFAULT_LEDGER) -> int | None:
    """N for multiple-testing corrections: every strategy / portfolio /
    execution / universe experiment in the ledger. None when no ledger exists."""
    p = Path(path)
    if not p.exists():
        return None
    return summarize(load_entries(p))["recommended_trial_count"]


def ledger_fingerprint(path: str | Path = DEFAULT_LEDGER) -> tuple[int, str] | None:
    """(entry_count, sha256 of file bytes) to pin inside a freeze manifest."""
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    return (raw.count(b"\n"), hashlib.sha256(raw).hexdigest())


def deflated_sharpe_against_ledger(
    returns: list[float],
    path: str | Path = DEFAULT_LEDGER,
    periods_per_year: int = 365,
) -> dict:
    """DSR computed against the LEDGER's total experiment count.

    Trial-Sharpe variance uses the Sharpe-valued results recorded in the
    ledger (portfolio/execution experiments typically report OOS Sharpe);
    strategy-pool Sharpes from walk-forward selection should be appended to
    that sample by callers who have them (see `validate`). Falls back to the
    pool-only correction when the ledger is absent."""
    from .stats_validation import dsr, expected_max_sharpe_annual

    entries = load_entries(path)
    if not entries:
        return {"available": False, "reason": "no research ledger"}
    summary = summarize(entries)
    n_trials = summary["recommended_trial_count"]
    trial_sharpes = summary["trial_sharpes"]
    benchmark = expected_max_sharpe_annual(trial_sharpes, n_trials)
    d = dsr(returns, trial_sharpes or [0.0], n_trials, periods_per_year)
    return {
        "available": True,
        "n_trials": n_trials,
        "benchmark_sharpe": benchmark,
        "dsr": d,
        "note": (
            f"deflated against {n_trials} ledger experiments "
            f"({summary['sharpe_valued_trials']} Sharpe-valued)"
        ),
    }
