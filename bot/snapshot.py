"""Reproducible benchmark snapshots.

A benchmark number that cannot be re-derived is an anecdote. A snapshot pins:
- the exact input datasets (sha256 of each candle series),
- the configuration (seeds, frictions, folds),
- the resulting metrics.
`verify_snapshot` recomputes the metrics later (same seed, same data hashes)
and reports any drift beyond floating-point tolerance — so CI or a reviewer
can detect nondeterminism or silent data changes immediately.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

SNAPSHOT_VERSION = 1


def dataset_hash(candles: list[dict]) -> str:
    """Stable hash of a candle series (order-sensitive by design)."""
    canonical = json.dumps(candles, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def make_snapshot(
    name: str,
    config: dict,
    data_hashes: dict[str, str],
    metrics: dict[str, float],
    seed: int | None = None,
    git_commit: str | None = None,
    created: str | None = None,
) -> dict:
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "name": name,
        "created": created or datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "seed": seed,
        "config": config,
        "data_hashes": data_hashes,
        "metrics": metrics,
    }


def write_snapshot(snapshot: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return p


def load_snapshot(path: str | Path) -> dict:
    snap = json.loads(Path(path).read_text())
    if snap.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version {snap.get('snapshot_version')!r}")
    return snap


def verify_snapshot(
    path: str | Path,
    current_metrics: dict[str, float],
    current_data_hashes: dict[str, str] | None = None,
    atol: float = 1e-9,
) -> dict:
    """Compare freshly computed metrics against a stored snapshot.

    Returns {ok, metric_drifts: [{metric, expected, actual}], data_changed}.
    Non-finite values must match exactly; otherwise |diff| <= atol.
    """
    snap = load_snapshot(path)
    drifts = []
    for key, expected in snap["metrics"].items():
        actual = current_metrics.get(key)
        if actual is None:
            drifts.append({"metric": key, "expected": expected, "actual": None})
            continue
        same = (
            (math.isnan(expected) and math.isnan(actual))
            or (not math.isfinite(expected) and not math.isfinite(actual))
            or abs(expected - actual) <= atol
        )
        if not same:
            drifts.append({"metric": key, "expected": expected, "actual": actual})
    extra = sorted(set(current_metrics) - set(snap["metrics"]))
    for key in extra:
        drifts.append({"metric": key, "expected": None, "actual": current_metrics[key]})

    data_changed = []
    if current_data_hashes is not None:
        for ds, h in current_data_hashes.items():
            if snap["data_hashes"].get(ds) != h:
                data_changed.append(ds)

    return {
        "ok": not drifts and not data_changed,
        "metric_drifts": drifts,
        "data_changed": data_changed,
        "snapshot_name": snap["name"],
    }
