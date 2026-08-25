"""Run records: every serious benchmark writes run.json; reproduce replays it.

A `run.json` pins the full experiment context:
  - environment: python version, git commit, whole-code fingerprint, plus
    per-module hashes (strategy definitions / portfolio rules / universe)
  - parameters: the complete argparse namespace of the invocation
  - seeds: recorded even where the pipeline is deterministic
  - datasets: per-asset sha256, provider, download timestamp, start/end
  - results: every reported metric block + verdict

`python -m bot reproduce <run-id>` then:
  1. refuses unless the running code fingerprints match the record,
  2. refuses unless each frozen dataset is still in cache with the SAME
     sha256 (frozen source data required — no silent refreshes),
  3. re-executes the benchmark deterministically,
  4. compares every stored metric and reports PASS/FAIL with diffs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

RUNS_DIR = Path("runs")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def run_record_hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def save_run_record(results: dict, runs_dir: str | Path = RUNS_DIR, run_id: str | None = None) -> str:
    """Persist results as runs/<id>/run.json. Auto ids embed a UTC timestamp;
    an explicit `run_id` (e.g. "canonical-v1") names an authoritative record."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    assets = len(results.get("universe", []))
    mode = "compare"
    if run_id is None:
        run_id = f"{ts}-{mode}-{assets}a"
    record = {
        "run_id": run_id,
        "created_at": _now(),
        "results": results,
        "tolerance": {"rel": 1e-12, "abs": 1e-12},
    }
    record["record_sha256"] = run_record_hash(record)
    d = Path(runs_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(record, indent=2))
    return run_id


def load_run_record(run_id: str, runs_dir: str | Path = RUNS_DIR) -> dict:
    path = Path(runs_dir) / run_id / "run.json"
    record = json.loads(path.read_text())
    stored = record.get("record_sha256")
    if stored != run_record_hash(record):
        raise ValueError(f"run.json integrity failure for {run_id} — the record was edited")
    return record


def list_runs(runs_dir: str | Path = RUNS_DIR) -> list[dict]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        f = d / "run.json"
        if f.exists():
            try:
                r = json.loads(f.read_text())
                out.append({"run_id": r["run_id"], "created_at": r["created_at"],
                            "verdict": r["results"].get("verdict")})
            except (json.JSONDecodeError, KeyError):
                continue
    return out


# ---------------------------------------------------------------------------
# reproduction
# ---------------------------------------------------------------------------

class ReproduceRefused(Exception):
    """Raised when the environment or data cannot recreate the run."""


def check_environment(record: dict) -> None:
    env = record["results"]["environment"]
    from .identity import code_fingerprint, source_file_fingerprint

    current_fp = code_fingerprint()
    if current_fp != env["code_fingerprint"]["sha256"]:
        raise ReproduceRefused(
            "code fingerprint mismatch: the running implementation differs from "
            f"the recorded one ({env['code_fingerprint']['sha256'][:12]}… vs "
            f"{current_fp[:12]}…). Check out the recorded git commit to reproduce."
        )
    for label, paths in (
        ("strategy_definitions_hash", ["bot/strategy.py"]),
        ("portfolio_rules_hash", ["bot/portfolio_rules.py"]),
        ("universe_hash", ["bot/universe.py", "bot/universe_pit.py"]),
    ):
        current = source_file_fingerprint(paths)["combined"]
        if current != env[label]["combined"]:
            raise ReproduceRefused(f"{label} mismatch — module changed since the run")


def check_datasets(record: dict) -> None:
    """Frozen-source-data gate: every dataset must still be in cache with an
    identical sha256. Never refreshes from the network."""
    from .cache import load_or_fetch_meta
    from .snapshot import dataset_hash

    mismatches = []
    missing = []
    for name, meta in record["results"]["datasets"].items():
        try:
            candles, _c, fetched_at = load_or_fetch_meta(name, lambda s: (_ for _ in ()).throw(FileNotFoundError()), cache_only=True)
            # fetch_fn unreachable: cache_only path raises before calling it
        except FileNotFoundError:
            missing.append(name)
            continue
        got = dataset_hash(candles)
        if got != meta["sha256"]:
            mismatches.append(f"{name} ({meta['sha256'][:10]}… -> {got[:10]}…)")
        if meta.get("downloaded_at") and fetched_at and float(meta["downloaded_at"]) != float(fetched_at):
            # same bytes but re-downloaded: warn-level, not fatal
            print(f"note: {name} cache was refreshed since the run (bytes verified identical)")
    if missing:
        raise ReproduceRefused("frozen source data not available in cache: " + ", ".join(missing))
    if mismatches:
        raise ReproduceRefused("dataset hashes differ: " + "; ".join(mismatches))


def compare_metrics(stored: dict, fresh: dict, rel: float, abs_: float) -> list[str]:
    import math

    diffs = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if not math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_):
                diffs.append(f"{path}: stored={a} fresh={b}")
        elif a != b:
            diffs.append(f"{path}: stored={a!r} fresh={b!r}")

    walk(stored, fresh, "metrics")
    return diffs


def reproduce_run(run_id: str, runs_dir: str | Path = RUNS_DIR) -> dict:
    """Full reproduction. Returns {'status','diffs',...}; raises
    ReproduceRefused when preconditions fail."""
    record = load_run_record(run_id, runs_dir)
    check_environment(record)
    check_datasets(record)

    params = dict(record["results"]["parameters"])
    params["cache_only"] = True
    params["universe_symbols"] = record["results"]["universe"]

    import argparse

    from .__main__ import compute_compare_results

    args = argparse.Namespace(**params)
    fresh = compute_compare_results(args, log=lambda *a, **k: None, save_run=False)
    rel = record["tolerance"]["rel"]
    abs_ = record["tolerance"]["abs"]
    diffs = compare_metrics(record["results"]["metrics"], fresh["metrics"], rel, abs_)
    return {
        "run_id": run_id,
        "status": "PASS" if not diffs else "FAIL",
        "n_compared_paths": len(record["results"]["metrics"]),
        "diffs": diffs,
        "fresh_exit_code": fresh.get("exit_code"),
    }
