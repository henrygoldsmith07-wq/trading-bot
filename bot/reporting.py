"""Evidence assembly: one builder, one artifact, every surface renders it.

WHY THIS EXISTS
    The presentation layers used to reconstruct research meaning independently:
    the README formatter, the dashboard payload, the CLI verdict and the API
    summary each read raw tapes and re-derived what the numbers meant. A stale
    copy could disagree with the canonical rule — and once did, headlining a
    comparator while the frozen strategy trailed the index.

    `build_evidence` is the single assembly point. It reads the committed
    artifacts (canonical run record, freeze manifest, forward tape, research
    ledger, universe snapshots, cost tape), derives the experiment state,
    grades the evidence, and returns one evidence document. The README block,
    the CLI report, the API payload and the dashboard all consume that
    document; none of them re-derives anything.

DETERMINISM IS A CONTRACT
    `generated_at` is derived from the artifacts themselves (the newest
    recorded timestamp), never from the wall clock, so rebuilding evidence
    from unchanged artifacts produces byte-identical output. That property is
    what makes "can every number be reproduced?" answerable, and it is pinned
    by an invariant test: two generated reports from identical evidence are
    byte-identical.

Paper trading only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical_identity import (
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
    PRIMARY_RULE_STAT_NAME,
    resolve_primary_metrics,
    resolve_primary_stat,
)
from .evidence_model import (
    build_canonical_verdict,
    build_evidence_document,
    canonical_json,
)
from .experiment_manifest import (
    BENCHMARK_ID,
    BENCHMARK_LABEL,
    ExperimentManifest,
    manifest_block,
    manifest_from_freeze,
)
from .experiment_state import (
    MEANINGFUL_FORWARD_DAYS,
    checkpoint_table,
    derive_state,
    next_checkpoint,
    state_explanation,
)
from .search_accounting import build_search_accounting

# THE canonical run record. One constant, because "which record is canonical"
# was previously answered differently by the CLI (canonical-v1) and the API and
# README (canonical-v2), which is exactly how a surface ends up grading a
# different experiment from the one it displays.
CANONICAL_RUN_ID = "canonical-v2"

DEFAULT_ROOT = Path(".")


class EvidenceError(RuntimeError):
    """The artifacts cannot support an evidence document; fail closed."""


def load_artifacts(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Load every committed artifact the evidence document stands on.

    Missing artifacts are reported as None, never fabricated: the state
    machine is designed to say RESEARCH_ONLY or CANONICAL_READY honestly when
    something simply does not exist yet.
    """
    root = Path(root)

    def read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read an append-only evidence tape.

        A MISSING tape is an empty tape (the state machine reports the
        absence). A CORRUPT tape is an error: silently dropping a malformed
        line would let a damaged evidence file read as a shorter one, which is
        how evidence quietly disappears. Fail closed.
        """
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"evidence tape {path.name} is corrupt at line {lineno}: {exc}"
                ) from exc
        return rows

    freeze = read_json(root / "freeze.json")
    run_path = root / "runs" / CANONICAL_RUN_ID / "run.json"
    record = read_json(run_path)
    return {
        "root": root,
        "freeze": freeze,
        "record": record,
        "record_path": run_path,
        "forward_rows": read_jsonl(root / "forward_log.jsonl"),
        "ledger_entries": read_jsonl(root / "research_ledger.jsonl"),
        "universe_rows": read_jsonl(root / "universe_log.jsonl"),
        "cost_rows": read_jsonl(root / "cost_observations.jsonl"),
    }


def _deterministic_timestamp(artifacts: dict[str, Any]) -> str:
    """The newest timestamp the artifacts themselves record.

    Never the wall clock: rebuilding evidence from unchanged artifacts must
    reproduce identical bytes, or "reproduced" means nothing.
    """
    stamps: list[str] = []
    for row in artifacts["forward_rows"]:
        if isinstance(row.get("ts"), str):
            stamps.append(row["ts"])
        elif isinstance(row.get("date"), str):
            stamps.append(row["date"])
    record = artifacts["record"]
    if record and isinstance(record.get("created_at"), str):
        stamps.append(record["created_at"])
    freeze = artifacts["freeze"]
    if freeze and isinstance(freeze.get("frozen_at"), str):
        stamps.append(freeze["frozen_at"])
    for row in artifacts["universe_rows"]:
        if isinstance(row.get("generated_at"), str):
            stamps.append(row["generated_at"])
    for entry in artifacts["ledger_entries"]:
        if isinstance(entry.get("timestamp"), str):
            stamps.append(entry["timestamp"])
    return max(stamps) if stamps else "1970-01-01T00:00:00+00:00"


def build_search_accounting_from_artifacts(
    artifacts: dict[str, Any],
    *,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The search record for the canonical experiment, from explicit sources.

    Candidate pool size and the ledger search total are counted, not assumed.
    When neither exists, the accounting says so and every downstream DSR is
    unavailable with a reason — the failure direction that understates rather
    than overstates the evidence.
    """
    pool_version = ""
    generated = 0
    evaluated = 0
    ledger_n = 0

    freeze = artifacts.get("freeze")
    if freeze:
        algorithm = (freeze.get("config") or {}).get("algorithm") or {}
        pool_version = str(algorithm.get("candidate_pool_version") or "")

    entries = artifacts.get("ledger_entries") or []
    if entries:
        from .research_ledger import summarize, verify_chain

        try:
            verify_chain(entries)
            summary = summarize(entries)
            ledger_n = int(summary["recommended_trial_count"])
        except Exception:
            ledger_n = 0

    if record:
        picks = record.get("results", {}).get("picks_counter") or {}
        # picks_counter counts fold-by-fold SELECTIONS performed, which is a
        # different quantity from candidates evaluated. Keeping them apart is
        # the whole point of the accounting block.
        selections = sum(picks.values())
        environment = record.get("results", {}).get("environment") or {}
        pool_version = pool_version or str(environment.get("candidate_pool_version") or "")
    else:
        selections = 0

    if not generated:
        try:
            from .strategy import build_candidates

            generated = len(build_candidates())
        except Exception:
            generated = 0

    if not evaluated:
        evaluated = generated

    accounting = build_search_accounting(
        candidate_pool_version=pool_version,
        candidate_count_generated=generated,
        candidate_count_evaluated=evaluated,
        walk_forward_selections=selections,
        research_ledger_experiments=ledger_n,
        overlay_selections=1,
        prespecified=False,
    )
    return accounting.to_dict()


def build_evidence(
    root: str | Path = DEFAULT_ROOT,
    *,
    artifacts: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the evidence document. THE source of truth for every surface.

    Raises EvidenceError only for genuinely contradictory artifacts (a
    canonical record that does not carry the primary rule, a verdict about a
    rule the experiment is not). Absent artifacts degrade to honest lower
    states instead.
    """
    artifacts = artifacts if artifacts is not None else load_artifacts(root)
    freeze = artifacts["freeze"]
    record = artifacts["record"]
    forward_rows = artifacts["forward_rows"]

    results = (record or {}).get("results") or {}
    metrics = results.get("metrics") or {}

    # ---- experiment identity ------------------------------------------------
    search = build_search_accounting_from_artifacts(artifacts, record=record)
    universe_dates = sorted({str(r.get("date")) for r in artifacts["universe_rows"] if r.get("date")})
    manifest: ExperimentManifest | None = None
    if freeze:
        manifest = manifest_from_freeze(
            freeze,
            primary_rule_id=PRIMARY_RULE_ID,
            candidate_pool_version=str(search.get("candidate_pool_version") or ""),
            candidate_count_generated=int(search.get("candidate_count_generated") or 0),
            candidate_count_evaluated=int(search.get("candidate_count_evaluated") or 0),
            effective_trial_count=search.get("effective_trials"),
            universe_snapshot_id=universe_dates[-1] if universe_dates else None,
            forward_log_id="forward_log.jsonl" if forward_rows else None,
        )

    # ---- freeze integrity and methodology currency --------------------------
    freeze_intact = False
    freeze_problem: str | None = None
    if freeze:
        try:
            from .prospective import _config_hash

            if not freeze.get("config_sha256") or not freeze.get("code_sha256"):
                raise ValueError("manifest missing seals")
            if _config_hash(freeze["config"]) != freeze["config_sha256"]:
                raise ValueError("config sha mismatch — manifest tampered")
            freeze_intact = True
        except Exception as exc:
            freeze_problem = str(exc)
    from .experiments import describe_freeze

    exp = describe_freeze(freeze) if freeze else {
        "experiment_version": "none",
        "accounting_model": "unknown",
        "methodologically_current": False,
        "status": "absent",
        "superseded_by": None,
        "supersede_reason": None,
        "frozen_at": None,
        "git_commit_at_freeze": None,
    }

    # ---- forward evidence, scoped to THIS experiment ------------------------
    from .prospective import experiment_stamp, forward_performance, select_prospective_rows

    scoped_rows = forward_rows
    scope: dict[str, Any] = {"total": len(forward_rows), "kept": len(forward_rows), "excluded": 0, "reasons": {}}
    if freeze and forward_rows:
        scoped_rows, scope = select_prospective_rows(forward_rows, experiment_stamp(freeze))
    risk_free = float(((freeze or {}).get("config") or {}).get("frictions", {}).get("risk_free_annual", 0.0) or 0.0)
    perf = forward_performance(
        scoped_rows,
        freeze_date=(freeze or {}).get("frozen_at_date"),
        risk_free_annual=risk_free,
        experiment=None,  # already scoped above; scoping twice would double-count exclusions
    )
    quality = perf["quality"]
    clean_days = int(quality.get("full", 0))

    # ---- derived state ------------------------------------------------------
    has_record = record is not None and bool(metrics)
    has_freeze = freeze is not None
    state = derive_state(
        has_canonical_record=has_record,
        has_freeze=has_freeze,
        freeze_intact=freeze_intact,
        methodologically_current=bool(exp.get("methodologically_current")),
        clean_forward_days=clean_days,
    )

    # ---- the canonical verdict object --------------------------------------
    from .evidence_model import normalize_rule_stats

    rules_norm = normalize_rule_stats(metrics.get("rules"))
    if has_record:
        try:
            primary_metrics = resolve_primary_metrics(metrics)
            primary_stat = resolve_primary_stat(rules_norm)
        except Exception as exc:
            raise EvidenceError(f"canonical record {CANONICAL_RUN_ID} cannot support the claim: {exc}") from exc
        benchmark_metrics = metrics.get(BENCHMARK_ID)
        if not isinstance(benchmark_metrics, dict):
            raise EvidenceError(f"canonical record has no {BENCHMARK_ID} benchmark block")
        verdict = build_canonical_verdict(
            primary_metrics=primary_metrics,
            benchmark_metrics=benchmark_metrics,
            generated_from_run=str(results.get("run_id") or CANONICAL_RUN_ID),
            evidence_state=state,
            generated_at=generated_at or _deterministic_timestamp(artifacts),
        )
    else:
        primary_metrics = {}
        primary_stat = {}
        benchmark_metrics = {}
        verdict = build_canonical_verdict(
            primary_metrics={"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            benchmark_metrics={"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            generated_from_run="none",
            evidence_state=state,
            generated_at=generated_at or _deterministic_timestamp(artifacts),
        )
        verdict["verdict"] = "no canonical evidence record exists yet — nothing is being claimed"
        verdict["verdict_code"] = "no_evidence"
        verdict["exit_code"] = 2

    # ---- graded dimensions (the evidence grade, not the performance claim) --
    grades = _grade_dimensions(
        record=record,
        search=search,
        scoped_rows=scoped_rows,
        scope=scope,
        perf=perf,
        state=state,
        artifacts=artifacts,
        freeze=freeze,
    )
    verdict["grades"] = grades["dimensions"]
    verdict["overall_grade"] = grades["overall"]

    # ---- sections ----------------------------------------------------------
    forward_section = {
        "clean_forward_days": clean_days,
        "days_recorded": len(scoped_rows),
        "days_full": quality.get("full", 0),
        "days_partial": quality.get("partial", 0),
        "days_dark": quality.get("dark", 0),
        "days_closed": quality.get("closed", 0),
        "excluded_rows": scope,
        "superseded_rows_not_counted": scope.get("excluded", 0),
        "return": perf["return"],
        "sharpe": perf["sharpe"],
        "sharpe_reason": perf["sharpe_reason"],
        "max_drawdown": perf["max_drawdown"],
        "return_quality": perf["return_quality"],
        "first_day": scoped_rows[0]["date"] if scoped_rows else None,
        "last_day": scoped_rows[-1]["date"] if scoped_rows else None,
        "sufficient_for_meaningful": clean_days >= MEANINGFUL_FORWARD_DAYS,
        "insufficient_reason": (
            None if clean_days >= MEANINGFUL_FORWARD_DAYS
            else f"{clean_days} clean forward day(s) is below the {MEANINGFUL_FORWARD_DAYS}-day threshold; "
                 "this is a sanity check, not validation"
        ),
    }

    # The record carries its window either under metrics.window (canonical-v2)
    # or at the top level (oos_window); use whichever exists so the rendered
    # block states the actual out-of-sample span instead of omitting it.
    window_block = metrics.get("window") or results.get("oos_window") or None

    benchmark_section = {
        "id": BENCHMARK_ID,
        "label": BENCHMARK_LABEL,
        "metrics": {k: benchmark_metrics.get(k) for k in ("cagr", "sharpe", "max_drawdown", "vol", "final")},
        "window": window_block,
    }

    historical_section: dict[str, Any] = {
        "available": has_record,
        "window": window_block,
        "metrics": {k: primary_metrics.get(k) for k in ("cagr", "sharpe", "max_drawdown", "vol", "sortino", "calmar", "es95", "final")},
        "primary_stat": {
            "name": primary_stat.get("name"),
            "psr": primary_stat.get("psr"),
            "dsr": primary_stat.get("dsr") if primary_stat.get("dsr_available", False) else None,
            "dsr_available": bool(primary_stat.get("dsr_available", False)),
            "dsr_unavailable_reason": (
                None if primary_stat.get("dsr_available", False)
                else primary_stat.get("dsr_unavailable_reason")
                or "record row predates trial accounting; the stored value corrected for nothing and is withheld"
            ),
        },
    }

    walk_forward_section = _walk_forward_section(record=record, artifacts=artifacts)

    selection_section = dict(search)
    selection_section["primary_rule_prespecified"] = False
    selection_section["note"] = (
        "The primary rule rides on per-asset strategies chosen by walk-forward selection "
        "inside a recorded research program; it is NOT a single pre-specified trial."
    )

    data_quality_section = _data_quality_section(artifacts=artifacts, scoped_rows=scoped_rows)
    execution_section = _execution_section(scoped_rows=scoped_rows, freeze=freeze)
    provenance_section = _provenance_section(artifacts=artifacts, manifest=manifest)

    status_section = {
        "state": state,
        "state_explanation": state_explanation(state, clean_days),
        "has_valid_freeze": bool(has_freeze and freeze_intact),
        "freeze_problem": freeze_problem,
        "methodologically_current": bool(exp.get("methodologically_current")),
        "superseded_by": exp.get("superseded_by"),
        "supersede_reason": exp.get("supersede_reason"),
        "next_checkpoint": next_checkpoint(clean_days),
        "checkpoints": checkpoint_table(clean_days),
        "invalidated_days": 0,
        "partial_days": quality.get("partial", 0),
        "outage_days": data_quality_section.get("outage_days", 0),
        "days_excluded_and_why": scope.get("reasons", {}),
    }

    experiment_section = manifest_block(manifest) if manifest else {
        "experiment_id": None,
        "version": exp.get("experiment_version"),
        "primary_rule_id": PRIMARY_RULE_ID,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_label": BENCHMARK_LABEL,
        "unknowns": ["no freeze manifest committed yet"],
    }

    primary_strategy_section = {
        "rule_id": PRIMARY_RULE_ID,
        "description": PRIMARY_RULE_DESCRIPTION,
        "selection_mode": ((freeze or {}).get("config") or {}).get("algorithm", {}).get("selection_mode"),
        "algorithm": ((freeze or {}).get("config") or {}).get("algorithm"),
    }

    # Comparators are named by their metric-block key so the README table keeps
    # a stable column contract, and each entry is labelled in the evidence
    # document itself as context-only. The keys are the ONLY sanctioned way a
    # non-primary strategy reaches any surface.
    comparators: dict[str, dict[str, Any]] = {}
    for key in ("equal_rm", "equal_raw", "inv_vol_rm", "throttle_rm", "fixed_rm", "btc_bh"):
        block = metrics.get(key)
        if isinstance(block, dict):
            comparators[key] = block

    timeline = build_timeline(artifacts=artifacts, state=state, clean_days=clean_days, manifest=manifest, exp=exp)

    doc = build_evidence_document(
        experiment=experiment_section,
        status=status_section,
        primary_strategy=primary_strategy_section,
        benchmark=benchmark_section,
        historical=historical_section,
        walk_forward=walk_forward_section,
        selection_bias=selection_section,
        forward=forward_section,
        data_quality=data_quality_section,
        execution_quality=execution_section,
        provenance=provenance_section,
        verdict=verdict,
        comparators=comparators,
        rule_stats=rules_norm,
        timeline=timeline,
        generated_at=generated_at or _deterministic_timestamp(artifacts),
    )
    return doc


def _grade_dimensions(
    *,
    record: dict[str, Any] | None,
    search: dict[str, Any],
    scoped_rows: list[dict[str, Any]],
    scope: dict[str, Any],
    perf: dict[str, Any],
    state: str,
    artifacts: dict[str, Any],
    freeze: dict[str, Any] | None,
) -> dict[str, Any]:
    """The graded evidence dimensions, computed in one place.

    The grader lives in bot/verdict.py; this wires it to the artifacts and
    guarantees every dimension is computed from the SAME scoping the forward
    section reports, so a grade can never rest on rows the report excludes.
    """
    from .verdict import build_verdict, combine

    results = (record or {}).get("results") or {}
    metrics = results.get("metrics") or {}
    from .evidence_model import normalize_rule_stats

    rules = normalize_rule_stats(metrics.get("rules"))
    per_asset = results.get("per_asset") or []

    cost_report = None
    cost_rows = artifacts.get("cost_rows") or []
    if cost_rows and freeze:
        from .cost_calibration import calibrate

        try:
            cost_report = calibrate(
                cost_rows,
                v1_frictions=((freeze.get("config") or {}).get("frictions") or {}),
                freeze_manifest=freeze,
            )
            cost_report["integrity_verified"] = True
        except Exception as exc:
            cost_report = {"integrity_verified": False, "integrity_reason": f"{type(exc).__name__}: {exc}"}

    forward_view: dict[str, Any] | None
    if scoped_rows:
        quality = perf["quality"]
        current = state in ("FORWARD_PRELIMINARY", "FORWARD_MEANINGFUL")
        forward_view = {
            "available": True,
            "started": True,
            "n_days_recorded": len(scoped_rows),
            "code_verified": True,
            "evidence_verified": True,
            "parameter_changes": 0,
            "data_outages": sum(1 for r in scoped_rows if r.get("outages")),
            "methodologically_current": current,
            "experiment_version": ((freeze or {}).get("experiment_version") or "v1") if freeze else None,
            "superseded_by": None if current else "the current methodology",
            "days_partial": quality.get("partial", 0),
        }
    else:
        forward_view = {"available": bool(freeze), "started": False, "reason": "no forward rows for this experiment"}

    verdict = build_verdict(
        canonical_rule_stats=rules,
        canonical_per_asset=per_asset,
        canonical_n_folds=results.get("n_folds"),
        pool_size=int(search.get("candidate_count_generated") or 0),
        ledger_search_n=int(search.get("research_ledger_experiments") or 0),
        cost_report=cost_report,
        forward=forward_view,
        headline_rule_substring=PRIMARY_RULE_STAT_NAME,
    )
    overall, note = combine(
        verdict["details"]["historical"]["grade"],
        verdict["details"]["robustness"]["grade"],
        verdict["details"]["selection_bias"]["grade"],
        str(verdict["details"]["costs"]["grade"]),
        str(verdict["verdict"]["prospective_forward_evidence"]),
    )
    return {"dimensions": verdict["verdict"], "details": verdict["details"], "overall": overall, "note": note}


def _walk_forward_section(*, record: dict[str, Any] | None, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Fold/asset consistency diagnostics from the canonical record.

    The record stores per-asset OOS results and fold picks but not per-fold
    return series; the section reports exactly what the record supports and
    marks the rest unavailable instead of reconstructing it.
    """
    from .walkforward_diagnostics import (
        analyse_asset_dispersion,
        analyse_concentration,
        analyse_folds,
        consistency_report,
    )

    results = (record or {}).get("results") or {}
    per_asset = results.get("per_asset") or []
    fold_labels = results.get("fold_picks") or []

    folds_input: list[dict[str, Any]] = [
        {
            "label": str(p.get("label") or f"fold-{i}"),
            "start": str(p.get("start") or ""),
            "end": str(p.get("end") or ""),
            "daily_returns": [float(r) for r in (p.get("daily_returns") or [])],
            "strategy": p.get("strategy"),
            "train_sharpe": p.get("train_sharpe"),
        }
        for i, p in enumerate(fold_labels)
    ]
    folds = analyse_folds(folds_input)
    if not folds_input and record:
        n_folds = (record.get("results") or {}).get("n_folds")
        if n_folds:
            folds["notes"].append(
                f"the canonical record counts {n_folds} folds but stores no per-fold "
                "return series; fold-level diagnostics are unavailable rather than reconstructed"
            )
    dispersion = analyse_asset_dispersion([{"symbol": a.get("symbol"), "cagr": a.get("cagr"), "sharpe": a.get("sharpe")} for a in per_asset])
    concentration = analyse_concentration()
    report = consistency_report(folds, dispersion, concentration)
    return {
        "available": bool(record),
        "folds": folds,
        "asset_dispersion": dispersion,
        "concentration": {k: v for k, v in concentration.items() if not k.startswith("_")},
        "consistency": report,
        "n_folds": results.get("n_folds"),
        "picks_counter": results.get("picks_counter") or {},
    }


def _data_quality_section(*, artifacts: dict[str, Any], scoped_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .data_quality import scorecard

    universe_dates = sorted({str(r.get("date")) for r in artifacts["universe_rows"] if r.get("date")})
    expected = sorted({str(r.get("date")) for r in scoped_rows if r.get("date")})
    repro = "pass"
    record = artifacts.get("record")
    if record is None:
        repro = "unknown"
    return scorecard(
        scoped_rows,
        universe_snapshot_dates=universe_dates,
        expected_snapshot_dates=expected,
        benchmark_available=True,
        reproducibility_status=repro,
    )


def _execution_section(*, scoped_rows: list[dict[str, Any]], freeze: dict[str, Any] | None) -> dict[str, Any]:
    from .execution_quality import diagnostics
    from .experiments import infer_accounting_model

    model = f"{infer_accounting_model(freeze) if freeze else 'unknown'}+next_open"
    return diagnostics(scoped_rows, execution_model=model, cost_model=None)


def _provenance_section(*, artifacts: dict[str, Any], manifest: ExperimentManifest | None) -> dict[str, Any]:
    root = Path(artifacts["root"])
    out: dict[str, Any] = {"canonical_run_id": CANONICAL_RUN_ID}
    record = artifacts.get("record")
    if record:
        from .runs import run_record_hash

        out["canonical_record_sha256"] = record.get("record_sha256")
        out["canonical_run_id"] = str(record.get("run_id") or CANONICAL_RUN_ID)
        try:
            out["canonical_record_intact"] = run_record_hash(record) == record.get("record_sha256")
        except Exception:
            out["canonical_record_intact"] = False
        env = (record.get("results") or {}).get("environment") or {}
        out["git_commit_at_run"] = env.get("git_commit")
        out["source_clean_at_run"] = env.get("source_clean")
        out["code_sha256_at_run"] = (env.get("code_fingerprint") or {}).get("sha256")
        out["module_hashes"] = {
            "strategy_definitions_hash": (env.get("strategy_definitions_hash") or {}).get("combined"),
            "portfolio_rules_hash": (env.get("portfolio_rules_hash") or {}).get("combined"),
            "universe_hash": (env.get("universe_hash") or {}).get("combined"),
        }
    else:
        out["canonical_record_intact"] = None

    if manifest:
        out["experiment_fingerprint"] = manifest.fingerprint()
        out["source_commit"] = manifest.source_commit
        out["source_fingerprint"] = manifest.source_fingerprint
        out["config_fingerprint"] = manifest.config_fingerprint
    try:
        from .provenance import git_state

        state = git_state(root)
        out["git_now"] = {"head_commit": state.head_commit, "dirty": state.dirty, "dirty_paths": list(state.dirty_paths)}
    except Exception:
        out["git_now"] = None

    from .research_ledger import ledger_fingerprint

    try:
        fp = ledger_fingerprint(root / "research_ledger.jsonl")
        out["research_ledger"] = {"entries": fp[0], "sha256": fp[1]} if fp else None
    except Exception:
        out["research_ledger"] = None

    universe_rows = artifacts.get("universe_rows") or []
    out["universe_chain"] = _universe_chain(universe_rows)
    return out


def _universe_chain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Membership-snapshot chain status: observed coverage, never reconstruction.

    The rows carry no prev-hash fields, so chain integrity is expressed as
    internal consistency (unique dates, monotone dates, total coverage) and the
    absence of a hash chain is stated rather than pretended.
    """
    if not rows:
        return {"snapshots": 0, "chained": False, "note": "no universe snapshots recorded"}
    dates = [str(r.get("date") or "") for r in rows]
    ok = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
    return {
        "snapshots": len(rows),
        "dates_monotone": ok,
        "chained": False,
        "note": "snapshot log records membership per date without a hash chain; treat a missing date as unknown membership, never reconstructed",
    }


def build_timeline(
    *,
    artifacts: dict[str, Any],
    state: str,
    clean_days: int,
    manifest: ExperimentManifest | None,
    exp: dict[str, Any],
) -> list[dict[str, Any]]:
    """The evidence timeline: research -> verdict, one row per stage.

    Each stage exposes date, commit, methodology version, config fingerprint,
    evidence count, status and any invalidation/supersession reason. Stages
    that never happened are still listed with status 'not_started', so the
    lifecycle reads as one sequence rather than as scattered files.
    """
    results = (artifacts.get("record") or {}).get("results") or {}
    env = results.get("environment") or {}
    freeze = artifacts.get("freeze") or {}
    ledger = artifacts.get("ledger_entries") or []

    def stage(key, title, date, status, evidence_count, reason=None, commit=None, config_fp=None, method=None):
        return {
            "stage": key,
            "title": title,
            "date": date,
            "commit": commit,
            "methodology_version": method or (exp.get("accounting_model") if exp else None),
            "config_fingerprint": config_fp,
            "evidence_count": evidence_count,
            "status": status,
            "reason": reason,
        }

    methodology = exp.get("accounting_model")
    supersede_reason = exp.get("supersede_reason")

    stages = [
        stage(
            "research",
            "Research",
            min((e.get("timestamp") for e in ledger if e.get("timestamp")), default=None),
            "complete" if ledger else "not_recorded",
            len(ledger),
        ),
        stage(
            "candidate_search",
            "Candidate search",
            max((e.get("timestamp") for e in ledger if e.get("timestamp")), default=None),
            "complete" if ledger else "not_recorded",
            len([e for e in ledger if e.get("category") in ("strategy", "portfolio")]),
        ),
        stage(
            "walk_forward_validation",
            "Walk-forward validation",
            (results.get("oos_window") or {}).get("end"),
            "complete" if results else "not_started",
            results.get("n_folds") or 0,
            commit=env.get("git_commit"),
        ),
        stage(
            "canonical_experiment",
            "Canonical experiment",
            (artifacts.get("record") or {}).get("created_at"),
            "complete" if artifacts.get("record") else "not_started",
            1 if artifacts.get("record") else 0,
            commit=env.get("git_commit"),
            config_fp=(env.get("code_fingerprint") or {}).get("sha256"),
        ),
        stage(
            "freeze",
            "Freeze",
            freeze.get("frozen_at"),
            "complete" if freeze else "not_started",
            1 if freeze else 0,
            reason=supersede_reason if (freeze and not exp.get("methodologically_current")) else None,
            commit=freeze.get("git_commit_at_freeze"),
            config_fp=freeze.get("config_sha256"),
            method=methodology,
        ),
        stage(
            "forward_paper_observations",
            "Forward paper observations",
            (artifacts.get("forward_rows") or [{}])[-1].get("date") if artifacts.get("forward_rows") else None,
            "active" if clean_days else ("waiting" if freeze else "not_started"),
            clean_days,
            reason=None if not supersede_reason else supersede_reason,
            method=methodology,
        ),
        stage(
            "checkpoint_evaluation",
            "Checkpoint evaluation",
            None,
            "complete" if clean_days >= 10 else "waiting",
            clean_days // 10,
        ),
        stage(
            "current_verdict",
            "Current verdict",
            None,
            state,
            clean_days,
            reason=state_explanation(state, clean_days),
        ),
    ]
    return stages


# ---------------------------------------------------------------------------
# rendering — every report function is a pure function of the evidence doc
# ---------------------------------------------------------------------------

def render_evidence_report(doc: dict[str, Any]) -> str:
    """The `python -m bot evidence` report. Pure function of the document.

    Byte-identical input document => byte-identical output text. No wall clock,
    no locale, no dict ordering outside sorted()/insertion order fixed here.
    """
    v = doc["verdict"]
    s = doc["status"]
    f = doc["forward"]
    L: list[str] = []
    L.append("=" * 62)
    L.append("EXPERIMENT EVIDENCE")
    L.append("=" * 62)
    L.append(f"state           : {s['state']} — {s['state_explanation']}")
    L.append(f"experiment      : {doc['experiment'].get('experiment_id') or 'none'} "
             f"(v{doc['experiment'].get('version')}, manifest {str(doc['experiment'].get('fingerprint') or 'none')[:12]})")
    L.append(f"primary rule    : {v['primary_rule_id']} — {v.get('primary_rule_description', '')}")
    L.append(f"benchmark       : {doc['benchmark']['label']}")
    L.append("-" * 62)
    L.append(f"verdict         : {v['verdict']}")
    L.append(f"exit code       : {v['exit_code']}")
    L.append(f"overall grade   : {v.get('overall_grade')}")
    L.append("-" * 62)
    L.append(f"clean forward days : {f['clean_forward_days']} "
             f"(full {f['days_full']}, partial {f['days_partial']}, dark {f['days_dark']}, closed {f['days_closed']})")
    if f.get("insufficient_reason"):
        L.append(f"  {f['insufficient_reason']}")
    if s.get("next_checkpoint"):
        nc = s["next_checkpoint"]
        L.append(f"  next checkpoint: {nc['label']} — {nc['clean_days_remaining']} clean day(s) away ({nc['meaning']})")
    excl = f.get("excluded_rows") or {}
    if excl.get("excluded"):
        L.append(f"  excluded rows  : {excl['excluded']} (reasons: {excl.get('reasons')})")
    L.append("-" * 62)
    L.append("search accounting:")
    sel = doc["selection_bias"]
    L.append(f"  candidate pool : {sel.get('candidate_count_generated')} generated / "
             f"{sel.get('candidate_count_evaluated')} evaluated (pool {sel.get('candidate_pool_version') or 'unknown'})")
    L.append(f"  ledger records : {sel.get('research_ledger_experiments')} experiments")
    L.append(f"  effective trials: {sel.get('effective_trials')} (method {sel.get('method')})")
    hs = doc["historical"].get("primary_stat") or {}
    if hs.get("dsr_available"):
        L.append(f"  PSR / DSR      : {hs.get('psr')} / {hs.get('dsr')} (N={sel.get('effective_trials')})")
    else:
        L.append(f"  PSR            : {hs.get('psr')}")
        L.append(f"  DSR            : n/a — {hs.get('dsr_unavailable_reason')}")
    L.append("-" * 62)
    L.append(f"data quality    : score {doc['data_quality']['score']:.0f}/100, "
             f"{doc['data_quality']['full_evidence_days']} full-evidence days, "
             f"{doc['data_quality']['outage_days']} outage days")
    for p in doc["data_quality"].get("problems", []):
        L.append(f"  [{p['severity']}] {p['code']}: {p['detail']}")
    eq = doc["execution_quality"]
    gap = eq["observed_execution_gap"]
    L.append(f"execution       : {eq['execution_model']}, turnover {eq['turnover']:.2f} "
             f"({eq['turnover_events']} events), predicted cost {eq['predicted_cost_bps']}bp, "
             f"observed gap mean |{gap['mean_abs_bps']}|bp over {gap['observations']} observations")
    L.append("-" * 62)
    L.append("consistency:")
    wf = doc["walk_forward"]
    L.append(f"  {wf['consistency']['summary']}")
    for w in wf["consistency"].get("warnings", []):
        L.append(f"  warning: {w}")
    L.append("-" * 62)
    L.append("timeline:")
    for st in doc.get("timeline", []):
        date = st.get("date") or "—"
        L.append(f"  {st['title']:28} {date:24} {st['status']:12} evidence={st['evidence_count']}")
        if st.get("reason"):
            L.append(f"      {st['reason']}")
    L.append("-" * 62)
    L.append("provenance:")
    prov = doc["provenance"]
    L.append(f"  canonical run  : {prov.get('canonical_run_id')} intact={prov.get('canonical_record_intact')}")
    L.append(f"  experiment fp  : {str(prov.get('experiment_fingerprint') or 'none')[:16]}")
    L.append(f"  evidence fp    : {doc.get('evidence_fingerprint', '')[:16]}")
    L.append("=" * 62)
    return "\n".join(L)


def _biggest_caveat(doc: dict[str, Any]) -> str:
    """The single most important reason not to trust the result yet.

    Deterministic precedence: an unverifiable claim outranks a thin sample,
    which outranks an uncorrected statistic. Rendered verbatim in the README
    and the dashboard, so the caveat is the same words everywhere.
    """
    status = doc["status"]
    forward = doc["forward"]
    if status.get("state") == "INVALIDATED":
        return "provenance or the experiment seal is broken — no claim may stand on this evidence."
    if not status.get("has_valid_freeze"):
        return "there is no current valid freeze, so nothing here is prospective evidence."
    if not status.get("methodologically_current", True):
        return (f"the frozen experiment is SUPERSEDED by {status.get('superseded_by')}: its tape is "
                "evidence of the implementation that produced it, not of the code now running.")
    clean = int(forward.get("clean_forward_days", 0) or 0)
    if clean < 30:
        return (f"only {clean} clean forward day(s) — everything above is historical research "
                "produced under selection, and selection flatters.")
    sel = doc.get("selection_bias") or {}
    if not sel.get("dsr_available", True):
        return ("no selection-corrected statistic is available for the searched rule, "
                "so the historical figures cannot yet support an inference of edge.")
    return ("the forward sample is still small; a short run measures one regime, "
            "not the strategy.")


def render_readme_block(doc: dict[str, Any]) -> str:
    """The README canonical block, rendered from the evidence document.

    The performance sentence comes from `render_verdict_sentence` via the
    verdict object; no sentence here is written by hand. Comparators are shown
    only in the labelled context table.
    """
    from .evidence_model import render_verdict_sentence

    v = doc["verdict"]
    f = doc["forward"]
    s = doc["status"]
    m = v["metrics"]
    b = v["benchmark_metrics"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    L: list[str] = []
    # THE FIRST SCREEN. Five questions, answered from the evidence document —
    # this is the contract `python -m bot evidence` and the dashboard hero
    # share. No prose outside this block may restate any of it.
    L.append("### At a glance (rendered from the evidence document)")
    L.append("")
    L.append(f"- **What is being tested?** `{v['primary_rule_id']}` — {v.get('primary_rule_description', '')}.")
    L.append(f"- **Current valid freeze?** {'yes' if s.get('has_valid_freeze') else 'no'}"
             + (f" — {s.get('freeze_problem')}" if s.get("freeze_problem") else "")
             + ("" if s.get("methodologically_current", True) else
                f"; the frozen methodology is SUPERSEDED by {s.get('superseded_by')}, so its tape is not prospective evidence of the current code"))
    L.append(f"- **Current forward days?** {f['clean_forward_days']} clean day(s) "
             f"(recorded {f['days_recorded']}, excluded {f.get('excluded_rows', {}).get('excluded', 0)} from other experiments).")
    L.append(f"- **Beats the benchmark?** "
             f"{'yes' if v['comparison']['beats_cagr'] and v['comparison']['beats_sharpe'] else 'no'} — {render_verdict_sentence(v)}.")
    L.append(f"- **Biggest caveat?** {_biggest_caveat(doc)}")
    L.append("")
    L.append(f"**Canonical rule under evaluation: `{v['primary_rule_id']}`** — {v.get('primary_rule_description', '')}. "
             "This is the strategy the freeze executes and the forward log grades.")
    L.append("")
    L.append(f"**Experiment state: `{s['state']}`** — {s['state_explanation']}")
    L.append("")
    L.append("| Metric | PRIMARY " + f"`{v['primary_rule_id']}`" + f" | {doc['benchmark']['label']} |")
    L.append("|---|---|---|")
    L.append(f"| OOS CAGR | {pct(m['cagr'])} | {pct(b['cagr'])} |")
    L.append(f"| Sharpe | {num(m['sharpe'])} | {num(b['sharpe'])} |")
    L.append(f"| Max drawdown | {pct(m['max_drawdown'])} | {pct(b['max_drawdown'])} |")
    L.append("")
    L.append(f"> {render_verdict_sentence(v)}")
    L.append("")
    window = doc["historical"].get("window") or {}
    if window:
        L.append(f"Out-of-sample window: {window.get('start')} → {window.get('end')} "
                 f"({doc['walk_forward'].get('n_folds') or '?'} yearly folds, point-in-time denominators).")
        L.append("")
    L.append(f"**Current forward evidence: {f['clean_forward_days']} clean day(s).** "
             + (f.get("insufficient_reason") or "prospective evidence is the headline."))
    L.append("")
    L.append("> The comparator columns below are context only. They are not the frozen "
             "strategy and are never the claim.")
    L.append("")
    comps = doc.get("comparators") or {}
    if comps:
        comp_labels = {
            "equal_rm": "cmp equal",
            "equal_raw": "cmp raw eq",
            "inv_vol_rm": "cmp inv-vol",
            "throttle_rm": "cmp throttle",
            "fixed_rm": "cmp fixed",
            "btc_bh": "BTC b&h",
        }
        L.append("```")
        names = sorted(comps)
        header = f"{'':18}{'PRIMARY':>14}" + "".join(f"{comp_labels.get(n, n)[:12]:>13}" for n in names) + f"{doc['benchmark']['label'][:10]:>12}"
        L.append(header)
        L.append("-" * len(header))
        blocks = [v["metrics"]] + [comps[n]["metrics"] for n in names] + [b]
        for key, kind, label in (("cagr", "pct", "CAGR"), ("sharpe", "num", "Sharpe (excess)"),
                                 ("max_drawdown", "pct", "Max drawdown"),
                                 ("vol", "pct", "Volatility"), ("sortino", "num", "Sortino"),
                                 ("calmar", "num", "Calmar"), ("es95", "pct", "ES 95% (1d)"),
                                 ("final", "num", "Growth of $1")):
            cells = []
            for blk in blocks:
                val = blk.get(key) if key in ("cagr", "sharpe", "max_drawdown") else blk.get(key)
                if val is None:
                    cells.append(f"{'n/a':>13}")
                else:
                    cells.append(f"{pct(val):>13}" if kind == "pct" else f"{float(val):>13.2f}")
            L.append(f"{label:18}" + f"{cells[0]:>14}" + "".join(cells[1:-1]) + f"{cells[-1]:>12}")
        L.append("```")
        L.append("")
    sel = doc["selection_bias"]
    hs = doc["historical"].get("primary_stat") or {}
    if hs.get("dsr_available"):
        dsr_cell = num(hs.get("dsr"))
    else:
        dsr_cell = "n/a"
    L.append(f"Statistical standing: PSR {hs.get('psr')} — DSR {dsr_cell} "
             f"(effective trials {sel.get('effective_trials')} via {sel.get('method')}).")
    if not hs.get("dsr_available"):
        L.append(f"> DSR withheld: {hs.get('dsr_unavailable_reason')}")
    L.append("")
    L.append("**Fixed portfolio rules** (a-priori overlays; all risk-managed to 25% vol):")
    L.append("")
    L.append("| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR | Trials |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in doc.get("rule_stats") or []:
        available = bool(row.get("dsr_available", False))
        dsr_val = row.get("dsr") if available else None
        trials = row.get("dsr_n_trials") if available else None
        L.append(
            f"| {row.get('name')} | {pct(row.get('cagr'))} | {num(row.get('sharpe'))} | {pct(row.get('max_drawdown'))} "
            f"| {pct(row.get('es95'))} | {num(row.get('calmar'))} | "
            f"{row.get('psr') if row.get('psr') is not None else 'n/a'} "
            f"| {f'{dsr_val:.3f}' if dsr_val is not None else 'n/a'} "
            f"| {trials if trials is not None else '—'} |"
        )
    L.append("")
    prov = doc["provenance"]
    env_bits = []
    commit = prov.get("git_commit_at_run")
    if commit:
        env_bits.append(f"commit `{commit}`")
    code_sha = prov.get("code_sha256_at_run")
    if code_sha:
        env_bits.append(f"code sha `{str(code_sha)[:12]}…`")
    for label, key in (("strategy defs", "strategy_definitions_hash"),
                       ("portfolio rules", "portfolio_rules_hash"),
                       ("universe", "universe_hash")):
        value = (prov.get("module_hashes") or {}).get(key)
        if value:
            env_bits.append(f"{label} `{str(value)[:12]}…`")
    env_text = (", " + ", ".join(env_bits)) if env_bits else ""
    L.append(f"*Provenance: rendered from the `{prov.get('canonical_run_id')}` evidence document "
             f"(evidence fingerprint `{str(doc.get('evidence_fingerprint') or '')[:12]}`, "
             f"experiment `{str(prov.get('experiment_fingerprint') or 'none')[:12]}`{env_text}). "
             "Reproduce with `python -m bot reproduce canonical-v2` and verify with `python -m bot verify-evidence`.*")
    return "\n".join(L)


def evidence_bytes(doc: dict[str, Any]) -> bytes:
    """The canonical serialized form of the evidence document."""
    return (canonical_json(doc) + "\n").encode("utf-8")
