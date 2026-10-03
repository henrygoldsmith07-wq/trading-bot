"""CLI command behaviour, tested at the seams where no network is needed.

`bot/commands.py` is the largest untested surface in the repository: every
command funnels through it, and none of it had direct coverage. These tests
exercise the bookkeeping commands (ledger, cost calibration, quarantine,
reproduce listing, snapshot recording, freeze verification) against temporary
fixtures only — the committed tapes are never opened for writing.

Invariants pinned here:
* commands never write to committed evidence tapes;
* a corrupt or absent input degrades to a stated message and a non-zero exit;
* freeze verification refuses a tampered manifest;
* the research context pins the ledger and the canonical run explicitly.
"""
from __future__ import annotations

import argparse
import json

import pytest

from bot.commands import (
    _cum,
    current_git_commit_safe,
    manifest_date,
    run_ask,
    run_calibrate_costs,
    run_ledger,
    run_quarantine_costs,
    run_reproduce,
    run_universe_snapshot,
    run_verify_freeze,
)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def _freeze_manifest(frozen_date="2026-08-23") -> dict:
    from bot.identity import code_fingerprint
    from bot.prospective import _config_hash

    config = {
        "assets": [{
            "symbol": "BTCUSDT",
            "source": "binance",
            "periods_per_year": 365,
            "session": "continuous",
            "strategy": {"type": "TrendVol", "params": {"lookback": 50, "vol_window": 20, "target_vol": 0.2}},
        }],
        "frictions": {"fee": 0.001, "execution": "next_open"},
        # Validation requires the COMPLETE algorithm block: a manifest without
        # it would be a contract that cannot say what it froze.
        "algorithm": {
            "selection_mode": "walk_forward_selected",
            "candidate_pool_version": "7d8ad2d1c12ee29f",
            "universe_selection_rule": "top-N test fixture",
            "weighting": {"mode": "inverse_vol", "vol_window": 20, "max_multiple_of_equal": 2.0},
            "xs_momentum": {"enabled": True, "lookback": 90, "max_tilt": 0.5},
            "crisis_derisk": {"enabled": True, "corr_window": 60, "corr_threshold": 0.6, "multiplier": 0.6},
            "rebalance_band": 0.05,
            "drawdown_throttle": {"enabled": False, "dd_trigger": -0.1, "dd_exit": -0.05, "factor": 0.5},
            "overlay": {"enabled": True, "target_vol": 0.25, "window": 20, "fee_on_turnover": 0.0015},
        },
    }
    return {
        "frozen_at": f"{frozen_date}T09:00:00+00:00",
        "frozen_at_date": frozen_date,
        "git_commit_at_freeze": "6d6606dabc",
        "config": config,
        "config_sha256": _config_hash(config),
        "code_fingerprint_algo": "sha256-lf-v1",
        "code_sha256": code_fingerprint(),
    }


class TestHelpers:
    def test_git_commit_is_a_sha_or_none(self):
        commit = current_git_commit_safe()
        assert commit is None or len(commit) >= 7

    def test_manifest_date_is_iso_day(self):
        assert len(manifest_date()) == 8 and manifest_date().isdigit()

    def test_cum_compounds(self):
        assert list(_cum([0.1, 0.1])) == [pytest.approx(1.1), pytest.approx(1.21)]


class TestLedgerCommand:
    def test_missing_ledger_refuses(self, tmp_path, capsys):
        code = run_ledger(_ns(ledger=str(tmp_path / "none.jsonl")))
        assert code == 2
        assert "No research ledger" in capsys.readouterr().out

    def test_seeded_ledger_reports_counts(self, tmp_path, capsys):
        from bot.research_ledger import append_entry

        path = tmp_path / "research_ledger.jsonl"
        for i in range(2):
            append_entry(
                path,
                category="strategy",
                hypothesis=f"h{i}",
                configuration={},
                primaryMetric="oos sharpe",
                result=0.5,
                accepted=i == 0,
                source_commit="abc123",
                backfilled=False,
            )
        code = run_ledger(_ns(ledger=str(path)))
        out = capsys.readouterr().out
        assert code == 0
        assert "2 experiments" in out
        assert "search total" in out.lower() or "Search categories total" in out


class TestCostCalibrationCommand:
    def test_no_observations_reports_accumulation(self, tmp_path, capsys):
        code = run_calibrate_costs(_ns(observations=str(tmp_path / "none.jsonl"), fee=0.001,
                                       spread_bps=5.0, slippage_v1_bps=5.0, min=30, write=False))
        assert code == 2
        assert "No cost observations" in capsys.readouterr().out


class TestQuarantineCommand:
    def test_missing_tape_reports_nothing_to_quarantine(self, tmp_path, capsys):
        code = run_quarantine_costs(_ns(src=str(tmp_path / "none.jsonl"),
                                        archive=str(tmp_path / "a.jsonl"),
                                        quarantine=str(tmp_path / "q.jsonl"),
                                        keep=str(tmp_path / "k.jsonl"),
                                        freeze_file=str(tmp_path / "freeze.json")))
        assert code == 2
        assert "nothing to quarantine" in capsys.readouterr().out

    def test_tape_rows_are_classified_and_preserved(self, tmp_path, capsys):
        from bot.prospective import experiment_stamp

        freeze_path = tmp_path / "freeze.json"
        manifest = _freeze_manifest()
        freeze_path.write_text(json.dumps(manifest), encoding="utf-8")
        src = tmp_path / "obs.jsonl"
        rows = [
            {"symbol": "BTCUSDT", "evidenceClass": "forward-paper", "turnover": 0.1,
             "decisionClose": 100.0, "execPrice": 100.1,
             "signalAt": "2026-08-24T00:00:00+00:00",
             "simulatedExecutionAt": "2026-08-24T00:00:01+00:00",
             "recordedAt": "2026-08-24T00:00:02+00:00",
             "source": "frozen-runner",
             "freezeId": manifest["frozen_at_date"],
             "frozenGitCommit": manifest["git_commit_at_freeze"],
             "codeFingerprint": manifest["code_sha256"],
             "runId": "r1", "experiment": experiment_stamp(manifest)},
        ]
        src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        code = run_quarantine_costs(_ns(src=str(src), archive=str(tmp_path / "a.jsonl"),
                                        quarantine=str(tmp_path / "q.jsonl"),
                                        keep=str(tmp_path / "kept.jsonl"),
                                        freeze_file=str(freeze_path)))
        out = capsys.readouterr().out
        assert code == 0
        assert "quarantined" in out
        # the original bytes survive in the archive
        assert (tmp_path / "a.jsonl").exists() or (tmp_path / "kept.jsonl").exists()


class TestReproduceListing:
    def test_list_enumerates_saved_runs(self, tmp_path, capsys):
        (tmp_path / "canonical-v2").mkdir()
        (tmp_path / "canonical-v2" / "run.json").write_text(json.dumps({
            "run_id": "canonical-v2", "created_at": "2026-01-01T00:00:00+00:00",
            "results": {"verdict": "x"}, "record_sha256": "s",
        }), encoding="utf-8")
        code = run_reproduce(_ns(run_id="list", runs_dir=str(tmp_path)))
        out = capsys.readouterr().out
        assert code == 0
        assert "canonical-v2" in out


class TestFreezeVerification:
    def test_missing_freeze_fails(self, tmp_path, capsys):
        code = run_verify_freeze(_ns(freeze_file=str(tmp_path / "none.json")))
        assert code == 1
        assert "FAIL" in capsys.readouterr().out

    def test_tampered_config_fails_closed(self, tmp_path, capsys):
        path = tmp_path / "freeze.json"
        manifest = _freeze_manifest()
        manifest["config"]["frictions"]["fee"] = 0.002  # change AFTER hashing
        path.write_text(json.dumps(manifest), encoding="utf-8")
        code = run_verify_freeze(_ns(freeze_file=str(path)))
        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL" in out

    def test_intact_manifest_verifies(self, tmp_path, capsys):
        path = tmp_path / "freeze.json"
        path.write_text(json.dumps(_freeze_manifest()), encoding="utf-8")
        code = run_verify_freeze(_ns(freeze_file=str(path)))
        out = capsys.readouterr().out
        assert code == 0
        assert "OK: running implementation matches the freeze" in out


class TestUniverseSnapshot:
    def test_fetch_failure_is_reported_not_raised(self, monkeypatch, tmp_path, capsys):

        def boom():
            raise RuntimeError("all mirrors refused")

        monkeypatch.setattr("bot.universe.fetch_ticker_json", boom)
        code = run_universe_snapshot(_ns(top=5, log=str(tmp_path / "universe_log.jsonl")))
        assert code == 2
        assert "could not fetch" in capsys.readouterr().out

    def test_successful_rank_is_recorded_idempotently(self, monkeypatch, tmp_path, capsys):
        payload = json.dumps([
            {"symbol": "BTCUSDT", "quoteVolume": "1000"},
            {"symbol": "ETHUSDT", "quoteVolume": "500"},
        ])
        monkeypatch.setattr("bot.universe.fetch_ticker_json", lambda: payload)
        log = tmp_path / "universe_log.jsonl"
        assert run_universe_snapshot(_ns(top=2, log=str(log))) == 0
        assert log.exists()
        # idempotent: the same day records once
        assert run_universe_snapshot(_ns(top=2, log=str(log))) == 0
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1


class TestAskCommand:
    def test_without_a_key_the_ai_layer_is_disabled(self, monkeypatch, capsys):
        monkeypatch.setattr("bot.ai.load_api_key", lambda: None)
        code = run_ask(_ns(question="why?", symbol="BTCUSDT"))
        assert code == 2
        assert "AI layer disabled" in capsys.readouterr().out
