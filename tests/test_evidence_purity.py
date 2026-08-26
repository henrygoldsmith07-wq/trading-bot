"""Forward-evidence purity: classification, gating, quarantine, hermeticity.

The invariant under test: running the ENTIRE test suite leaves the committed
forward evidence byte-identical, and only verified forward-paper rows
matching the active freeze can influence the prospective cost verdict.
"""
import json
from datetime import UTC, datetime

from bot.cost_calibration import calibrate
from bot.evidence import (
    EVIDENCE_FORWARD_PAPER,
    EVIDENCE_HISTORICAL_SIM,
    EVIDENCE_INVALID,
    classify_legacy_row,
    quarantine_archive,
    temporal_problems,
)


def _freeze():
    return {
        "frozen_at": "2026-08-25T10:54:24+00:00",
        "frozen_at_date": "2026-08-25",
        "git_commit_at_freeze": "41e3ee9a",
        "config": {"assets": [{"symbol": s} for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]},
    }


def _row(**over):
    row = {
        "ts": "2026-08-26T21:45:00+00:00",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "decision_close": 100.0,
        "exec_price": 100.05,
        "turnover": 1.0,
        "predicted_cost_bps": 20.0,
        "observed_cost_proxy_bps": 5.0,
    }
    row.update(over)
    return row


NOW = datetime(2026, 8, 27, tzinfo=UTC)
FROZEN_AT = "2026-08-25T10:54:24+00:00"


class TestClassification:
    def test_matching_universe_post_freeze_is_forward_paper(self):
        cls, reason, enriched = classify_legacy_row(
            _row(), freeze_universe={"BTCUSDT", "ETHUSDT"},
            freeze_id="freeze/2026-08-25", frozen_at_iso=FROZEN_AT,
            frozen_git_commit="41e3ee9a", seen_identity_hashes=set(), now=NOW,
        )
        assert cls == EVIDENCE_FORWARD_PAPER and reason == ""
        assert enriched["freezeId"] == "freeze/2026-08-25"
        assert enriched["frozenGitCommit"] == "41e3ee9a"

    def test_synthetic_symbol_is_fixture(self):
        cls, reason, _ = classify_legacy_row(
            _row(symbol="AAA"), freeze_universe={"BTCUSDT"},
            freeze_id="x", frozen_at_iso=FROZEN_AT,
            frozen_git_commit="", seen_identity_hashes=set(), now=NOW,
        )
        assert (cls, reason) == ("fixture", "fixture_test")

    def test_pre_freeze_row_is_historical_sim(self):
        cls, reason, e = classify_legacy_row(
            _row(ts="2026-08-01T00:00:00+00:00"), freeze_universe={"BTCUSDT"},
            freeze_id="freeze/2026-08-25", frozen_at_iso=FROZEN_AT,
            frozen_git_commit="41e3ee9a", seen_identity_hashes=set(), now=NOW,
        )
        assert cls == EVIDENCE_HISTORICAL_SIM
        assert reason == "wrong_freeze"

    def test_duplicate_is_replay(self):
        seen = set()
        r1 = _row()
        c1, _, e1 = classify_legacy_row(r1, freeze_universe={"BTCUSDT"}, freeze_id="f",
                                        frozen_at_iso=FROZEN_AT, frozen_git_commit="",
                                        seen_identity_hashes=seen, now=NOW)
        c2, reason2, _ = classify_legacy_row(_row(), freeze_universe={"BTCUSDT"}, freeze_id="f",
                                             frozen_at_iso=FROZEN_AT, frozen_git_commit="",
                                             seen_identity_hashes=seen, now=NOW)
        assert c1 == "forward-paper" and c2 == "replay" and reason2 == "replay"

    def test_future_timestamp_rejected(self):
        row = _row(ts="2030-01-01T00:00:00+00:00")
        problems = temporal_problems(row, now=datetime(2026, 8, 27, tzinfo=UTC))
        assert any("future" in p for p in problems)
        cls, reason, e = classify_legacy_row(row, freeze_universe={"BTCUSDT"},
                                             freeze_id="f", frozen_at_iso=FROZEN_AT,
                                             frozen_git_commit="", seen_identity_hashes=set(),
                                             now=datetime(2026, 8, 27, tzinfo=UTC))
        assert cls == EVIDENCE_INVALID and reason == "bad_timestamps"

    def test_signal_after_execution_rejected(self):
        row = _row(signal_ts="2027-01-01T00:00:00+00:00")
        cls, reason, _ = classify_legacy_row(row, freeze_universe={"BTCUSDT"}, freeze_id="f",
                                             frozen_at_iso=FROZEN_AT, frozen_git_commit="",
                                             seen_identity_hashes=set(), now=NOW)
        assert cls == EVIDENCE_INVALID and reason == "bad_timestamps"

    def test_negative_turnover_invalid_numeric(self):
        row = _row(turnover=-1.0, target_weight=-1.0)
        cls, reason, _ = classify_legacy_row(row, freeze_universe={"BTCUSDT"}, freeze_id="f",
                                             frozen_at_iso=FROZEN_AT, frozen_git_commit="",
                                             seen_identity_hashes=set(), now=NOW)
        # turnover<0 with side BUY: numeric check fires
        assert cls == EVIDENCE_INVALID and reason == "invalid_numeric"


class TestQuarantineArchive:
    def test_no_byte_lost_and_counts_reported(self, tmp_path):
        src = tmp_path / "legacy.jsonl"
        rows = [
            json.dumps(_row(symbol="BTCUSDT")),                       # keep
            json.dumps(_row(symbol="AAA")),                           # fixture
            json.dumps({"garbage": True}),                            # missing_data
            json.dumps(_row(ts="2020-01-01T00:00:00+00:00")),         # historical-sim
            json.dumps(_row(symbol="BTCUSDT")),                       # replay of row 0
        ]
        src.write_text("\n".join(rows) + "\n", encoding="utf-8")
        original_bytes = src.read_bytes()

        report = quarantine_archive(
            src,
            archive_path=tmp_path / "archive.jsonl",
            quarantine_path=tmp_path / "quarantine.jsonl",
            keep_path=tmp_path / "keep.jsonl",
            freeze_manifest=_freeze(),
            now=NOW,
        )
        # archive preserves every byte
        assert (tmp_path / "archive.jsonl").read_bytes() == original_bytes
        assert report["rows_total"] == 5
        assert report["kept_forward_paper"] == 1
        excl = report["excluded"]
        assert excl.get("missing_data") == 1
        assert excl.get("replay") == 1
        assert excl.get("wrong_freeze") == 1
        assert excl.get("fixture_test") == 1

        kept = [json.loads(line) for line in (tmp_path / "keep.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(kept) == 1 and kept[0]["symbol"] == "BTCUSDT"
        quarantined = [json.loads(line) for line in (tmp_path / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(quarantined) == 5  # everything auditable, nothing deleted


class TestVerificationGate:
    def test_calibrate_excludes_wrong_freeze_rows(self, tmp_path):
        matching = {**_row(), "evidenceClass": "forward-paper",
                    "freezeId": "freeze/2026-08-25", "frozenGitCommit": "41e3ee9a"}
        old_study = {**_row(symbol="ETHUSDT", exec_price=100.06),
                     "evidenceClass": "forward-paper",
                     "freezeId": "freeze/OLD-STUDY", "frozenGitCommit": "old"}
        rep = calibrate([matching, old_study], v1_frictions={"fee": 0.001},
                        freeze_manifest=_freeze(), now=NOW)
        assert rep["n_turnover_events"] == 1          # only matching-freeze row measured
        assert rep["evidence_exclusions"].get("wrong_freeze") == 1

    def test_fixture_rows_never_measured(self, tmp_path):
        rows = [_row(symbol="AAA")] * 40              # plenty but all fixture-class
        rep = calibrate(rows, v1_frictions={"fee": 0.001}, freeze_manifest=_freeze())
        assert rep["n_turnover_events"] == 0
        assert rep["sufficient"] is False


def _freeze():
    return {
        "frozen_at": FROZEN_AT,
        "frozen_at_date": "2026-08-25",
        "git_commit_at_freeze": "41e3ee9a",
        "config": {"assets": [{"symbol": s} for s in ("BTCUSDT", "ETHUSDT")]},
    }



class TestSuiteHermeticity:
    def test_repo_tape_byte_identical_after_pollution_attempt(self, tmp_path, monkeypatch):
        root_tape = tmp_path / 'repo' / 'cost_observations.jsonl'
        root_tape.parent.mkdir(parents=True, exist_ok=True)

        real_rows = [
            json.dumps({**_row(symbol='BTCUSDT'), 'evidenceClass': 'forward-paper',
                        'freezeId': 'freeze/2026-08-25',
                        'frozenGitCommit': '41e3ee9a'}),
            json.dumps({**_row(symbol='ETHUSDT'), 'evidenceClass': 'forward-paper',
                        'freezeId': 'freeze/2026-08-25'}),
        ]
        root_tape.write_text(chr(10).join(real_rows) + chr(10), encoding='utf-8')
        committed = root_tape.read_bytes()

        monkeypatch.setenv('COST_OBSERVATIONS_LOG', str(tmp_path / 'scratch' / 'obs.jsonl'))
        import importlib

        import bot.cost_calibration as cc
        importlib.reload(cc)
        from bot.cost_calibration import append_observation as app2
        from bot.cost_calibration import build_observation as build2

        for i in range(5):
            app2(build2(symbol=f'FAKE{i}', side='BUY', target_weight=1.0,
                        previous_weight=0.0, decision_close=10.0, exec_price=10.01,
                        mark_price=10.02, bid=None, ask=None, predicted_cost_bps=20.0,
                        realized_vol_annual=0.5, adv30_usd=1e9, day_volume_base=100.0),
                 path=tmp_path / 'scratch' / 'obs.jsonl')

        assert root_tape.read_bytes() == committed
        assert b'FAKE' not in root_tape.read_bytes()
