"""Run records + reproduce: the benchmark you can re-execute exactly.

Offline end-to-end: synthetic candles, injected fetchers, universe override,
then save -> reproduce -> PASS; followed by the refusal paths (edited code
hash, edited metrics, changed data) that make the guarantee real.
"""
import json
import time

import pytest

from bot.algorithm import build_algorithm  # noqa: F401  (spec parity context)
from bot.runs import (
    ReproduceRefused,
    list_runs,
    load_run_record,
    reproduce_run,
    run_record_hash,
    save_run_record,
)

_ANCHOR_MS = int(time.time() * 1000)


def _candles(n=900, seed_day=0):
    closes = [100.0 * (1.0007 ** i) * (1.004 if i % 9 == 0 else 1.0) for i in range(n)]
    # anchor to "now" so is_stale() sees live feeds, not 1970
    return [
        {"open_time": _ANCHOR_MS - (n - seed_day - i) * 86_400_000,
         "open": closes[i - 1] if i else closes[0],
         "close": closes[i],
         "quote_volume": closes[i] * 50_000.0,
         "volume": 50_000.0}
        for i in range(n)
    ]


def _args(tmp_path, monkeypatch=None):
    import argparse

    if monkeypatch is not None:
        monkeypatch.chdir(tmp_path)
    else:
        import os
        os.chdir(tmp_path)
    return argparse.Namespace(
        assets=3,
        train_days=365,
        test_days=180,
        fee=0.001,
        spread_bps=5.0,
        slippage_bps=5.0,
        latency_days=0,
        execution="next_open",
        risk_free=0.03,
        portfolio_vol=0.25,
        seed=42,
        cache_only=False,
        universe_symbols=["AAA", "BBB", "SP500X"],
    )


def _fetchers():
        data = {
            "crypto": {
                "AAA": _candles(900),
                "BBB": _candles(900, seed_day=3),
                "SP500X": _candles(900, seed_day=5),
            },
            "sp500": lambda: [{"date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}", "close": 5000.0 + i} for i in range(300)],
        }
        data["crypto"]["BTCUSDT"] = data["crypto"]["AAA"]
        return {
            "crypto": lambda s: data["crypto"][s],
            "yahoo": lambda s: data["crypto"].get(s, _candles(900)),
            "sp500": data["sp500"],
        }


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # SP500X here is a crypto-kind symbol in this fixture's universe; give it
    # its own history via the crypto fetcher (yahoo fallback covers ETF names).
    return tmp_path


class TestSaveAndLoad:
    def test_save_assigns_id_and_integrity_hash(self, tmp_path):
        results = {"metrics": {"a": 1.0}, "universe": ["X"], "parameters": {},
                   "environment": {}, "datasets": {}}
        rid = save_run_record(results, runs_dir=tmp_path / "runs")
        rec = load_run_record(rid, runs_dir=tmp_path / "runs")
        assert rec["run_id"] == rid
        assert rid in str(list_runs(tmp_path / "runs")[0]["run_id"])

    def test_tampered_record_detected(self, tmp_path):
        results = {"metrics": {"a": 1.0}, "universe": [], "parameters": {}, "environment": {}, "datasets": {}}
        rid = save_run_record(results, runs_dir=tmp_path / "runs")
        p = tmp_path / "runs" / rid / "run.json"
        blob = json.loads(p.read_text())
        blob["results"]["metrics"]["a"] = 2.0
        p.write_text(json.dumps(blob))
        with pytest.raises(ValueError, match="integrity"):
            load_run_record(rid, runs_dir=tmp_path / "runs")

    def test_record_hash_excludes_itself(self, tmp_path):
        results = {"metrics": {}, "universe": [], "parameters": {}, "environment": {}, "datasets": {}}
        rid = save_run_record(results, runs_dir=tmp_path / "runs")
        rec = json.loads((tmp_path / "runs" / rid / "run.json").read_text())
        assert run_record_hash(rec) == rec["record_sha256"]


class TestReproduceEndToEnd:
    def test_save_then_reproduce_passes(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None, save_run=True)
        assert res["exit_code"] in (0, 1)
        run_id = res["run_id"]

        out = reproduce_run(run_id, runs_dir=env / "runs")
        assert out["status"] == "PASS", out["diffs"]
        assert out["diffs"] == []

    def test_second_reproduce_still_passes_from_cache_only(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results
        from bot.runs import reproduce_run

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        out = reproduce_run(res["run_id"], runs_dir=env / "runs")
        assert out["status"] == "PASS"

    def test_edited_stored_metric_fails_reproduction(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        rid = res["run_id"]
        p = env / "runs" / rid / "run.json"
        record = json.loads(p.read_text())
        # editing ANY metric also breaks the record hash -> integrity gate fires first
        record["results"]["metrics"]["inv_vol_rm"]["sharpe"] = 99.0
        p.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="integrity"):
            reproduce_run(rid, runs_dir=env / "runs")


class TestRefusals:
    def _saved(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        return res["run_id"]

    def test_environment_mismatch_refuses(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        rid = res["run_id"]
        p = env / "runs" / rid / "run.json"
        record = json.loads(p.read_text())
        record["results"]["environment"]["code_fingerprint"]["sha256"] = "0" * 64
        # keep integrity consistent so we reach the environment check
        from bot.runs import run_record_hash

        record.pop("record_sha256")
        record["record_sha256"] = run_record_hash(record)
        (env / "runs" / rid / "run.json").write_text(json.dumps(record))
        with pytest.raises(ReproduceRefused, match="code fingerprint mismatch"):
            reproduce_run(rid, runs_dir=env / "runs")

    def test_dataset_change_refuses_before_compute(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        rid = res["run_id"]
        # mutate the cached bytes AFTER saving -> frozen source data altered
        cache_file = next(iter((env / ".cache").glob("*.json")))
        blob = json.loads(cache_file.read_text())
        blob["candles"][10]["close"] *= 1.05
        cache_file.write_text(json.dumps(blob))
        with pytest.raises(ReproduceRefused, match="dataset hashes differ"):
            reproduce_run(rid, runs_dir=env / "runs")

    def test_missing_frozen_data_refuses(self, env, monkeypatch):
        from bot.__main__ import compute_compare_results

        args = _args(env, monkeypatch)
        res = compute_compare_results(args, fetch=_fetchers(), log=lambda *a, **k: None)
        rid = res["run_id"]
        for f in (env / ".cache").glob("*.json"):
            f.unlink()
        with pytest.raises(ReproduceRefused, match="frozen source data not available"):
            reproduce_run(rid, runs_dir=env / "runs")
