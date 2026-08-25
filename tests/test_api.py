import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _load_api():
    spec = importlib.util.spec_from_file_location("api_summary", Path(__file__).resolve().parents[1] / "api" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api():
    return _load_api()


def _fake_candles(n=1600, start=100.0):
    out = []
    px = start
    for i in range(n):
        px *= 1.003 if i > 300 else 1.0
        out.append({"open_time": 1_500_000_000_000 + i * 86_400_000, "close": round(px, 8)})
    return out


def test_build_summary_shape(monkeypatch, api, tmp_path):
    # isolate from the committed canonical record so legacy shape holds
    monkeypatch.setattr(api, "CANONICAL_RUN", str(tmp_path / "missing" / "run.json"))
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    s = api.build_summary("BTCUSDT")
    assert s["symbol"] == "BTCUSDT"
    assert "RiskEnsemble" in s["strategy"]
    assert -1.0 <= s["oos"]["max_drawdown"] <= 0.0
    assert 0.0 <= s["live"]["weight_now"] <= 1.0
    assert 2 <= len(s["curve"]) <= 200
    assert s["curve"][-1]["v"] == pytest.approx(s["oos"]["final"], rel=1e-3)
    assert "paper" in s["disclaimer"].lower()
    ts = [p["t"] for p in s["curve"]]
    assert ts == sorted(ts)


def test_build_summary_prefers_canonical_record(api):
    """When runs/canonical-v1 exists, headline NUMBERS come from the sealed
    record — same source as the README table — while curve/live stay live."""
    import os

    if not os.path.exists(api.CANONICAL_RUN):
        pytest.skip("canonical run not generated yet")
    s = api.build_summary("BTCUSDT")
    record = json.loads(open(api.CANONICAL_RUN, encoding="utf-8").read())
    iv = record["results"]["metrics"]["inv_vol_rm"]
    assert s["oos"]["sharpe"] == pytest.approx(iv["sharpe"], abs=1e-6)
    assert s["oos"]["cagr"] == pytest.approx(iv["cagr"], abs=1e-9)
    assert "canonical-v1" in s["canonical_run_id"]
    assert len(s.get("rules_table", [])) >= 1
    # curve remains the live-computed research curve (may differ from record)
    assert len(s["curve"]) >= 2


def test_asgi_app_serves_json(monkeypatch, api):
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/api/summary"}
    asyncio.run(api.app(scope, receive, send))

    start = [m for m in sent if m["type"] == "http.response.start"][0]
    body = [m for m in sent if m["type"] == "http.response.body"][0]
    assert start["status"] == 200
    headers = dict((k.decode(), v.decode()) for k, v in start["headers"])
    assert headers["content-type"] == "application/json"
    payload = json.loads(body["body"])
    # evidence-first shape: forward block always present, research nested below
    assert "forward" in payload and "research" in payload
    assert payload["research"].get("evidence_label", "").startswith("RESEARCH / HISTORICAL ONLY")
    assert "oos" in payload["research"]


def test_asgi_app_research_failure_degrades_not_500(monkeypatch, api):
    def boom(symbol):
        raise RuntimeError("network down")

    monkeypatch.setattr(api, "fetch_daily_history", boom)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(api.app({"type": "http", "method": "GET", "path": "/api/summary"}, receive, send))
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    assert start["status"] == 200  # forward evidence still served
    payload = json.loads([m for m in sent if m["type"] == "http.response.body"][0]["body"])
    assert "network down" in payload["research"]["error"]
    fwd = payload["forward"]
    # repo-root freeze.json absent in this test env -> unavailable; if a real
    # freeze exists, it must at least be intact
    assert fwd["available"] is False or fwd["code_verified"] is True


# ---------------------------------------------------------------------------
# forward (evidence-first) summary
# ---------------------------------------------------------------------------

FREEZE_TS = "2026-08-23T09:00:00+00:00"


def _write_freeze(path: Path, frozen_date="2026-08-23", commit="6d6606dabc"):

    from bot.identity import code_fingerprint
    from bot.prospective import _config_hash

    config = {"assets": [{"symbol": "BTCUSDT"}], "frictions": {"fee": 0.001}}
    cfg_sha = _config_hash(config)
    path.write_text(json.dumps({
        "frozen_at": FREEZE_TS,
        "frozen_at_date": frozen_date,
        "git_commit_at_freeze": commit,
        "config": config,
        "config_sha256": cfg_sha,
        "code_fingerprint_algo": "sha256-lf-v1",
        "code_sha256": code_fingerprint(),
    }))


def _write_log(path: Path, days=10, first="2026-08-24", ret=0.001):
    from datetime import timedelta

    d0 = datetime.fromisoformat(first)
    with open(path, "w") as f:
        for i in range(days):
            f.write(json.dumps({
                "date": (d0 + timedelta(days=i)).date().isoformat(),
                "port_ret": ret * (-1 if i % 5 == 4 else 1),
                "outages": [{"p": 1}] if i == 3 else [],
                "missed_fills": [{"s": 1}] if i == 6 else [],
            }) + "\n")


class TestForwardSummary:
    def test_unavailable_without_freeze(self, api, tmp_path):
        res = api.build_forward_summary(freeze_path=str(tmp_path / "none.json"))
        assert res["available"] is False
        assert "freeze" in res["reason"]

    def test_started_false_before_first_day(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(tmp_path / "log.jsonl"))
        assert res["available"] and not res["started"]
        assert res["parameter_changes"] == 0
        assert res["days_untouched"] is None

    def test_full_metrics_from_log(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=12)
        today = datetime(2026, 9, 20, tzinfo=UTC)  # 28 days after freeze
        res = api.build_forward_summary(
            freeze_path=str(fp), log_path=str(lp),
            benchmark_fetch=lambda: [
                {"date": "2026-08-24", "close": 5000.0},
                {"date": "2026-09-04", "close": 5150.0},
            ],
            today=today,
        )
        assert res["available"] and res["started"]
        assert res["frozen_date"] == "2026-08-23"
        assert res["code_verified"] is True
        assert res["days_untouched"] == 28
        assert res["parameter_changes"] == 0
        assert res["n_days_recorded"] == 12
        assert res["data_outages"] == 1
        assert res["missed_fills"] == 1
        assert res["benchmark_return"] == pytest.approx(0.03)
        assert -1.0 <= res["max_drawdown"] <= 0.0
        assert len(res["curve"]) == 12
        # compounding check
        eq = 1.0
        for c in res["curve"]:
            eq *= 1 + 0.001 * (-1 if res["curve"].index(c) % 5 == 4 else 1)
        assert res["forward_return"] == pytest.approx(eq - 1, abs=1e-3)

    def test_manifest_tamper_surfaces_not_verified(self, api, tmp_path):
        """Reader-side verification covers the manifest seal (config hash).
        A stale code sha is NOT flagged here — the tag pins what traded, and
        execution-time refusal guards the writer. Runtime mismatch is
        reported transparently instead."""
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        blob = json.loads(fp.read_text())
        blob["code_sha256"] = "f" * 64
        blob["config"]["frictions"]["fee"] = 9.99  # real tamper: config change
        from bot.prospective import _config_hash

        blob["config_sha256"] = _config_hash(blob["config"])  # keep config seal valid? no—recompute shows tamper path separately
        # first: config-tamper variant keeps OLD config hash -> mismatch
        blob["config_sha256"] = "e" * 64
        fp.write_text(json.dumps(blob))
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=3)
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["code_verified"] is False
        assert "config sha mismatch" in res["code_reason"]

    def test_stale_code_sha_reported_as_runtime_mismatch(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        blob = json.loads(fp.read_text())
        blob["code_sha256"] = "f" * 64  # reader-side cannot validate this value
        fp.write_text(json.dumps(blob))
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(tmp_path / "log.jsonl"))
        assert res["code_verified"] is True
        assert res["runtime_matches_freeze"] is False

    def test_benchmark_failure_degrades_to_none(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=3)

        def boom():
            raise RuntimeError("fred down")

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp),
                                        benchmark_fetch=boom)
        assert res["benchmark_return"] is None
