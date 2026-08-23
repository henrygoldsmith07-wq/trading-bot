import asyncio
import importlib.util
import json
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


def test_build_summary_shape(monkeypatch, api):
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
    assert "oos" in payload and "curve" in payload


def test_asgi_app_clean_error_on_failure(monkeypatch, api):
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
    body = [m for m in sent if m["type"] == "http.response.body"][0]
    assert start["status"] == 502
    assert "network down" in json.loads(body["body"])["error"]
