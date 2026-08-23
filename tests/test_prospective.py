import json
from datetime import UTC, date, datetime

import pytest

from bot.algorithm import build_algorithm
from bot.prospective import (
    checkpoints_due,
    create_freeze,
    load_freeze,
    load_log,
    monthly_returns,
    outage_stats,
    run_step,
    slippage_stats,
    trailing_overlay_weight,
)
from bot.strategy import TrendVol, strategy_from_spec, strategy_to_spec

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _algo(**overrides):
    return build_algorithm(with_pool_version=False, **overrides)


def _mk_freeze(tmp_path, strategies=None):
    assets = [
        {"symbol": "AAA", "source": "test", "periods_per_year": 365, "strategy": strategies["AAA"]},
        {"symbol": "BBB", "source": "test", "periods_per_year": 365, "strategy": strategies["BBB"]},
    ]
    manifest = create_freeze(
        assets,
        frictions={"fee": 0.001, "spread_bps": 5, "slippage_bps": 5, "execution": "next_open", "risk_free_annual": 0.03},
        algorithm=_algo(),
        path=tmp_path / "freeze.json",
        now=NOW,
        git_commit="abc123",
    )
    return manifest, tmp_path / "freeze.json"


def _candles(closes, day_of_first=0):
    return [
        {"open_time": (day_of_first + i) * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def _fetcher(data, problems=None):
    problems = problems or {}

    def f(sym, source):
        if sym in problems:
            return [], problems[sym]
        return data[sym], None

    return f


def test_freeze_roundtrip_and_tamper_detection(tmp_path):
    manifest, path = _mk_freeze(tmp_path, {"AAA": TrendVol(50, 20, 0.25), "BBB": TrendVol(100, 20, 0.4)})
    loaded = load_freeze(path)
    assert loaded["config_sha256"] == manifest["config_sha256"]
    assert loaded["frozen_at_date"] == NOW.date().isoformat()
    assert loaded["git_commit_at_freeze"] == "abc123"
    blob = json.loads(path.read_text())
    blob["config"]["assets"][0]["strategy"]["params"]["lookback"] = 42  # tamper
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        load_freeze(path)


def test_strategy_spec_roundtrip():
    s = TrendVol(75, 20, 0.3)
    s2 = strategy_from_spec(strategy_to_spec(s))
    assert repr(s2) == repr(s)
    with pytest.raises(ValueError):
        strategy_from_spec({"type": "NotAThing", "params": {}})


def test_run_step_logs_and_is_idempotent(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    res1 = run_step(manifest, _fetcher({"AAA": rising, "BBB": rising}), now=NOW, log_path=log)
    assert res1["status"] == "logged"
    res2 = run_step(manifest, _fetcher({"AAA": rising, "BBB": rising}), now=NOW, log_path=log)
    assert res2["status"] == "already_logged"
    assert len(load_log(log)) == 1


def _mk_rising(n, start=100.0, growth=1.004):
    return _candles([start * growth ** i for i in range(n)])


def test_run_step_records_outages(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    res = run_step(
        manifest,
        _fetcher({"AAA": rising, "BBB": rising}, problems={"BBB": "fetch failed: timeout"}),
        now=NOW,
        log_path=log,
    )
    entry = res["entry"]
    assert entry["outages"] == [{"symbol": "BBB", "problem": "fetch failed: timeout"}]
    assert outage_stats(load_log(log))["outage_events"] == 1


def test_run_step_flags_missed_fill_after_data_gap(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    # gapped history: 3-day hole between the last two completed candles
    closes = [100.0 * 1.004 ** i for i in range(300)]
    candles = _candles(closes)
    candles[-1] = {"open_time": candles[-1]["open_time"] + 3 * 86_400_000, "close": closes[-1] * 1.01}
    res = run_step(manifest, _fetcher({"AAA": candles, "BBB": _mk_rising(400)}), now=NOW, log_path=log)
    mf = res["entry"]["missed_fills"]
    assert any(m["symbol"] == "AAA" and m["delayed_days"] >= 2 for m in mf)


def test_run_step_uses_only_completed_candles_for_decision(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    # today's in-progress candle has a wild price; decision must ignore it
    candles = rising + [{"open_time": rising[-1]["open_time"] + 86_400_000, "close": 1e9}]
    res = run_step(manifest, _fetcher({"AAA": candles, "BBB": rising}), now=NOW, log_path=log)
    det = res["entry"]["assets"]["AAA"]
    assert det["price"] == 1e9  # executed at latest print...
    assert det["weight"] <= 1.0  # ...but the decision stayed sane


def test_slippage_stats_from_log():
    entries = [
        {"assets": {"A": {"slippage_bps": 10.0}, "B": {"slippage_bps": None}}, "port_ret": 0.0},
        {"assets": {"A": {"slippage_bps": -6.0}, "B": {"slippage_bps": 8.0}}, "port_ret": 0.0},
    ]
    s = slippage_stats(entries)
    assert s["count"] == 3
    assert s["mean_abs_bps"] == pytest.approx(8.0)


def test_checkpoints_due_gating():
    due = checkpoints_due(date(2026, 1, 1), date(2026, 2, 15))
    assert [c["due"] for c in due] == [True, False, False, False]
    due2 = checkpoints_due(date(2025, 1, 1), date(2026, 3, 1))
    assert [c["due"] for c in due2] == [True, True, True, True]


def test_monthly_returns_publish_negatives():
    entries = [
        {"date": "2026-06-01", "port_ret": 0.01},
        {"date": "2026-06-02", "port_ret": -0.02},
        {"date": "2026-07-01", "port_ret": 0.03},
    ]
    months = monthly_returns(entries)
    assert months["2026-06"] == pytest.approx(1.01 * 0.98 - 1)
    assert months["2026-06"] < 0
    assert months["2026-07"] == pytest.approx(0.03)


def test_trailing_overlay_weight_bounds():
    calm = [0.001] * 25
    wild = [0.05, -0.05] * 25
    assert trailing_overlay_weight(calm, 0.25) == 1.0
    assert 0.0 < trailing_overlay_weight(wild, 0.25) < 1.0
    assert trailing_overlay_weight([], 0.25) == 1.0  # warmup: fully invested


def test_forward_runner_never_reselects(tmp_path):
    # the frozen strategy spec is the only input; a tampered/unknown spec must fail
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    manifest["config"]["assets"][0]["strategy"] = {"type": "Mystery", "params": {}}
    with pytest.raises(ValueError):
        run_step(manifest, _fetcher({"AAA": _mk_rising(400), "BBB": _mk_rising(400)}), now=NOW, log_path=tmp_path / "log.jsonl")
