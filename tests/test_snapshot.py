"""Reproducible-snapshot tests: the deterministic pipeline must re-derive
the committed benchmark numbers bit-for-bit (same seed => same metrics)."""
import pytest

from bot.snapshot import (
    dataset_hash,
    load_snapshot,
    make_snapshot,
    verify_snapshot,
    write_snapshot,
)


def test_dataset_hash_stable_and_sensitive():
    a = [{"open_time": 0, "close": 1.0}]
    b = [{"open_time": 0, "close": 1.0}]
    c = [{"open_time": 0, "close": 2.0}]
    assert dataset_hash(a) == dataset_hash(b)
    assert dataset_hash(a) != dataset_hash(c)
    assert dataset_hash([*a, *a]) != dataset_hash(a)  # order/length sensitive


def test_snapshot_roundtrip_and_verify(tmp_path):
    snap = make_snapshot(
        "unit",
        config={"fee": 0.001, "seed": 42},
        data_hashes={"BTC": "abc123"},
        metrics={"sharpe": 1.234, "cagr": 0.19},
        seed=42,
        git_commit="deadbeef",
    )
    path = write_snapshot(snap, tmp_path / "snap" / "unit.json")
    loaded = load_snapshot(path)
    assert loaded["name"] == "unit"
    res = verify_snapshot(path, {"sharpe": 1.234, "cagr": 0.19}, {"BTC": "abc123"})
    assert res["ok"] and res["metric_drifts"] == []


def test_verify_detects_metric_drift(tmp_path):
    snap = make_snapshot("drift", {}, {}, {"sharpe": 1.0})
    path = write_snapshot(snap, tmp_path / "s.json")
    res = verify_snapshot(path, {"sharpe": 0.9})
    assert not res["ok"]
    assert res["metric_drifts"][0]["metric"] == "sharpe"


def test_verify_detects_extra_and_missing_metrics(tmp_path):
    snap = make_snapshot("keys", {}, {}, {"a": 1.0})
    path = write_snapshot(snap, tmp_path / "s.json")
    res = verify_snapshot(path, {"b": 2.0})
    assert not res["ok"]
    metrics = {d["metric"]: d for d in res["metric_drifts"]}
    assert metrics["b"]["expected"] is None
    assert metrics["a"]["actual"] is None


def test_verify_detects_data_change(tmp_path):
    snap = make_snapshot("data", {}, {"SPY": "h1"}, {"x": 1.0})
    path = write_snapshot(snap, tmp_path / "s.json")
    ok = verify_snapshot(path, {"x": 1.0}, {"SPY": "h1"})
    changed = verify_snapshot(path, {"x": 1.0}, {"SPY": "h2"})
    assert ok["ok"]
    assert not changed["ok"] and changed["data_changed"] == ["SPY"]


def test_nan_metrics_match_exactly(tmp_path):
    snap = make_snapshot("nan", {}, {}, {"weird": float("nan")})
    path = write_snapshot(snap, tmp_path / "s.json")
    assert verify_snapshot(path, {"weird": float("nan")})["ok"]
    res = verify_snapshot(path, {"weird": 1.0})
    assert not res["ok"]


def test_unsupported_version_raises(tmp_path):
    import json as _json

    p = tmp_path / "bad.json"
    p.write_text(_json.dumps({"snapshot_version": 99}))
    with pytest.raises(ValueError, match="version"):
        load_snapshot(p)
