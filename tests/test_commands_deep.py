"""Deep CLI command coverage: the report commands run end to end on fixtures.

`bot/commands.py` carried zero test coverage at HEAD, which is what kept the
coverage gate failing. These tests exercise the heavy commands — the
statistical battery, the research battery, sensitivity, freezing and the
forward report — against synthetic candles and monkeypatched data layers.
No network is touched, and no committed evidence tape is written.

Invariants pinned here:
* the pure data-handling helpers (gaps, forward-fill, timeline placement,
  delisting) implement the documented survivorship-safe semantics;
* a freeze created through the CLI seals the PRIMARY rule's algorithm;
* the forward report never writes to the log it is reading.
"""
from __future__ import annotations

import argparse
import json
import time as _time
from pathlib import Path

import pytest

DAY = 86_400_000
# Histories must END near "now": `is_stale` drops any asset whose last print is
# older than 45 days, so a fixture frozen in 2020 reads as a delisted asset.
T0 = int(_time.time() * 1000) - 900 * DAY


def _candles(n=900, start=100.0, drift=0.0004):
    out = []
    px = start
    for i in range(n):
        px *= 1.0 + drift + 0.01 * ((i % 7) - 3) / 100
        out.append({
            "open_time": T0 + i * DAY,
            "open": round(px * 0.999, 8),
            "high": round(px * 1.002, 8),
            "low": round(px * 0.998, 8),
            "close": round(px, 8),
            "volume": 1000.0 + i,
            "quote_volume": 1_000_000.0 + i,
        })
    return out


def _patch_market_data(monkeypatch, candles=None):
    """Every data source returns the same synthetic history; no network."""
    data = candles or _candles()
    monkeypatch.setattr("bot.cache.load_or_fetch", lambda sym, fn, **kw: (data, True))
    monkeypatch.setattr("bot.data.fetch_daily_history", lambda *a, **k: data)
    monkeypatch.setattr("bot.data.fetch_yahoo_daily", lambda *a, **k: data)
    monkeypatch.setattr("bot.universe.top_symbols", lambda n=20: ["BTCUSDT", "ETHUSDT", "SOLUSDT"][: n + 1])
    monkeypatch.setattr("bot.benchmark.fetch_sp500", lambda: [
        {"date": __import__("datetime").date.fromtimestamp((T0 + i * DAY) / 1000, tz=__import__("datetime").UTC),
         "close": 100.0 + i * 0.1}
        for i in range(0, 900, 3)
    ])
    return data


# ---------------------------------------------------------------------------
# bot/data.py pure helpers
# ---------------------------------------------------------------------------

class TestDataHelpers:
    def test_is_stale_detects_a_stopped_history(self):
        from bot.data import is_stale

        now = T0 + 900 * DAY
        assert is_stale(_candles(10), now_ms=now) is True
        assert is_stale(_candles(10), now_ms=T0 + 5 * DAY) is False
        assert is_stale([], now_ms=now) is True

    def test_gap_report_finds_only_real_gaps(self):
        from bot.data import gap_report

        c = _candles(3)
        holes = [c[0], {**c[1], "open_time": c[0]["open_time"] + 5 * DAY}, c[2]]
        gaps = gap_report(holes)
        assert len(gaps) == 1
        assert gaps[0]["days"] == pytest.approx(5.0)
        assert gap_report(_candles(3)) == []

    def test_fill_small_gaps_flags_filled_bars(self):
        from bot.data import fill_small_gaps

        c = _candles(2)
        gapped = [c[0], {**c[1], "open_time": c[0]["open_time"] + 3 * DAY, "close": 42.0}]
        out = fill_small_gaps(gapped, max_gap_days=3)
        filled = [x for x in out if x.get("filled")]
        assert len(filled) == 2
        assert all(x["close"] == c[0]["close"] for x in filled)
        assert out[-1]["close"] == 42.0  # the real print survives

    def test_fill_leaves_long_outages_visible(self):
        from bot.data import fill_small_gaps

        c = _candles(2)
        gapped = [c[0], {**c[1], "open_time": c[0]["open_time"] + 30 * DAY}]
        out = fill_small_gaps(gapped, max_gap_days=3)
        assert not [x for x in out if x.get("filled")]

    def test_timeline_placement_holds_cash_before_and_after(self):
        from bot.data import extend_returns_to_timeline

        timeline = [T0 + i * DAY for i in range(5)]
        daily = {T0 + DAY: 0.01, T0 + 3 * DAY: 0.02}
        out = extend_returns_to_timeline(daily, timeline)
        # interior gap is an INVESTED flat day
        assert out[T0 + 2 * DAY] == 0.0
        # before first and after last the sleeve sits in cash: ABSENT
        assert T0 not in out
        assert T0 + 4 * DAY not in out
        assert extend_returns_to_timeline({}, timeline) == {}

    def test_delisting_truncates_and_marks_down(self):
        from bot.data import simulate_delisting

        c = _candles(5)
        out = simulate_delisting(c, delist_at_ms=c[2]["open_time"], terminal_cost_bps=100.0)
        assert len(out) == 3
        assert out[-1]["note"] == "simulated delisting"
        assert out[-1]["close"] == pytest.approx(c[2]["close"] * 0.99)
        assert simulate_delisting(c, delist_at_ms=c[0]["open_time"] - DAY) == []

    def test_recent_window_slices_the_tail(self):
        from bot.data import recent_window

        c = _candles(10)
        assert recent_window(c, 3) == c[-3:]
        assert recent_window(c, 50) == c

    def test_yahoo_fetch_parses_and_cleans(self, monkeypatch):
        import bot.data as data

        # Yahoo timestamps are whole SECONDS; the parser converts with int(t)*1000,
        # so expectations must be second-aligned too.
        t0 = (T0 // 1000) * 1000
        payload = json.dumps({"chart": {"result": [{
            "timestamp": [t0 // 1000, t0 // 1000 + 86400, t0 // 1000 + 2 * 86400],
            "indicators": {"quote": [{
                "open": [100.0, None, 103.0],
                "close": [101.0, -5.0, 104.0],
                "volume": [10.0, 11.0, None],
            }]},
        }]}})
        monkeypatch.setattr(data, "_get", lambda url: payload)
        out = data.fetch_yahoo_daily("SPY")
        # the negative close row is dropped, the null open tolerated
        assert [c["open_time"] for c in out] == [t0, t0 + 2 * DAY]
        assert out[1]["open"] == 103.0
        assert "volume" not in out[1]


# ---------------------------------------------------------------------------
# the heavy report commands, on synthetic fixtures
# ---------------------------------------------------------------------------

def _validate_ns(**kw):
    base = dict(symbol="BTCUSDT", train_days=300, test_days=150, fee=0.001, spread_bps=5.0,
                slippage_bps=5.0, execution="next_open", risk_free=0.03, embargo_days=5,
                purge_days=20, boots=10, rc_boots=5, seed=42)
    base.update(kw)
    return argparse.Namespace(**base)


class TestReportCommands:
    def test_sensitivity_sweep_runs_on_synthetic_data(self, monkeypatch, capsys):
        from bot.commands import run_sensitivity

        _patch_market_data(monkeypatch)
        code = run_sensitivity(_validate_ns())
        out = capsys.readouterr().out
        assert code == 0
        assert "Parameter sensitivity" in out
        assert "Transaction-cost sensitivity" in out

    def test_statistical_battery_runs_and_reports_deflation(self, monkeypatch, capsys):
        from bot.commands import run_validate

        _patch_market_data(monkeypatch)
        code = run_validate(_validate_ns())
        out = capsys.readouterr().out
        assert code == 0
        assert "PSR" in out and "DSR" in out
        assert "Nested walk-forward" in out

    def test_research_battery_runs(self, monkeypatch, capsys):
        from bot.commands import run_research

        _patch_market_data(monkeypatch)
        code = run_research(_validate_ns())
        out = capsys.readouterr().out
        assert code == 0
        assert "SPA" in out or "research" in out.lower()

    def test_thin_history_refuses_the_battery(self, monkeypatch, capsys):
        from bot.commands import run_validate

        _patch_market_data(monkeypatch, candles=_candles(5))
        code = run_validate(_validate_ns())
        assert code == 2
        assert "Not enough history" in capsys.readouterr().out


class TestFreezeCommand:
    def _ns(self, freeze_file: Path, **kw):
        base = dict(assets=2, train_days=300, test_days=150, fee=0.001, spread_bps=5.0,
                    slippage_bps=5.0, execution="next_open", risk_free=0.03, portfolio_vol=0.25,
                    embargo_days=5, freeze_file=str(freeze_file), allow_dirty_tree=True, no_tag=True,
                    tag=None, image_digest=None, fixed=False, band=0.05, no_tilt=False,
                    tilt_lookback=90, max_tilt=0.5, no_crisis=False, corr_window=60,
                    corr_threshold=0.60, derisk=0.60, throttle=False, dd_trigger=-0.10,
                    dd_exit=-0.05, throttle_factor=0.5, vol_window=20, max_multiple_of_equal=2.0,
                    no_overlay=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_freeze_seals_the_primary_rule(self, monkeypatch, tmp_path, capsys):
        from bot.canonical_identity import PRIMARY_RULE_ID
        from bot.commands import run_freeze

        _patch_market_data(monkeypatch)
        target = tmp_path / "freeze.json"
        code = run_freeze(self._ns(target))
        out = capsys.readouterr().out
        assert code == 0, out
        manifest = json.loads(target.read_text(encoding="utf-8"))
        from bot.canonical_identity import assert_primary_rule_is_frozen

        assert assert_primary_rule_is_frozen(manifest["config"]["algorithm"]) == PRIMARY_RULE_ID
        assert manifest["config_sha256"] and manifest["code_sha256"]
        assert "Froze" in out

    def test_freeze_refuses_a_non_primary_rule(self, monkeypatch, tmp_path, capsys):
        from bot.commands import run_freeze

        _patch_market_data(monkeypatch)
        target = tmp_path / "freeze.json"
        # disabling both overlays makes the config map to inv_vol_rm, which is
        # a comparator: the CLI must refuse to freeze it as the experiment
        code = run_freeze(self._ns(target, no_tilt=True, no_crisis=True, band=0.0))
        assert code == 1
        assert not target.exists()
        assert "does not implement the canonical primary rule" in capsys.readouterr().out


class TestForwardReport:
    def test_report_reads_the_log_without_writing(self, monkeypatch, tmp_path, capsys):
        from bot.commands import run_forward
        from bot.prospective import experiment_stamp
        from tests.test_commands import _freeze_manifest

        manifest = _freeze_manifest()
        freeze_path = tmp_path / "freeze.json"
        freeze_path.write_text(json.dumps(manifest), encoding="utf-8")
        log = tmp_path / "forward_log.jsonl"
        rows = [
            {"date": "2026-08-24", "ts": "2026-08-24T23:00:00+00:00", "port_ret": 0.001,
             "rule_ret": 0.001, "overlay_weight": 1.0, "exposure": 1.0, "throttled": False,
             "dayStatus": "traded",
             "assets": {"BTCUSDT": {"weight": 1.0, "target": 1.0, "price": 100.0,
                                    "mark_price": 100.0, "exec_price": 100.0,
                                    "decision_close": 100.0, "accounting_anchor": 100.0,
                                    "overnight": 0.0, "intraday": 0.001, "sleeve_ret": 0.001,
                                    "slippage_bps": 0.0}},
             "outages": [], "missed_fills": [], "orders": [],
             "experiment": experiment_stamp(manifest)},
        ]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        before = log.read_text(encoding="utf-8")
        monkeypatch.setattr("bot.benchmark.fetch_sp500", lambda: [])

        code = run_forward(argparse.Namespace(
            step=False, report=True, freeze_file=str(freeze_path),
            log_file=str(log), as_of_date=None,
        ))
        out = capsys.readouterr().out
        assert code == 0
        assert "Prospective validation" in out
        assert "Checkpoints" in out
        assert log.read_text(encoding="utf-8") == before, "a report must never rewrite the evidence tape"
