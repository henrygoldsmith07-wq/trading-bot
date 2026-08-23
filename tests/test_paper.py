"""Tests for paper-trading reliability: persistence, ledger, idempotency,
crash recovery, staleness alerts, audit reports, decision explanations."""
import json
import time

import pytest

from bot.paper import (
    OrderLedger,
    PaperPortfolio,
    build_idem_key,
    daily_audit_report,
    decide_orders,
    run_cycle,
    staleness_alerts,
)

DAY_MS = 86_400_000


def _candles(prices):
    """Candles ending 'now' so staleness checks see them as fresh."""
    now_ms = time.time() * 1000
    n = len(prices)
    return [
        {"open_time": int(now_ms - (n - i) * DAY_MS), "close": p}
        for i, p in enumerate(prices)
    ]


class TestOrderLedger:
    def test_append_and_replay(self, tmp_path):
        led = OrderLedger(tmp_path / "ledger.jsonl")
        assert led.entries() == []
        led.append({"kind": "fill", "idem_key": "k1", "qty": 1.0})
        led.append({"kind": "fill", "idem_key": "k2", "qty": 2.0})
        assert [e["idem_key"] for e in led.entries()] == ["k1", "k2"]
        assert led.idem_keys() == {"k1", "k2"}

    def test_torn_final_line_ignored(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        p.write_text('{"kind": "fill", "idem_key": "k1"}\n{"kind": "fil')
        led = OrderLedger(p)
        assert len(led.entries()) == 1
        assert led.idem_keys() == {"k1"}


class TestPersistenceAndRecovery:
    def test_state_survives_reload(self, tmp_path):
        pf = PaperPortfolio(start_cash=1000.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        pf.rebalance("BTC", 1.0, price=100.0, idem_key="d|BTC|REBAL|1")
        pf2 = PaperPortfolio(start_cash=1000.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        assert pf2.cash == pytest.approx(pf.cash)
        assert pf2.positions["BTC"] == pytest.approx(pf.positions["BTC"])

    def test_corrupt_state_recovers_from_ledger(self, tmp_path):
        state = tmp_path / "s.json"
        led = tmp_path / "l.jsonl"
        pf = PaperPortfolio(start_cash=2000.0, state_file=state, ledger_file=led)
        pf.rebalance("BTC", 0.5, price=50.0, idem_key="a-key")  # buy ~20 BTC
        pf.rebalance("ETH", 1.0, price=10.0, idem_key="b-key")
        final_cash = pf.cash
        # simulate a crash mid-write: garbage in the state file
        state.write_text("{ this is not json !!!")
        recovered = PaperPortfolio(start_cash=999_999.0, state_file=state, ledger_file=led)
        assert recovered.cash == pytest.approx(final_cash)
        assert recovered.positions["BTC"] == pytest.approx(pf.positions["BTC"])
        assert recovered.positions["ETH"] == pytest.approx(pf.positions["ETH"])

    def test_checksum_mismatch_triggers_recovery(self, tmp_path):
        state = tmp_path / "s.json"
        led = tmp_path / "l.jsonl"
        pf = PaperPortfolio(state_file=state, ledger_file=led)
        pf.rebalance("BTC", 1.0, 100.0, idem_key="x")
        # tamper: rewrite balances but keep valid JSON (checksum now stale)
        wrapped = json.loads(state.read_text())
        wrapped["cash"] = 999_999.0
        state.write_text(json.dumps(wrapped))
        pf2 = PaperPortfolio(state_file=state, ledger_file=led)
        assert pf2.cash < 900.0  # replayed from ledger, not the tampered value


class TestDuplicatePrevention:
    def test_same_idem_key_never_executes_twice(self, tmp_path):
        pf = PaperPortfolio(start_cash=1000.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        key = build_idem_key("BTC", "REBAL", 1.0, "2026-08-01")
        first = pf.rebalance("BTC", 1.0, price=100.0, idem_key=key)
        second = pf.rebalance("BTC", 1.0, price=100.0, idem_key=key)
        assert first.get("side") == "BUY"
        assert second == {"skipped": "duplicate_order", "idem_key": key}
        assert len(OrderLedger(tmp_path / "l.jsonl").entries()) == 1

    def test_rerun_cycle_is_noop(self, tmp_path):
        prices = _candles([50.0] * 40)
        calls = {"n": 0}

        def fetch(_sym):
            calls["n"] += 1
            return prices

        pf = PaperPortfolio(state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        r1 = run_cycle(["BTC"], lambda s, c: 1.0, fetch, pf, reports_dir=tmp_path / "r")
        r2 = run_cycle(["BTC"], lambda s, c: 1.0, fetch, pf, reports_dir=tmp_path / "r")
        buys1 = [f for f in r1["fills"] if f.get("kind") == "fill"]
        buys2 = [f for f in r2["fills"] if f.get("kind") == "fill"]
        assert len(buys1) == 1
        assert buys2 == [] or all(f.get("skipped") for f in buys2)


class TestRebalanceMechanics:
    def test_buy_sell_roundtrip_charges_fees(self, tmp_path):
        pf = PaperPortfolio(start_cash=1000.0, fee=0.001, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        pf.rebalance("BTC", 1.0, 100.0, idem_key="k1")
        # buy is clamped so cash never goes negative: spendable = 1000/1.001
        assert pf.cash == pytest.approx(0.0, abs=1e-6)
        assert pf.positions["BTC"] * 100.0 == pytest.approx(1000.0 / 1.001)
        pf.rebalance("BTC", 0.0, 100.0, idem_key="k2")
        assert "BTC" not in pf.positions
        assert pf.cash == pytest.approx((1000.0 / 1.001) * 0.999)

    def test_buy_clamped_to_available_cash(self, tmp_path):
        pf = PaperPortfolio(start_cash=100.0, fee=0.001, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        pf.rebalance("A", 0.9, 10.0, idem_key="a")  # leaves cash
        eq_before = pf.cash + pf.positions["A"] * 10.0
        pf.rebalance("B", 1.0, 5.0, idem_key="b")  # demands more than available
        assert pf.cash >= -1e-6
        assert pf.cash + sum(q * p for q, p in [(pf.positions.get("A", 0), 10.0), (pf.positions.get("B", 0), 5.0)]) <= eq_before + 1e-6

    def test_dust_skipped(self, tmp_path):
        pf = PaperPortfolio(start_cash=100.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        res = pf.rebalance("X", 0.00001, 100.0, idem_key="dust", min_notional=1.0)
        assert res["skipped"] == "below_min_notional"

    def test_fill_records_explanation(self, tmp_path):
        pf = PaperPortfolio(state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        pf.rebalance("BTC", 1.0, 42.0, idem_key="e", reason="TrendVol w=1.000 trend=up")
        entry = OrderLedger(tmp_path / "l.jsonl").entries()[0]
        assert entry["reason"] == "TrendVol w=1.000 trend=up"
        assert entry["target_weight"] == 1.0
        assert entry["fee"] > 0


class TestStalenessAlerts:
    NOW_MS = 86_400_000 * 1000

    def test_stale_symbol_flagged(self):
        fresh = [{"open_time": self.NOW_MS - DAY_MS, "close": 1.0}]
        stale = [{"open_time": self.NOW_MS - DAY_MS * 30, "close": 1.0}]
        alerts = staleness_alerts({"FRESH": fresh, "DEAD": stale}, now_ms=self.NOW_MS, max_age_days=3.0)
        symbols = {a["symbol"] for a in alerts}
        assert symbols == {"DEAD"}

    def test_empty_history_flagged(self):
        alerts = staleness_alerts({"GHOST": []}, now_ms=self.NOW_MS)
        assert alerts[0]["symbol"] == "GHOST"

    def test_fresh_within_tolerance_not_flagged(self):
        candles = [{"open_time": self.NOW_MS - 86_400_000, "close": 1.0}]
        assert staleness_alerts({"OK": candles}, now_ms=self.NOW_MS) == []


class TestDecideOrders:
    def test_weights_relative_to_total_equity(self):
        targets = {"A": (0.5, "because"), "B": (0.0, "flat")}
        positions = {"A": 5.0}  # A @ 10 = 50; cash 50 => equity 100
        decisions = decide_orders(targets, cash=50.0, current_positions=positions, prices={"A": 10.0}, stale_symbols=set())
        by_sym = {d["symbol"]: d for d in decisions}
        assert by_sym["A"]["action"] == "hold"  # already exactly at 0.5
        assert by_sym["B"]["action"] == "hold"  # flat target, flat position
        assert "because" in by_sym["A"]["explanation"]

    def test_buy_and_sell_signals(self):
        targets = {"A": (0.8, ""), "C": (0.0, "")}
        positions = {"A": 5.0, "C": 2.0}
        decisions = decide_orders(targets, cash=0.0, current_positions=positions, prices={"A": 10.0, "C": 10.0}, stale_symbols=set())
        by_sym = {d["symbol"]: d for d in decisions}
        assert by_sym["A"]["action"] == "BUY"
        assert by_sym["C"]["action"] == "SELL"

    def test_stale_blocked_with_explanation(self):
        decisions = decide_orders({"A": (1.0, "want in")}, 0.0, {}, {"A": 5.0}, stale_symbols={"A"})
        assert decisions[0]["action"] == "blocked_stale"


class TestAuditReport:
    def test_report_contains_sections(self, tmp_path):
        pf = PaperPortfolio(start_cash=1000.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        fill = pf.rebalance("BTC", 0.5, 100.0, idem_key="r1", reason="test reason")
        decisions = [{"symbol": "BTC", "action": "hold", "explanation": "at target"}]
        report = daily_audit_report(
            pf,
            {"BTC": 110.0},
            decisions,
            [f for f in [fill] if f.get("kind") == "fill"],
            [{"symbol": "ETH", "age_days": 9.5, "level": "stale_data"}],
        )
        assert "# Paper audit report" in report
        assert "## Positions" in report
        assert "| BTC |" in report
        assert "## Decisions & explanations" in report
        assert "at target" in report
        assert "test reason" in report
        assert "stale_data" in report


class TestRunCycle:
    def test_full_cycle_writes_report_and_ledger(self, tmp_path):
        data = {"BTC": _candles([100 + i for i in range(60)]), "ETH": []}
        pf = PaperPortfolio(start_cash=5000.0, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        result = run_cycle(["BTC", "ETH"], lambda s, c: 0.75, lambda s: data[s], pf, reports_dir=tmp_path / "reports")
        fills = [f for f in result["fills"] if f.get("kind") == "fill"]
        assert len(fills) == 1 and fills[0]["symbol"] == "BTC"
        blocked = [d for d in result["decisions"] if d["action"] == "blocked_stale"]
        assert {b["symbol"] for b in blocked} == {"ETH"}
        assert (tmp_path / "reports").exists()
        content = "\n".join(p.read_text() for p in (tmp_path / "reports").glob("*.md"))
        assert "forced flat" in content  # ETH's explanation is in the report
        assert "no candle data" in content

    def test_cycle_idempotent_across_reruns(self, tmp_path):
        data = {"BTC": _candles([100.0] * 30)}
        pf = PaperPortfolio(state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        run_cycle(["BTC"], lambda s, c: 0.6, lambda s: data[s], pf, reports_dir=tmp_path / "r")
        eq_after_first = pf.cash + pf.positions.get("BTC", 0.0) * 100.0
        n_entries = len(OrderLedger(tmp_path / "l.jsonl").entries())
        run_cycle(["BTC"], lambda s, c: 0.6, lambda s: data[s], pf, reports_dir=tmp_path / "r")
        assert len(OrderLedger(tmp_path / "l.jsonl").entries()) == n_entries
        assert pf.cash + pf.positions.get("BTC", 0.0) * 100.0 == pytest.approx(eq_after_first)
