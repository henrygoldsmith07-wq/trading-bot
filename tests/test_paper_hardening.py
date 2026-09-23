"""Regression tests for paper-broker fail-closed reliability guarantees."""

import json
from datetime import UTC, datetime

import pytest

from bot.paper import (
    LedgerCorruptionError,
    OrderLedger,
    PaperPortfolio,
    decide_orders,
    run_cycle,
)

NOW = datetime(2026, 9, 22, 17, 0, tzinfo=UTC)
DAY_MS = 86_400_000


def _candles(prices):
    end_ms = int(NOW.timestamp() * 1000)
    n = len(prices)
    return [
        {"open_time": end_ms - (n - i) * DAY_MS, "close": price}
        for i, price in enumerate(prices)
    ]


class TestLedgerFailClosed:
    def test_interior_json_corruption_is_rejected(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            '{"kind":"note","idem_key":"a"}\n'
            '{"kind":BROKEN}\n'
            '{"kind":"note","idem_key":"b"}\n',
            encoding="utf-8",
        )
        with pytest.raises(LedgerCorruptionError, match="line 2"):
            OrderLedger(path).entries()

    def test_append_repairs_torn_final_record(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            '{"kind":"note","idem_key":"old"}\n{"kind":BROKEN',
            encoding="utf-8",
        )
        ledger = OrderLedger(path)
        ledger.append({"kind": "note", "idem_key": "new"})
        assert [e["idem_key"] for e in ledger.entries()] == ["old", "new"]

    def test_append_restores_missing_separator_after_complete_record(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text('{"kind":"note","idem_key":"old"}', encoding="utf-8")
        ledger = OrderLedger(path)
        ledger.append({"kind": "note", "idem_key": "new"})
        assert [e["idem_key"] for e in ledger.entries()] == ["old", "new"]

    def test_nonfinite_values_cannot_be_appended(self, tmp_path):
        ledger = OrderLedger(tmp_path / "ledger.jsonl")
        with pytest.raises(ValueError):
            ledger.append({"kind": "fill", "price": float("nan")})
        assert ledger.entries() == []

    def test_valid_but_stale_state_is_reconciled_from_newer_ledger(self, tmp_path):
        state = tmp_path / "state.json"
        ledger = tmp_path / "ledger.jsonl"
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=state,
            ledger_file=ledger,
        )
        pf.rebalance("BTC", 0.5, 100.0, idem_key="first")
        stale_state = state.read_text(encoding="utf-8")

        pf.rebalance("BTC", 1.0, 100.0, idem_key="second")
        expected_cash = pf.cash
        expected_qty = pf.positions["BTC"]

        # Crash-window simulation: the second ledger record is durable, but
        # the old state snapshot is what survives on disk.
        state.write_text(stale_state, encoding="utf-8")

        recovered = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=state,
            ledger_file=ledger,
        )
        assert recovered.cash == pytest.approx(expected_cash)
        assert recovered.positions["BTC"] == pytest.approx(expected_qty)
        persisted = json.loads(state.read_text(encoding="utf-8"))
        assert persisted["cash"] == pytest.approx(expected_cash)


    def test_traded_state_without_ledger_is_rejected(self, tmp_path):
        state = tmp_path / "state.json"
        ledger = tmp_path / "ledger.jsonl"
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=state,
            ledger_file=ledger,
        )
        pf.rebalance("BTC", 0.5, 100.0, idem_key="first")
        ledger.unlink()

        with pytest.raises(LedgerCorruptionError, match="ledger is empty/missing"):
            PaperPortfolio(
                start_cash=1000.0,
                fee=0.0,
                state_file=state,
                ledger_file=ledger,
            )


class TestOrderPreflight:
    def test_nan_price_is_rejected_without_persistent_side_effects(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        pf = PaperPortfolio(
            start_cash=1000.0,
            state_file=tmp_path / "state.json",
            ledger_file=ledger,
        )
        result = pf.rebalance("BTC", 0.5, float("nan"), idem_key="nan-price")
        assert result == {"skipped": "no usable price for BTC"}
        assert OrderLedger(ledger).entries() == []
        assert pf.cash == pytest.approx(1000.0)
        assert pf.positions == {}

    def test_nonfinite_target_is_blocked_before_an_order_is_built(self):
        decision = decide_orders(
            {"BTC": (float("nan"), "bad signal")},
            cash=1000.0,
            current_positions={},
            prices={"BTC": 100.0},
            stale_symbols=set(),
            as_of=NOW,
        )[0]
        assert decision["action"] == "blocked_invalid_target"
        assert "idem_key" not in decision

    def test_idempotency_date_comes_from_cycle_timestamp(self):
        decision = decide_orders(
            {"BTC": (0.5, "signal")},
            cash=1000.0,
            current_positions={},
            prices={"BTC": 100.0},
            stale_symbols=set(),
            as_of=NOW,
        )[0]
        assert decision["idem_key"].startswith("2026-09-22|BTC|REBAL|")

    def test_cash_clamp_does_not_record_zero_quantity_fill(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=ledger,
        )
        pf.rebalance("A", 1.0, 100.0, idem_key="fund-a", market_prices={"A": 100.0})
        count_before = len(OrderLedger(ledger).entries())

        result = pf.rebalance(
            "B",
            1.0,
            100.0,
            idem_key="buy-b",
            market_prices={"A": 100.0, "B": 100.0},
        )
        assert result["skipped"] == "insufficient_cash"
        assert len(OrderLedger(ledger).entries()) == count_before
        assert "B" not in pf.positions


class TestCycleResilience:
    def test_sells_execute_before_buys_even_when_symbol_order_is_reversed(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance("A", 1.0, 100.0, idem_key="seed-a", market_prices={"A": 100.0})

        data = {"A": _candles([100.0] * 5), "B": _candles([100.0] * 5)}
        result = run_cycle(
            ["B", "A"],
            lambda sym, _candles_: 1.0 if sym == "B" else 0.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        fills = [f for f in result["fills"] if f.get("kind") == "fill"]
        assert [f["symbol"] for f in fills] == ["A", "B"]
        assert "A" not in pf.positions
        assert pf.positions["B"] == pytest.approx(10.0)
        assert pf.cash == pytest.approx(0.0)

    def test_data_fetch_failure_isolated_and_audited(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        btc = _candles([100.0] * 5)

        def fetch(sym):
            if sym == "ETH":
                raise TimeoutError("provider timeout")
            return btc

        result = run_cycle(
            ["BTC", "ETH"],
            lambda _sym, _candles_: 0.5,
            fetch,
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        assert any(a["level"] == "data_fetch_error" and a["symbol"] == "ETH" for a in result["alerts"])
        assert any(d["symbol"] == "ETH" and d["action"] == "blocked_stale" for d in result["decisions"])
        assert pf.positions["BTC"] == pytest.approx(5.0)
        assert "data_fetch_error" in result["report"]


    def test_overallocated_targets_are_normalized_proportionally(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        data = {"A": _candles([100.0] * 5), "B": _candles([100.0] * 5)}

        result = run_cycle(
            ["A", "B"],
            lambda _sym, _candles_: 1.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        by_symbol = {d["symbol"]: d for d in result["decisions"]}
        assert by_symbol["A"]["target_weight"] == pytest.approx(0.5)
        assert by_symbol["B"]["target_weight"] == pytest.approx(0.5)
        assert pf.positions["A"] == pytest.approx(5.0)
        assert pf.positions["B"] == pytest.approx(5.0)
        assert pf.cash == pytest.approx(0.0)
        assert any(a["level"] == "target_weights_normalized" for a in result["alerts"])

    def test_blocked_holding_reserves_portfolio_capacity(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance("A", 0.5, 100.0, idem_key="seed-a", market_prices={"A": 100.0})

        stale_open = int(NOW.timestamp() * 1000) - 10 * DAY_MS
        data = {
            "A": [{"open_time": stale_open, "close": 100.0}],
            "B": _candles([100.0] * 5),
        }
        result = run_cycle(
            ["A", "B"],
            lambda sym, _candles_: 0.5 if sym == "A" else 1.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        b = next(d for d in result["decisions"] if d["symbol"] == "B")
        assert b["target_weight"] == pytest.approx(0.5)
        assert pf.positions["A"] == pytest.approx(5.0)
        assert pf.positions["B"] == pytest.approx(5.0)
        assert pf.cash == pytest.approx(0.0)

    def test_audit_report_never_values_missing_held_mark_at_zero(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance("A", 0.5, 100.0, idem_key="seed-a", market_prices={"A": 100.0})

        result = run_cycle(
            ["B"],
            lambda _sym, _candles_: 0.0,
            lambda _sym: _candles([100.0] * 5),
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        assert "Equity: unavailable (missing marks: A)" in result["report"]
        assert "| A | 5.000000 | n/a | n/a | n/a |" in result["report"]
        assert any(d["action"] == "blocked_missing_mark" for d in result["decisions"])

    def test_ai_commentary_failure_cannot_erase_a_completed_cycle(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )

        def broken_ai(_report):
            raise RuntimeError("provider unavailable")

        result = run_cycle(
            ["BTC"],
            lambda _sym, _candles_: 0.5,
            lambda _sym: _candles([100.0] * 5),
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
            ai_note_fn=broken_ai,
        )

        assert any(a["level"] == "ai_commentary_error" for a in result["alerts"])
        assert "AI commentary" in result["report"]
        assert "(unavailable)" in result["report"]
        assert (tmp_path / "reports" / "audit_2026-09-22.md").exists()

    def test_overallocated_targets_are_scaled_proportionally(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        data = {"A": _candles([100.0] * 5), "B": _candles([100.0] * 5)}

        result = run_cycle(
            ["B", "A"],
            lambda _sym, _candles_: 1.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        assert pf.positions["A"] == pytest.approx(5.0)
        assert pf.positions["B"] == pytest.approx(5.0)
        assert pf.cash == pytest.approx(0.0)
        assert any(a["level"] == "target_weights_scaled" for a in result["alerts"])
        targets = {d["symbol"]: d.get("target_weight") for d in result["decisions"]}
        assert targets == {"B": pytest.approx(0.5), "A": pytest.approx(0.5)}

    def test_custom_gross_cap_is_respected(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        data = {"A": _candles([100.0] * 5), "B": _candles([100.0] * 5)}

        run_cycle(
            ["A", "B"],
            lambda _sym, _candles_: 1.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            max_gross_exposure=0.6,
            now=NOW,
        )

        assert pf.positions["A"] == pytest.approx(3.0)
        assert pf.positions["B"] == pytest.approx(3.0)
        assert pf.cash == pytest.approx(400.0)

    def test_malformed_candle_payload_is_blocked_not_crashed(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )

        result = run_cycle(
            ["BTC"],
            lambda _sym, _candles_: 1.0,
            lambda _sym: [{"open_time": int(NOW.timestamp() * 1000), "close": 100.0}, "broken"],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        assert pf.positions == {}
        assert any(a["level"] == "invalid_candle_payload" for a in result["alerts"])
        assert result["decisions"][0]["action"] == "blocked_stale"

    def test_future_dated_market_data_is_blocked(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        future_ms = int(NOW.timestamp() * 1000 + DAY_MS)

        result = run_cycle(
            ["BTC"],
            lambda _sym, _candles_: 1.0,
            lambda _sym: [{"open_time": future_ms, "close": 100.0}],
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )

        assert pf.positions == {}
        assert any(a["level"] == "future_data" for a in result["alerts"])
        assert result["decisions"][0]["action"] == "blocked_stale"

    def test_duplicate_symbols_are_rejected_before_fetch(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        calls = {"n": 0}

        def fetch(_sym):
            calls["n"] += 1
            return _candles([100.0])

        with pytest.raises(ValueError, match="unique"):
            run_cycle(
                ["btc", "BTC"],
                lambda _sym, _candles_: 0.5,
                fetch,
                pf,
                reports_dir=tmp_path / "reports",
                now=NOW,
            )
        assert calls["n"] == 0

    def test_blocked_holding_consumes_custom_gross_cap(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance("A", 0.4, 100.0, idem_key="seed-a", market_prices={"A": 100.0})
        stale_time = int(NOW.timestamp() * 1000 - 10 * DAY_MS)
        data = {
            "A": [{"open_time": stale_time, "close": 100.0}],
            "B": _candles([100.0] * 5),
        }

        result = run_cycle(
            ["A", "B"],
            lambda sym, _candles_: 0.0 if sym == "A" else 1.0,
            lambda sym: data[sym],
            pf,
            reports_dir=tmp_path / "reports",
            max_gross_exposure=0.6,
            now=NOW,
        )

        assert pf.positions["A"] == pytest.approx(4.0)
        assert pf.positions["B"] == pytest.approx(2.0)
        assert pf.cash == pytest.approx(400.0)
        assert result["decisions"][0]["action"] == "blocked_stale"
        assert any(a["level"] == "target_weights_scaled" for a in result["alerts"])

    def test_cycle_cannot_omit_an_existing_holding(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance("A", 0.5, 100.0, idem_key="seed-a", market_prices={"A": 100.0})

        with pytest.raises(ValueError, match="include all held positions"):
            run_cycle(
                ["B"],
                lambda _sym, _candles_: 0.5,
                lambda _sym: _candles([100.0] * 5),
                pf,
                reports_dir=tmp_path / "reports",
                now=NOW,
            )
