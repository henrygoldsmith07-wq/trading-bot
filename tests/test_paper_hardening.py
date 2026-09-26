"""Regression tests for paper-broker fail-closed reliability guarantees."""

import json
from datetime import UTC, datetime

import pytest

from bot.paper import (
    LedgerCorruptionError,
    OrderLedger,
    PaperPortfolio,
    _load_state,
    _save_state_atomic,
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
    def test_final_non_object_record_is_rejected_before_append(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(LedgerCorruptionError, match="final record is not an object"):
            OrderLedger(path).append({"kind": "note", "idem_key": "new"})

    def test_non_object_and_blank_ledger_lines_are_handled_strictly(self, tmp_path):
        blank = tmp_path / "blank.jsonl"
        blank.write_text("\n  \n{\"kind\":\"note\",\"idem_key\":\"ok\"}\n", encoding="utf-8")
        assert OrderLedger(blank).idem_keys() == {"ok"}

        bad = tmp_path / "bad.jsonl"
        bad.write_text("[]\n", encoding="utf-8")
        with pytest.raises(LedgerCorruptionError, match="not an object"):
            OrderLedger(bad).entries()

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

    @pytest.mark.parametrize("start_cash", [-1.0, float("nan")])
    def test_invalid_start_cash_is_rejected(self, tmp_path, start_cash):
        with pytest.raises(ValueError, match="start_cash"):
            PaperPortfolio(start_cash=start_cash, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")

    @pytest.mark.parametrize("fee", [-0.1, float("nan")])
    def test_invalid_fee_is_rejected(self, tmp_path, fee):
        with pytest.raises(ValueError, match="fee"):
            PaperPortfolio(fee=fee, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")

    def test_load_state_rejects_structural_and_numeric_corruption(self, tmp_path):
        path = tmp_path / "state.json"
        assert _load_state(path) is None
        path.write_text("{broken", encoding="utf-8")
        assert _load_state(path) is None
        path.write_text("[]", encoding="utf-8")
        assert _load_state(path) is None

        _save_state_atomic({"cash": 100.0, "positions": {"A": 1.0}}, path)
        valid = json.loads(path.read_text(encoding="utf-8"))

        wrong_checksum = dict(valid)
        wrong_checksum["cash"] = 99.0
        path.write_text(json.dumps(wrong_checksum), encoding="utf-8")
        assert _load_state(path) is None

        for state in (
            {"cash": -2.0, "positions": {}},
            {"cash": 100.0, "positions": []},
            {"cash": 100.0, "positions": {"": 1.0}},
            {"cash": 100.0, "positions": {"A": -1.0}},
        ):
            _save_state_atomic(state, path)
            assert _load_state(path) is None

    def _seed_two_fill_ledger(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=ledger_path,
        )
        pf.rebalance("A", 0.5, 100.0, idem_key="first", market_prices={"A": 100.0})
        pf.rebalance("A", 0.8, 100.0, idem_key="second", market_prices={"A": 100.0})
        return ledger_path

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda rows: rows[0].pop("fee"), "missing fields"),
            (lambda rows: rows[1].update({"idem_key": rows[0]["idem_key"]}), "duplicate/empty idempotency"),
            (lambda rows: rows[0].update({"side": "HOLD"}), "invalid symbol/side"),
            (lambda rows: rows[0].update({"price": float("nan")}), "non-finite number"),
            (lambda rows: rows[0].update({"price": 0.0}), "impossible balance/price"),
            (lambda rows: rows[0].update({"side": "SELL"}), "side disagrees with quantity"),
            (lambda rows: rows[1].update({"cash_before": rows[1]["cash_before"] + 10}), "cash continuity"),
            (lambda rows: rows[0].update({"notional": rows[0]["notional"] + 10}), "notional is inconsistent"),
            (lambda rows: rows[0].update({"cash_after": rows[0]["cash_after"] + 10}), "cash arithmetic"),
            (lambda rows: rows[0].update({"position_after": rows[0]["position_after"] + 1}), "position continuity"),
        ],
    )
    def test_ledger_recovery_rejects_inconsistent_fill_fields(self, tmp_path, mutate, match):
        ledger_path = self._seed_two_fill_ledger(tmp_path)
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(LedgerCorruptionError, match=match):
            PaperPortfolio(
                start_cash=1000.0,
                fee=0.0,
                state_file=tmp_path / "recovered.json",
                ledger_file=ledger_path,
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

    def test_presized_rebalance_rejects_invalid_inputs_and_dust(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=100.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        common = {"symbol": "A", "price": 10.0, "target_weight": 0.5}
        assert pf.rebalance_quantity(**common, target_quantity=float("nan"), idem_key="bad-q")["skipped"] == "invalid_target_quantity"
        assert pf.rebalance_quantity(**common, target_quantity=-1.0, idem_key="neg-q")["skipped"] == "invalid_target_quantity"
        assert pf.rebalance_quantity("A", 1.0, 0.0, "bad-px", target_weight=0.5)["skipped"] == "no usable price for A"
        assert pf.rebalance_quantity("A", 1.0, 10.0, "bad-w", target_weight=2.0)["skipped"] == "invalid_target_weight"
        assert pf.rebalance_quantity("A", 1.0, 10.0, "bad-min", target_weight=0.5, min_notional=-1.0)["skipped"] == "invalid_min_notional"
        dust = pf.rebalance_quantity("A", 0.01, 10.0, "dust-q", target_weight=0.001, min_notional=1.0)
        assert dust["skipped"] == "below_min_notional"

    def test_presized_rebalance_clamps_buy_and_duplicate_key_is_idempotent(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=100.0,
            fee=0.10,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        fill = pf.rebalance_quantity("A", 2.0, 100.0, "buy", target_weight=1.0, min_notional=0.0)
        assert fill["kind"] == "fill"
        assert fill["side"] == "BUY"
        assert pf.cash == pytest.approx(0.0, abs=1e-8)
        assert pf.positions["A"] == pytest.approx(100.0 / 110.0)
        duplicate = pf.rebalance_quantity("A", 2.0, 100.0, "buy", target_weight=1.0, min_notional=0.0)
        assert duplicate == {"skipped": "duplicate_order", "idem_key": "buy"}
        blocked = pf.rebalance_quantity("A", 2.0, 100.0, "second", target_weight=1.0, min_notional=0.0)
        assert blocked["skipped"] == "insufficient_cash"

    def test_presized_sell_can_close_position_with_naive_cycle_timestamp(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=1000.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        pf.rebalance_quantity("A", 5.0, 100.0, "buy", target_weight=0.5)
        sell = pf.rebalance_quantity(
            "A", 0.0, 100.0, "sell", target_weight=0.0,
            ts=datetime(2026, 9, 22, 17, 0),
        )
        assert sell["side"] == "SELL"
        assert sell["date"] == "2026-09-22"
        assert "A" not in pf.positions
        assert pf.cash == pytest.approx(1000.0)


class TestCycleResilience:
    def test_multi_buy_rebalance_is_symbol_order_invariant_with_fees(self, tmp_path):
        data = {"A": _candles([100.0] * 5), "B": _candles([100.0] * 5)}

        def execute(order, suffix):
            pf = PaperPortfolio(
                start_cash=1000.0,
                fee=0.001,
                state_file=tmp_path / f"state-{suffix}.json",
                ledger_file=tmp_path / f"ledger-{suffix}.jsonl",
            )
            result = run_cycle(
                order,
                lambda _sym, _candles_: 0.5,
                lambda sym: data[sym],
                pf,
                reports_dir=tmp_path / f"reports-{suffix}",
                now=NOW,
            )
            return pf, result

        left, left_result = execute(["A", "B"], "ab")
        right, right_result = execute(["B", "A"], "ba")

        assert left.cash == pytest.approx(right.cash, abs=1e-8)
        assert left.positions == pytest.approx(right.positions, abs=1e-10)
        assert left.positions["A"] == pytest.approx(left.positions["B"], abs=1e-10)
        assert left.cash >= -1e-8
        assert not any(a["level"] == "rebalance_weight_drift" for a in left_result["alerts"])
        assert not any(a["level"] == "rebalance_weight_drift" for a in right_result["alerts"])

    def test_post_trade_invariant_surfaces_dust_weight_drift(self, tmp_path):
        pf = PaperPortfolio(
            start_cash=100.0,
            fee=0.0,
            state_file=tmp_path / "state.json",
            ledger_file=tmp_path / "ledger.jsonl",
        )
        result = run_cycle(
            ["A"],
            lambda _sym, _candles_: 0.006,
            lambda _sym: _candles([100.0] * 5),
            pf,
            reports_dir=tmp_path / "reports",
            now=NOW,
        )
        assert any(a["level"] == "rebalance_weight_drift" for a in result["alerts"])
        assert result["fills"][0]["skipped"] == "below_min_notional"

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
        assert any(a["level"] == "target_weights_scaled" for a in result["alerts"])

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
            ["A", "B"],
            lambda _sym, _candles_: 0.0,
            lambda sym: [] if sym == "A" else _candles([100.0] * 5),
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
