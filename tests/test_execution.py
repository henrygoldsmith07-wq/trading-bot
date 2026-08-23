"""calculate_transition: one implementation for backtest AND forward P&L.

These tests pin the semantics the engine has always had (next_open splits
overnight/intraday; close mode credits the whole move to the new position)
so the shared function cannot drift from either caller.
"""
import pytest

from bot.execution import calculate_transition, open_of


class TestCalculateTransition:
    def test_flat_position_earns_full_move(self):
        t = calculate_transition(1.0, 1.0, 100.0, 105.0, 110.0)
        compounded = (1.0 + t["overnight"]) * (1.0 + t["intraday"]) - 1.0
        total_close_to_close = 110.0 / 100.0 - 1.0
        # the split legs COMPOUND to the full close/close move
        assert compounded == pytest.approx(total_close_to_close)
        # the additive accounting is the period return the engine logs
        assert t["return"] == pytest.approx(0.05 + (110.0 / 105.0 - 1.0))

    def test_next_open_split_attributes_gap_to_old_position(self):
        # overnight +2% belongs to the OLD (zero) position; intraday +10% to new
        t = calculate_transition(0.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.0)
        assert t["intraday"] == pytest.approx(0.10)
        assert t["return"] == pytest.approx(0.10)

    def test_gap_earned_by_old_position_when_held(self):
        # holding 1.0, no trade: overnight +2% then intraday to 112.2
        t = calculate_transition(1.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.02)
        assert t["intraday"] == pytest.approx(112.2 / 102.0 - 1.0)
        # compounding the legs reproduces the total close/close move exactly
        assert (1 + t["overnight"]) * (1 + t["intraday"]) - 1 == pytest.approx(112.2 / 100.0 - 1.0)
        assert t["return"] == pytest.approx(t["overnight"] + t["intraday"])

    def test_switch_attributions_match_engine_formula_exactly(self):
        # half-position flip: 0.25 -> 0.75
        t = calculate_transition(0.25, 0.75, 200.0, 204.0, 210.0)
        expected_overnight = 0.25 * (204.0 / 200.0 - 1.0)
        expected_intraday = 0.75 * (210.0 / 204.0 - 1.0)
        assert t["overnight"] == pytest.approx(expected_overnight)
        assert t["intraday"] == pytest.approx(expected_intraday)
        assert t["return"] == pytest.approx(expected_overnight + expected_intraday)

    def test_close_mode_degenerate_exec_at_previous_close(self):
        # execution AT the previous close: zero-length overnight, full move to new
        t = calculate_transition(0.0, 1.0, 100.0, 100.0, 110.0)
        assert t["overnight"] == pytest.approx(0.0)
        assert t["intraday"] == pytest.approx(0.10)
        assert t["return"] == pytest.approx(0.10)

    def test_costs_charged_on_turnover_only(self):
        no_trade = calculate_transition(0.5, 0.5, 100.0, 90.0, 95.0, costs=0.01)
        assert no_trade["turnover"] == pytest.approx(0.0)
        assert no_trade["cost"] == pytest.approx(0.0)

        trade = calculate_transition(0.0, 1.0, 100.0, 100.0, 110.0, costs=0.01)
        assert trade["turnover"] == pytest.approx(1.0)
        assert trade["cost"] == pytest.approx(0.01)
        assert trade["return"] == pytest.approx(0.10 - 0.01)

    def test_cash_accrual_previous_vs_target_basis(self):
        rate = 0.05 / 365
        prev_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="previous")
        tgt_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="target")
        assert prev_basis["cash"] == pytest.approx(rate)       # was fully in cash overnight
        assert tgt_basis["cash"] == pytest.approx(0.0)         # invested after the trade
        assert calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate)["cash"] == 0.0

    def test_invalid_prices_raise(self):
        for bad in (0.0, -5.0):
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, bad, 100.0, 100.0)
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, 100.0, bad, 100.0)
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, 100.0, 100.0, bad)

    def test_bad_cash_basis_raises(self):
        with pytest.raises(ValueError, match="cash_basis"):
            calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_basis="both")


class TestOpenOf:
    def test_valid_open_passes_through(self):
        assert open_of({"open": 42.0}, fallback=1.0) == 42.0

    def test_missing_or_invalid_open_falls_back(self):
        assert open_of({}, fallback=7.0) == 7.0
        assert open_of({"open": None}, fallback=7.0) == 7.0
        assert open_of({"open": 0.0}, fallback=7.0) == 7.0
