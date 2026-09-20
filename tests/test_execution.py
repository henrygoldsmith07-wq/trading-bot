"""Independent accounting invariants for shared backtest/forward execution.

Pin the close-mode, turnover and cash conventions while checking split-day
returns against portfolio balances rather than the implementation formula.
"""
import pytest

from bot.engine import run_strategy
from bot.execution import calculate_transition, open_of


class TestCalculateTransition:
    @pytest.mark.parametrize("opening,closing", [(105.0, 110.0), (90.0, 80.0), (120.0, 100.0)])
    def test_flat_position_earns_full_move(self, opening, closing):
        t = calculate_transition(1.0, 1.0, 100.0, opening, closing)
        assert t["return"] == pytest.approx(closing / 100.0 - 1.0)

    def test_next_open_split_attributes_gap_to_old_position(self):
        t = calculate_transition(0.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.0)
        assert t["intraday"] == pytest.approx(0.10)
        assert t["return"] == pytest.approx(0.10)

    def test_gap_earned_by_old_position_when_held(self):
        t = calculate_transition(1.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.02)
        assert t["intraday"] == pytest.approx(112.2 / 102.0 - 1.0)
        assert t["return"] == pytest.approx(112.2 / 100.0 - 1.0)

    @pytest.mark.parametrize("previous,target", [(0.25, 0.75), (0.75, 0.25), (1.0, 0.0), (0.0, 1.0)])
    def test_rebalance_matches_independent_portfolio_balance(self, previous, target):
        # Start with 1,000 units of equity, mark old shares to the open,
        # then allocate the target fraction of that equity to new shares.
        initial = 1000.0
        old_shares = initial * previous / 200.0
        opening_equity = initial * (1.0 - previous) + old_shares * 204.0
        new_shares = opening_equity * target / 204.0
        closing_equity = opening_equity * (1.0 - target) + new_shares * 210.0
        t = calculate_transition(previous, target, 200.0, 204.0, 210.0)
        assert t["overnight"] == pytest.approx(previous * (204.0 / 200.0 - 1.0))
        assert t["intraday"] == pytest.approx(target * (210.0 / 204.0 - 1.0))
        assert t["return"] == pytest.approx(closing_equity / initial - 1.0)

    def test_close_mode_degenerate_exec_at_previous_close(self):
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

    def test_split_return_preserves_additive_cost_and_cash_conventions(self):
        gross = calculate_transition(0.25, 0.75, 200.0, 204.0, 210.0)
        net = calculate_transition(0.25, 0.75, 200.0, 204.0, 210.0, costs=0.01, cash_rate_period=0.001)
        assert net["cost"] == pytest.approx(0.005)
        assert net["cash"] == pytest.approx(0.00075)
        assert net["return"] == pytest.approx(gross["return"] - 0.005 + 0.00075)
        assert set(net) == {"return", "overnight", "intraday", "turnover", "cost", "cash"}

    def test_cash_accrual_previous_vs_target_basis(self):
        rate = 0.05 / 365
        prev_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="previous")
        tgt_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="target")
        assert prev_basis["cash"] == pytest.approx(rate)
        assert tgt_basis["cash"] == pytest.approx(0.0)
        assert calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate)["cash"] == 0.0

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("field", ["previous_close", "execution_price", "closing_price"])
    def test_invalid_prices_raise(self, bad, field):
        prices = {"previous_close": 100.0, "execution_price": 100.0, "closing_price": 100.0}
        prices[field] = bad
        with pytest.raises(ValueError, match="positive"):
            calculate_transition(1.0, 1.0, **prices)

    def test_bad_cash_basis_raises(self):
        with pytest.raises(ValueError, match="cash_basis"):
            calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_basis="both")

    def test_engine_equity_matches_buy_and_hold_across_gaps(self):
        # First enter at 100, then remain fully invested through positive
        # and negative gaps. No fees or cash yield obscure the invariant.
        candles = [
            {"open_time": 0, "open": 100.0, "close": 100.0},
            {"open_time": 86400000, "open": 100.0, "close": 100.0},
            {"open_time": 172800000, "open": 105.0, "close": 110.0},
            {"open_time": 259200000, "open": 99.0, "close": 90.0},
        ]
        result = run_strategy(candles, lambda _candles, _i: 1.0, fee=0.0, execution="next_open")
        assert result["equity"] == pytest.approx([1.0, 1.0, 1.1, 0.9])


class TestOpenOf:
    def test_valid_open_passes_through(self):
        assert open_of({"open": 42.0}, fallback=1.0) == 42.0

    def test_missing_open_falls_back(self):
        assert open_of({}, fallback=7.0) == 7.0

    @pytest.mark.parametrize("bad", [None, 0.0, -5.0, float("nan"), float("inf"), float("-inf")])
    def test_invalid_open_falls_back(self, bad):
        assert open_of({"open": bad}, fallback=7.0) == 7.0
