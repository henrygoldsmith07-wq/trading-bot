"""calculate_transition: one implementation for backtest AND forward P&L.

These tests pin the EXACT WEALTH semantics the engine has always intended:
next_open splits overnight/intraday; close mode credits the whole move to the
new position; and in both cases the session return is the exact evolution of
portfolio wealth, not a sum of approximations.

Expected values are derived from first principles (a hand-built wealth ledger)
rather than by re-running the implementation under test, so these tests can
actually FAIL if the accounting is wrong.
"""
import pytest

from bot.execution import calculate_transition, open_of


def expected_wealth_return(w_prev, w_target, prev_close, exec_price, close, costs=0.0, cash=0.0, basis="previous"):
    """Independent reference implementation, written as a wealth ledger.

    Deliberately NOT the production code path: it re-derives the answer from
    the definition of a self-financing portfolio, so the test compares two
    different derivations of the same quantity.
    """
    asset = w_prev * exec_price / prev_close
    idle = (1.0 - w_prev) * (1.0 + (cash if basis == "previous" else 0.0))
    wealth = asset + idle
    drifted = asset / wealth
    turnover = abs(w_target - drifted)
    # The fee is charged on the traded NOTIONAL, i.e. a fraction of opening
    # wealth — not on the bare turnover fraction. This is why a fee cannot be
    # simply subtracted from the gross return.
    wealth -= costs * turnover * wealth
    asset = w_target * wealth * close / exec_price
    idle = (1.0 - w_target) * wealth * (1.0 + (cash if basis == "target" else 0.0))
    return (asset + idle) - 1.0


class TestExactWealthInvariants:
    """The invariant the whole accounting exists to satisfy."""

    def test_unchanged_full_holding_across_gap_equals_close_to_close(self):
        # THE motivating case: a 100% holding through a +5% gap and a +4.76%
        # intraday move must earn the full close-to-close return, exactly.
        t = calculate_transition(1.0, 1.0, 100.0, 105.0, 110.0)
        assert t["return"] == pytest.approx(110.0 / 100.0 - 1.0, abs=1e-15)
        # and NOT the additive approximation 0.05 + 0.047619
        assert t["return"] != pytest.approx(0.05 + (110.0 / 105.0 - 1.0), abs=1e-6)

    def test_unchanged_zero_holding_earns_no_asset_return(self):
        assert calculate_transition(0.0, 0.0, 100.0, 105.0, 110.0)["return"] == pytest.approx(0.0, abs=1e-15)
        # even when the market falls
        assert calculate_transition(0.0, 0.0, 100.0, 95.0, 90.0)["return"] == pytest.approx(0.0, abs=1e-15)

    def test_entering_at_next_open_receives_no_previous_gap(self):
        # Flat going in: the +2% overnight gap happened while holding nothing.
        t = calculate_transition(0.0, 1.0, 100.0, 102.0, 112.2)
        assert t["return"] == pytest.approx(112.2 / 102.0 - 1.0, abs=1e-15)
        assert t["overnight"] == pytest.approx(0.0, abs=1e-15)

    def test_exiting_at_next_open_still_receives_the_overnight_move(self):
        # Fully invested going in: the +2% gap WAS earned, then we exit at the
        # open and earn nothing intraday.
        t = calculate_transition(1.0, 0.0, 100.0, 102.0, 112.2)
        assert t["return"] == pytest.approx(0.02, abs=1e-15)

    @pytest.mark.parametrize("gap", [1.5, 1.05, 1.0, 0.8, 0.5])
    @pytest.mark.parametrize("intraday", [1.4, 1.1, 1.0, 0.9, 0.6])
    def test_full_holding_reproduces_close_to_close_across_any_path(self, gap, intraday):
        t = calculate_transition(1.0, 1.0, 100.0, 100.0 * gap, 100.0 * gap * intraday)
        assert t["return"] == pytest.approx(gap * intraday - 1.0, abs=1e-12)

    def test_decomposition_is_a_wealth_identity(self):
        # The legs are wealth deltas, so they ADD to the session return.
        t = calculate_transition(0.4, 0.8, 100.0, 103.0, 107.0, costs=0.002, cash_rate_period=0.0001)
        assert t["overnight"] + t["intraday"] + t["cash"] - t["cost"] == pytest.approx(t["return"], abs=1e-15)

    def test_wealth_walk_is_internally_consistent(self):
        t = calculate_transition(0.3, 0.9, 100.0, 105.0, 101.0, costs=0.01)
        assert t["wealth_start"] == 1.0
        # costs strictly reduce wealth between open and post-trade
        assert t["wealth_after_cost"] < t["wealth_open"]
        # the reported return is exactly closing/starting wealth
        assert t["return"] == pytest.approx(t["wealth_close"] / t["wealth_start"] - 1.0, abs=1e-15)


class TestSemanticsPinned:
    def test_next_open_split_attributes_gap_to_old_position(self):
        t = calculate_transition(0.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.0)
        assert t["return"] == pytest.approx(0.10)

    def test_gap_earned_by_old_position_when_held(self):
        t = calculate_transition(1.0, 1.0, 100.0, 102.0, 112.2)
        assert t["overnight"] == pytest.approx(0.02)
        # overnight + intraday (both wealth deltas) reconstruct close/close
        assert t["overnight"] + t["intraday"] == pytest.approx(112.2 / 100.0 - 1.0, abs=1e-15)

    def test_switch_matches_independent_wealth_ledger(self):
        # half-position flip 0.25 -> 0.75 across a +2% gap
        t = calculate_transition(0.25, 0.75, 200.0, 204.0, 210.0)
        assert t["return"] == pytest.approx(
            expected_wealth_return(0.25, 0.75, 200.0, 204.0, 210.0), abs=1e-15
        )
        # the new position, not the old, earns the intraday leg
        assert t["overnight"] == pytest.approx(0.25 * (204.0 / 200.0 - 1.0), abs=1e-15)

    def test_close_mode_degenerate_exec_at_previous_close(self):
        t = calculate_transition(0.0, 1.0, 100.0, 100.0, 110.0)
        assert t["overnight"] == pytest.approx(0.0)
        assert t["return"] == pytest.approx(0.10)

    def test_close_mode_full_holding_equals_close_to_close(self):
        t = calculate_transition(1.0, 1.0, 100.0, 100.0, 110.0)
        assert t["return"] == pytest.approx(0.10, abs=1e-15)




    @pytest.mark.parametrize("position", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_positions_raise(self, position):
        with pytest.raises(ValueError, match="position"):
            calculate_transition(position, 0.5, 100.0, 100.0, 101.0)
        with pytest.raises(ValueError, match="position"):
            calculate_transition(0.5, position, 100.0, 100.0, 101.0)

    @pytest.mark.parametrize("costs", [-0.01, float("nan"), float("inf")])
    def test_invalid_costs_raise(self, costs):
        with pytest.raises(ValueError, match="costs"):
            calculate_transition(0.5, 0.5, 100.0, 100.0, 101.0, costs=costs)

    @pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
    def test_invalid_cash_rate_raises(self, rate):
        with pytest.raises(ValueError, match="cash_rate_period"):
            calculate_transition(0.5, 0.5, 100.0, 100.0, 101.0, cash_rate_period=rate)


class TestTurnoverOnDriftedAllocation:
    def test_no_gap_and_unchanged_target_trades_nothing(self):
        t = calculate_transition(0.5, 0.5, 100.0, 100.0, 105.0, costs=0.01)
        assert t["turnover"] == pytest.approx(0.0, abs=1e-15)
        assert t["cost"] == pytest.approx(0.0, abs=1e-15)

    def test_unchanged_target_after_gap_requires_a_real_rebalance(self):
        # A 50% weight becomes 47.4% after a -10% gap. Holding the 50%
        # target therefore requires BUYING - a real trade with a real fee.
        # Charging zero here would understate costs on every gapping asset.
        t = calculate_transition(0.5, 0.5, 100.0, 90.0, 95.0, costs=0.01)
        drifted = (0.5 * 0.9) / (0.5 * 0.9 + 0.5)
        assert t["drifted_weight"] == pytest.approx(drifted, abs=1e-15)
        assert t["turnover"] == pytest.approx(0.5 - drifted, abs=1e-15)
        assert t["cost"] > 0.0

    def test_turnover_matches_target_minus_drifted_weight(self):
        t = calculate_transition(0.2, 0.8, 100.0, 120.0, 130.0)
        assert t["turnover"] == pytest.approx(0.8 - t["drifted_weight"], abs=1e-15)

    def test_costs_charged_on_turnover_only(self):
        no_trade = calculate_transition(0.5, 0.5, 100.0, 100.0, 95.0, costs=0.01)
        assert no_trade["turnover"] == pytest.approx(0.0)
        assert no_trade["cost"] == pytest.approx(0.0)

        trade = calculate_transition(0.0, 1.0, 100.0, 100.0, 110.0, costs=0.01)
        assert trade["turnover"] == pytest.approx(1.0)
        assert trade["cost"] == pytest.approx(0.01)


class TestFeesAndCash:
    def test_fees_reduce_wealth_not_the_return_arithmetic(self):
        # With NO gap and NO rebalance there is no trade, so no fee at all.
        free = calculate_transition(1.0, 1.0, 100.0, 100.0, 110.0, costs=0.01)
        assert free["return"] == pytest.approx(0.10, abs=1e-15)
        # A fee is charged exactly once, on the traded notional. It comes out
        # of the account BEFORE the position earns its intraday move, so the
        # surviving wealth compounds: (1 - 0.01) * 1.10 - 1 = +8.900%, NOT
        # 10% - 1% = 9%. That difference is the whole point.
        paid = calculate_transition(0.0, 1.0, 100.0, 100.0, 110.0, costs=0.01)
        assert paid["wealth_after_cost"] == pytest.approx(1.0 - 0.01, abs=1e-15)
        assert paid["return"] == pytest.approx(0.99 * 1.10 - 1.0, abs=1e-15)
        assert paid["return"] == pytest.approx(0.089, abs=1e-15)
        assert paid["return"] != pytest.approx(0.10 - 0.01, abs=1e-4)

    def test_fee_matched_against_independent_ledger(self):
        args = (0.6, 0.2, 100.0, 104.0, 99.0)
        t = calculate_transition(*args, costs=0.0035, cash_rate_period=0.0002)
        assert t["return"] == pytest.approx(
            expected_wealth_return(*args, costs=0.0035, cash=0.0002), abs=1e-15
        )

    def test_cash_accrual_previous_vs_target_basis(self):
        rate = 0.05 / 365
        prev_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="previous")
        tgt_basis = calculate_transition(0.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate, cash_basis="target")
        assert prev_basis["cash"] == pytest.approx(rate)       # was fully in cash overnight
        assert tgt_basis["cash"] == pytest.approx(0.0)         # invested after the trade
        assert calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_rate_period=rate)["cash"] == 0.0

    def test_all_cash_earns_the_cash_rate(self):
        rate = 0.05 / 365
        t = calculate_transition(0.0, 0.0, 100.0, 100.0, 100.0, cash_rate_period=rate, cash_basis="previous")
        assert t["return"] == pytest.approx(rate, abs=1e-15)


class TestValidation:
    def test_invalid_prices_raise(self):
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, bad, 100.0, 100.0)
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, 100.0, bad, 100.0)
            with pytest.raises(ValueError, match="positive"):
                calculate_transition(1.0, 1.0, 100.0, 100.0, bad)

    def test_bad_cash_basis_raises(self):
        with pytest.raises(ValueError, match="cash_basis"):
            calculate_transition(1.0, 1.0, 100.0, 100.0, 101.0, cash_basis="both")

    @pytest.mark.parametrize("position", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_positions_raise(self, position):
        with pytest.raises(ValueError, match="position"):
            calculate_transition(position, 0.5, 100.0, 100.0, 101.0)
        with pytest.raises(ValueError, match="position"):
            calculate_transition(0.5, position, 100.0, 100.0, 101.0)

    @pytest.mark.parametrize("costs", [-0.01, float("nan"), float("inf")])
    def test_invalid_costs_raise(self, costs):
        with pytest.raises(ValueError, match="costs"):
            calculate_transition(0.5, 0.5, 100.0, 100.0, 101.0, costs=costs)

    @pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
    def test_invalid_cash_rate_raises(self, rate):
        with pytest.raises(ValueError, match="cash_rate_period"):
            calculate_transition(0.5, 0.5, 100.0, 100.0, 101.0, cash_rate_period=rate)

    def test_bankrupting_cost_raises_instead_of_reporting_negative_wealth(self):
        with pytest.raises(ValueError, match="exhausted"):
            calculate_transition(0.0, 1.0, 100.0, 100.0, 100.0, costs=1.5)


class TestOpenOf:
    def test_valid_open_passes_through(self):
        assert open_of({"open": 42.0}, fallback=1.0) == 42.0

    def test_missing_or_invalid_open_falls_back(self):
        assert open_of({}, fallback=7.0) == 7.0
        assert open_of({"open": None}, fallback=7.0) == 7.0
        assert open_of({"open": 0.0}, fallback=7.0) == 7.0

        assert open_of({"open": float("nan")}, fallback=7.0) == 7.0
        assert open_of({"open": float("inf")}, fallback=7.0) == 7.0

    @pytest.mark.parametrize("fallback", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_fallback_raises(self, fallback):
        with pytest.raises(ValueError, match="fallback"):
            open_of({}, fallback=fallback)
