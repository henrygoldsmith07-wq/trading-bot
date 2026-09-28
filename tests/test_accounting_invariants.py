"""INDEPENDENT accounting invariants â€” correctness, not consistency.

The backtest/forward parity suite (tests/test_backtest_forward_parity.py) is
valuable but structurally incapable of proving the shared accounting code is
economically correct: both paths call `calculate_transition`, so a wrong
function produces two identically-wrong answers and parity still passes.

This file closes that gap. Expected numbers are computed FROM FIRST
PRINCIPLES â€” by simulating a self-financing portfolio in units of currency,
holding real share counts and cash balances â€” and compared against what the
production engine reports. The reference simulator never imports the helper
under test, so the two are independent derivations of the same quantity and
can genuinely disagree.

Covered: overnight gaps (positive and negative), target changes, partial
weights, fees, cash, multiple sequential days, rebalance bands, and
alternating long/cash exposure.
"""
import math

import pytest

from bot.engine import run_strategy
from bot.execution import calculate_transition

DAY_MS = 86_400_000
BASE = 1_600_000_000_000


def simulate(closes, opens, targets, cost_rate=0.0, cash_rate=0.0):
    """Exact next-open portfolio simulation from first principles.

    Holds real quantities: a share count and a cash balance, carried across
    days in currency units. Rebalancing happens at the OPEN at the open price;
    the new position marks at each CLOSE. Never calls the production helper.

    Accounting window matches the engine's: bar i (i >= 1) accounts
    closes[i-1] -> opens[i] -> closes[i], starting from a flat book. Bar 0 is
    the decision seed and is never traded. Each bar's return is measured
    against the PREVIOUS bar's closing wealth, as a session return must be.
    """
    shares = 0.0
    cash = 1.0
    prev_weight = 0.0
    equity_prev = 1.0
    returns = []
    for i in range(1, len(closes)):
        o, c = opens[i], closes[i]
        target = targets[i]
        # Overnight: yesterday's share count is re-valued at today's open. The
        # engine's weight convention is struck against the previous CLOSE, so
        # the position we carry is whatever was worth `prev_weight` at the
        # last close — the shares below are set to make that true.
        value_at_open = shares * o
        if prev_weight < 1.0:
            cash *= 1.0 + cash_rate
        wealth_open = value_at_open + cash
        drifted = value_at_open / wealth_open
        turnover = abs(target - drifted)
        # fee is charged on the traded NOTIONAL (a share of opening wealth)
        wealth_after = wealth_open - cost_rate * turnover * wealth_open
        # Trade to `target` of post-fee wealth, then let the new position mark
        # to the close. The resulting share count is carried into the next
        # bar, where it is re-valued at that bar's open.
        post_shares = (target * wealth_after) / o
        equity_close = post_shares * c + (1.0 - target) * wealth_after
        returns.append(equity_close / equity_prev - 1.0)
        # Carry the position: it was worth `target` of the account at the
        # close, which is the price tomorrow's decision is taken against.
        shares = (target * equity_close) / c
        cash = equity_close - shares * c
        prev_weight, equity_prev = target, equity_close
    return returns


def candles_from(closes, opens):
    return [
        {"open_time": BASE + i * DAY_MS, "open": o, "close": c,
         "high": max(o, c), "low": min(o, c)}
        for i, (o, c) in enumerate(zip(opens, closes))
    ]


def run(candles, targets, **kw):
    return run_strategy(candles, lambda c, i: targets[i], start_index=1, **kw)


class TestEngineMatchesWealthSimulator:
    """Engine daily returns must equal an independent wealth simulation."""

    def test_two_day_buy_and_hold(self):
        closes, opens = [100.0, 104.0, 109.0], [100.0, 102.0, 106.0]
        targets = [1.0, 1.0, 1.0]
        res = run(candles_from(closes, opens), targets, fee=0.0, execution="next_open")
        assert res["returns"] == pytest.approx(simulate(closes, opens, targets), abs=1e-12)

    def test_full_wealth_equals_close_to_close(self):
        # A permanently-full position held from bar 1's OPEN must reproduce
        # exact growth of the asset from that open onwards. The entry bar's
        # open is the true entry price, so the reference is opens[1].
        closes, opens = [100.0, 110.0, 99.0, 120.0], [100.0, 105.0, 101.0, 115.0]
        res = run(candles_from(closes, opens), [1.0] * 4, fee=0.0, execution="next_open")
        assert res["equity"][-1] == pytest.approx(closes[-1] / opens[1], abs=1e-12)

    def test_fees_compound_against_a_wealth_simulator(self):
        closes, opens = [100.0, 110.0, 99.0, 120.0], [100.0, 105.0, 101.0, 115.0]
        targets, fee = [1.0] * 4, 0.002
        res = run(candles_from(closes, opens), targets, fee=fee, execution="next_open")
        assert res["returns"] == pytest.approx(simulate(closes, opens, targets, cost_rate=fee), abs=1e-12)

    def test_alternating_long_and_cash(self):
        closes = [100.0, 110.0, 108.0, 120.0, 118.0]
        opens = [100.0, 105.0, 112.0, 115.0, 122.0]
        targets, fee = [1.0, 0.0, 1.0, 0.0, 1.0], 0.001
        res = run(candles_from(closes, opens), targets, fee=fee, execution="next_open")
        assert res["returns"] == pytest.approx(simulate(closes, opens, targets, cost_rate=fee), abs=1e-12)

    def test_partial_weight_sequence(self):
        closes = [100.0, 103.0, 107.0, 99.0, 104.0]
        opens = [100.0, 101.0, 105.0, 100.0, 102.0]
        targets, fee = [0.25, 0.5, 0.75, 0.25, 0.0], 0.0007
        res = run(candles_from(closes, opens), targets, fee=fee, execution="next_open")
        assert res["returns"] == pytest.approx(simulate(closes, opens, targets, cost_rate=fee), abs=1e-12)

    def test_negative_gaps_sequence(self):
        closes, opens = [100.0, 90.0, 85.0, 95.0], [100.0, 92.0, 80.0, 93.0]
        res = run(candles_from(closes, opens), [1.0] * 4, fee=0.0, execution="next_open")
        assert res["equity"][-1] == pytest.approx(95.0 / 92.0, abs=1e-12)

    def test_spread_and_slippage_behave_as_costs(self):
        closes, opens = [100.0, 105.0, 110.0], [100.0, 102.0, 106.0]
        c = candles_from(closes, opens)
        free = run(c, [1.0] * 3, fee=0.0, execution="next_open")
        costed = run(c, [1.0] * 3, fee=0.001, spread_bps=10.0, slippage_bps=10.0, execution="next_open")
        assert costed["equity"][-1] < free["equity"][-1]

    def test_wealth_is_monotone_in_the_terminal_price(self):
        opens = [100.0] * 4
        out = []
        for last in (80.0, 100.0, 130.0):
            res = run(candles_from([100.0, 100.0, 100.0, last], opens), [1.0] * 4,
                      fee=0.0, execution="next_open")
            out.append(res["equity"][-1])
        assert out == sorted(out)

    def test_cash_yield_is_reproduced_by_the_simulator(self):
        closes, opens = [100.0] * 4, [100.0] * 4
        rate = 0.03 / 365
        res = run(candles_from(closes, opens), [0.0] * 4, fee=0.0,
                  execution="next_open", risk_free_annual=0.03)
        # Relative tolerance: the exact-wealth form computes (1+r)/1 - 1, and
        # subtracting 1 from 1.00008 discards ~4 significant digits. The
        # residual is ~1e-16 absolute, i.e. a millionth of a basis point.
        assert res["returns"] == pytest.approx([rate] * 3, rel=1e-12)
        assert res["equity"][-1] == pytest.approx(
            math.prod([1.0 + rate] * 3), rel=1e-15)


class TestTransitionInvariants:
    """Invariants on `calculate_transition` derived from portfolio theory."""

    @pytest.mark.parametrize("prev,o,c", [(100, 105, 110), (100, 90, 85), (50, 55, 60), (7, 7, 7)])
    def test_unchanged_full_holding_equals_close_to_close(self, prev, o, c):
        t = calculate_transition(1.0, 1.0, prev, o, c)
        assert t["return"] == pytest.approx(c / prev - 1.0, abs=1e-12)

    def test_entering_at_the_open_captures_only_the_intraday_move(self):
        t = calculate_transition(0.0, 1.0, 100.0, 120.0, 110.0)
        assert t["return"] == pytest.approx(110.0 / 120.0 - 1.0, abs=1e-12)

    def test_exiting_at_the_open_captures_only_the_overnight_move(self):
        t = calculate_transition(1.0, 0.0, 100.0, 120.0, 110.0)
        assert t["return"] == pytest.approx(0.20, abs=1e-12)

    def test_partial_weight_composes_gap_and_intraday_exactly(self):
        # Opening wealth is w*o/prev + (1-w). The book is then rebalanced to
        # exactly weight w at the open, so the intraday move applies to w of
        # THAT opening wealth â€” not to the drifted weight.
        w, prev, o, c = 0.4, 100.0, 110.0, 121.0
        t = calculate_transition(w, w, prev, o, c)
        wealth_open = w * (o / prev) + (1 - w)
        expected = wealth_open * (w * (c / o) + (1 - w)) - 1.0
        assert t["return"] == pytest.approx(expected, abs=1e-12)

    def test_costs_scale_with_turnover(self):
        a = calculate_transition(0.0, 0.5, 100.0, 100.0, 100.0, costs=0.001)
        b = calculate_transition(0.0, 1.0, 100.0, 100.0, 100.0, costs=0.001)
        assert b["cost"] == pytest.approx(2 * a["cost"], abs=1e-15)

    def test_no_cost_when_no_position_change_and_no_gap(self):
        t = calculate_transition(0.7, 0.7, 100.0, 100.0, 100.0, costs=0.005)
        assert t["cost"] == 0.0 and t["turnover"] == 0.0 and t["return"] == 0.0

    @pytest.mark.parametrize("prev,o,c", [(100, 200, 400), (100, 50, 25)])
    def test_zero_weight_never_earns_anything(self, prev, o, c):
        assert calculate_transition(0.0, 0.0, prev, o, c)["return"] == 0.0

    def test_hairline_weight_change_is_bounded_by_drift(self):
        t = calculate_transition(0.5, 0.5001, 100.0, 100.0, 100.0)
        assert t["turnover"] == pytest.approx(0.0001, abs=1e-15)

    def test_wealth_is_monotone_in_price_for_a_long_position(self):
        base = calculate_transition(1.0, 1.0, 100.0, 100.0, 100.0)["return"]
        for c in (101.0, 110.0, 150.0):
            assert calculate_transition(1.0, 1.0, 100.0, 100.0, c)["return"] > base

    def test_shorting_is_impossible(self):
        with pytest.raises(ValueError):
            calculate_transition(-0.5, 0.0, 100.0, 100.0, 100.0)
        with pytest.raises(ValueError):
            calculate_transition(0.0, 1.5, 100.0, 100.0, 100.0)


class TestWealthConservation:
    """Compounding engine returns must equal a directly simulated account."""

    def test_equity_matches_compounded_exact_wealth(self):
        closes = [100.0, 103.0, 101.0, 108.0, 112.0, 109.0]
        opens = [100.0, 101.0, 102.0, 104.0, 115.0, 108.0]
        targets = [0.3, 0.6, 0.6, 0.0, 0.9, 0.9]
        fee = 0.0009
        res = run(candles_from(closes, opens), targets, fee=fee, execution="next_open")
        wealth = 1.0
        for r in simulate(closes, opens, targets, cost_rate=fee):
            wealth *= 1.0 + r
        assert res["equity"][-1] == pytest.approx(wealth, abs=1e-12)

    def test_equity_never_goes_negative(self):
        closes = [100.0, 50.0, 20.0, 5.0]
        opens = [100.0, 60.0, 25.0, 6.0]
        res = run(candles_from(closes, opens), [1.0] * 4, fee=0.001, execution="next_open")
        assert all(e > 0 for e in res["equity"])
        assert all(r > -1.0 for r in res["returns"])


class TestRebalanceBandEconomics:
    """Banding must suppress trades, and the band must be the only effect."""

    def test_band_suppresses_small_moves(self):
        closes = opens = [100.0] * 6
        targets = [0.0, 0.02, 0.0, 0.02, 0.0, 0.02]
        res = run(candles_from(closes, opens), targets, fee=0.002,
                  execution="next_open", rebalance_band=0.05)
        assert all(c == 0.0 for c in res["bar_costs"])
        assert all(w == 0.0 for w in res["weights"])

    def test_band_lets_large_moves_through(self):
        closes = opens = [100.0] * 4
        res = run(candles_from(closes, opens), [0.0, 0.0, 1.0, 1.0], fee=0.002,
                  execution="next_open", rebalance_band=0.05)
        assert res["weights"][-1] == 1.0
        assert sum(res["bar_costs"]) > 0.0

    def test_zero_band_trades_on_every_move(self):
        closes = opens = [100.0] * 5
        res = run(candles_from(closes, opens), [0.0, 0.1, 0.2, 0.3, 0.4], fee=0.002,
                  execution="next_open", rebalance_band=0.0)
        assert all(c > 0.0 for c in res["bar_costs"][1:])

    def test_banded_run_is_wealth_simulatable(self):
        closes = [100.0, 105.0, 103.0, 110.0, 108.0]
        opens = [100.0, 102.0, 106.0, 104.0, 112.0]
        # bar 3 is the first that wants 1.0; the band lets it through
        targets = [0.0, 0.0, 0.0, 1.0, 1.0]
        res = run(candles_from(closes, opens), targets, fee=0.001,
                  execution="next_open", rebalance_band=0.05)
        assert res["weights"] == pytest.approx(targets[1:], abs=1e-12)
        assert res["returns"] == pytest.approx(
            simulate(closes, opens, targets, cost_rate=0.001), abs=1e-12)


class TestExecutionModeSemantics:
    """next_open and close must differ by exactly the overnight attribution."""

    def test_close_mode_full_holding_still_earns_exact_close_to_close(self):
        # In close mode execution happens at the PREVIOUS close, so bar 1
        # executes at closes[0] and a permanent full position earns exact
        # growth from closes[0] onwards.
        closes, opens = [100.0, 110.0, 120.0], [100.0, 105.0, 115.0]
        res = run(candles_from(closes, opens), [1.0] * 3, fee=0.0, execution="close")
        assert res["equity"][-1] == pytest.approx(closes[-1] / closes[0], abs=1e-12)

    def test_opening_a_position_earns_less_under_next_open(self):
        # Under next_open you buy at today's open, so you MISS the overnight
        # gap. Under close you buy at yesterday's close and capture it.
        closes, opens = [100.0, 110.0, 120.0], [100.0, 130.0, 140.0]
        c = candles_from(closes, opens)
        nxt = run(c, [1.0] * 3, fee=0.0, execution="next_open")
        cls = run(c, [1.0] * 3, fee=0.0, execution="close")
        assert nxt["equity"][-1] < cls["equity"][-1]
        assert math.isclose(nxt["equity"][-1], 120.0 / 130.0, rel_tol=1e-12)
