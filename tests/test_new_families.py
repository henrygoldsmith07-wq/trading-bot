"""The two new strategy families, the primitives they rest on, and `Blend`.

These are the additions behind `build_candidates(extended=True)`. They are
tested here rather than only through the selection module because each one
makes a specific structural claim that is either true or the whole
justification for adding it collapses:

  * ChannelBreakout is SCALE INVARIANT — multiply every price by a
    constant and the signal is unchanged. This is the claim that
    distinguishes it from the SMA/momentum families and is what lets one
    parameterisation travel across assets at wildly different price
    levels. If it is false, the family is just TrendVol with extra steps.
  * MeanReversionZ is the only family that BUYS WEAKNESS, and it is gated
    behind a trend filter so it does not do so in a downtrend. Both halves
    are asserted: it must engage on a dip inside an uptrend, and it must
    stay flat below the long-term mean no matter how oversold.
  * Blend must survive a freeze round-trip WITH its weight vector intact.
    Dropping the weights would silently rebuild an equal-weight Ensemble,
    which would quietly undo shrinkage itself.
"""
import math
import random
from statistics import stdev as ref_stdev

import pytest

from bot.strategy import (
    Blend,
    BuyHold,
    ChannelBreakout,
    Ensemble,
    MeanReversionZ,
    TrendVol,
    _rolling_extreme,
    _Series,
    build_candidates,
    strategy_from_spec,
    strategy_to_spec,
)

PPY = 365


def _candles(closes):
    return [{"open_time": i * 86_400_000, "open": c, "close": c} for i, c in enumerate(closes)]


def _random_walk(n=300, seed=1, start=100.0, mu=0.0002, sigma=0.02):
    rng = random.Random(seed)
    px = [start]
    for _ in range(n - 1):
        px.append(px[-1] * (1 + rng.gauss(mu, sigma)))
    return px


def _scaled(candles, k):
    return [{"open_time": c["open_time"], "open": c["open"] * k, "close": c["close"] * k}
            for c in candles]


# --------------------------------------------------------------------------
# _rolling_extreme — the O(n) primitive the channel rules are built on
# --------------------------------------------------------------------------
class TestRollingExtreme:
    def test_matches_brute_force_max(self):
        for seed in range(6):
            vals = _random_walk(160, seed=seed)
            for window in (1, 2, 7, 20, 55):
                got = _rolling_extreme(vals, window, True)
                for i in range(len(vals)):
                    if i < window - 1:
                        assert math.isnan(got[i])
                    else:
                        assert got[i] == pytest.approx(max(vals[i - window + 1: i + 1]))

    def test_matches_brute_force_min(self):
        for seed in range(6):
            vals = _random_walk(160, seed=seed)
            for window in (1, 3, 13, 40):
                got = _rolling_extreme(vals, window, False)
                for i in range(window - 1, len(vals)):
                    assert got[i] == pytest.approx(min(vals[i - window + 1: i + 1]))

    def test_monotone_input_is_the_deque_worst_case(self):
        """A strictly falling series forces a full drain on every step, and a
        rising one never drains. Both must agree with brute force."""
        rising = [float(i) for i in range(200)]
        falling = list(reversed(rising))
        for vals in (rising, falling):
            for is_max in (True, False):
                got = _rolling_extreme(vals, 25, is_max)
                for i in range(24, len(vals)):
                    chunk = vals[i - 24: i + 1]
                    want = max(chunk) if is_max else min(chunk)
                    assert got[i] == pytest.approx(want)

    def test_constant_series(self):
        vals = [7.5] * 50
        for is_max in (True, False):
            got = _rolling_extreme(vals, 10, is_max)
            assert all(v == pytest.approx(7.5) for v in got[9:])

    @pytest.mark.parametrize("window", [0, -3])
    def test_non_positive_window_yields_all_nan(self, window):
        assert all(math.isnan(v) for v in _rolling_extreme([1.0, 2.0, 3.0], window, True))

    def test_empty_series(self):
        assert _rolling_extreme([], 10, True) == []


class TestSeriesStdev:
    def test_matches_statistics_stdev(self):
        closes = _random_walk(120, seed=11)
        s = _Series(_candles(closes))
        for end in (30, 60, 119):
            for period in (2, 10, 30):
                assert s.stdev(end, period) == pytest.approx(ref_stdev(closes[end - period: end]))

    def test_returns_zero_when_not_enough_history(self):
        s = _Series(_candles(_random_walk(20, seed=3)))
        assert s.stdev(5, 30) == 0.0   # window longer than the series
        assert s.stdev(20, 1) == 0.0   # period < 2

    def test_is_in_price_units(self):
        """Scaling the price scales the stdev by the same factor — this is
        what makes the z-score dimensionless."""
        closes = _random_walk(120, seed=5)
        lo = _Series(_candles(closes))
        hi = _Series(_candles([c * 1000.0 for c in closes]))
        assert hi.stdev(119, 30) == pytest.approx(lo.stdev(119, 30) * 1000.0)


# --------------------------------------------------------------------------
# ChannelBreakout
# --------------------------------------------------------------------------
class TestChannelBreakout:
    def test_weights_stay_in_unit_range(self):
        strat = ChannelBreakout()
        candles = _candles(_random_walk(400, seed=2))
        for i in range(len(candles)):
            w = strat.weight_at(candles, i)
            assert 0.0 <= w <= 1.0

    def test_flat_before_the_channel_fills(self):
        strat = ChannelBreakout(channel=50, vol_window=20)
        candles = _candles(_random_walk(400, seed=2))
        warmup = strat.channel + strat.vol_window + 1
        for i in range(warmup):
            assert strat.weight_at(candles, i) == 0.0

    def test_scale_invariance(self):
        """The headline claim: the signal depends on WHERE price sits inside
        its own range, not on the price level. A SMA rule would fail this.
        """
        strat = ChannelBreakout()
        candles = _candles(_random_walk(400, seed=4))
        for k in (0.001, 0.37, 1.0, 8.0, 1e5):
            scaled = _scaled(candles, k)
            for i in (120, 200, 320, 399):
                assert strat.weight_at(scaled, i) == pytest.approx(
                    strat.weight_at(candles, i), abs=1e-9
                ), f"scale {k} changed the signal at bar {i}"

    def test_engages_near_the_top_of_the_channel(self):
        """A fresh high, in a low-vol uptrend, should be fully on."""
        closes = [100.0 + 0.05 * i for i in range(200)]      # steady uptrend
        closes += [110.0] * 20                                # then flat
        closes.append(115.0)                                  # breakout bar
        strat = ChannelBreakout(channel=20, entry_frac=0.80, exit_frac=0.40, target_vol=1.0)
        candles = _candles(closes)
        assert strat.weight_at(candles, len(closes)) > 0.0

    def test_hysteresis_band_resolves_at_the_midpoint(self):
        """Inside the dead band the state must be recovered from the bar
        alone — `weight_at` cannot carry state, because the forward runner
        re-derives one day's weight from history in isolation."""
        strat = ChannelBreakout(channel=20, entry_frac=0.80, exit_frac=0.40,
                                vol_window=20, target_vol=1.0)
        mid = 0.5 * (0.80 + 0.40)

        # Build a series whose range position sits just under / over the
        # midpoint: a channel of [100, 110] and a chosen last close.
        closes = [100.0 + (i % 11) for i in range(40)]        # range 100..110
        closes = [c for c in closes if c <= 110.0]
        base = _candles(closes)
        just_below = _candles(closes + [100.0 + 10.0 * (mid - 0.02)])
        just_above = _candles(closes + [100.0 + 10.0 * (mid + 0.02)])
        bottom = _candles(closes + [100.0])

        assert strat.weight_at(bottom, len(bottom)) == 0.0
        # Deterministic, and opposite either side of the midpoint.
        w_below = strat.weight_at(just_below, len(just_below))
        w_above = strat.weight_at(just_above, len(just_above))
        assert w_below == 0.0
        assert w_above > 0.0
        # Calling out of order, or twice, must not change the answer.
        assert strat.weight_at(just_above, len(just_above)) == w_above
        _ = strat.weight_at(base, 10)
        assert strat.weight_at(just_above, len(just_above)) == w_above


# --------------------------------------------------------------------------
# MeanReversionZ
# --------------------------------------------------------------------------
class TestMeanReversionZ:
    def test_weights_stay_in_unit_range(self):
        strat = MeanReversionZ()
        candles = _candles(_random_walk(600, seed=6))
        for i in range(len(candles)):
            w = strat.weight_at(candles, i)
            assert 0.0 <= w <= 1.0

    def test_flat_before_warmup(self):
        strat = MeanReversionZ()
        candles = _candles(_random_walk(600, seed=6))
        warmup = max(strat.window, strat.trend_filter) + strat.vol_window + 1
        for i in range(warmup):
            assert strat.weight_at(candles, i) == 0.0

    def test_scale_invariance(self):
        strat = MeanReversionZ()
        candles = _candles(_random_walk(600, seed=8))
        for k in (0.002, 0.5, 3.0, 1e4):
            scaled = _scaled(candles, k)
            for i in (300, 420, 599):
                assert strat.weight_at(scaled, i) == pytest.approx(
                    strat.weight_at(candles, i), abs=1e-9
                ), f"scale {k} changed the signal at bar {i}"

    def test_buys_a_dip_inside_an_uptrend(self):
        """The whole point of the family: it engages when price is BELOW its
        own mean, which is where every trend family is flat or short.

        The dip has to clear BOTH bars: far enough below the short mean to
        be oversold, but still above the long mean so the trend filter lets
        it through. A deeper dip makes the trend filter block instead, and
        this would silently become a test of the filter rather than of the
        reversion — so both preconditions are asserted.
        """
        closes = [100.0 + 0.10 * i for i in range(400)]   # long uptrend
        closes += [140.0] * 40                            # plateau
        closes += [136.0]                                 # dip: ~5 sd down, still > 200d mean
        candles = _candles(closes)
        strat = MeanReversionZ(window=30, entry_z=-0.75, exit_z=0.50,
                               trend_filter=200, vol_window=20, target_vol=1.0)
        i = len(closes)
        s = _Series(candles)
        assert s.close[i - 1] > s.mean(i, strat.trend_filter)          # filter is open
        assert (s.close[i - 1] - s.mean(i, strat.window)) / s.stdev(i, strat.window) <= strat.entry_z
        assert strat.weight_at(candles, i) > 0.0

    def test_trend_filter_blocks_buying_weakness_in_a_downtrend(self):
        """Unfiltered mean reversion in a downtrend is the classic way to
        lose money slowly, so below the long-term mean it must stay flat —
        however oversold the z-score gets."""
        closes = [200.0 - 0.10 * i for i in range(400)]   # long downtrend
        closes += [160.0] * 40
        closes += [140.0]                                  # deeply oversold
        strat = MeanReversionZ(window=30, entry_z=-0.75, exit_z=0.50,
                               trend_filter=200, vol_window=20, target_vol=1.0)
        candles = _candles(closes)
        for i in range(380, len(closes)):
            assert strat.weight_at(candles, i) == 0.0

    def test_disagrees_with_the_trend_families_by_construction(self):
        """A sanity check on the diversity claim: on a dip inside an uptrend,
        TrendVol is off and MeanReversionZ is on. If they agreed, adding the
        new family would add candidates without adding families."""
        closes = [100.0 + 0.10 * i for i in range(400)]
        closes += [140.0] * 40
        closes += [136.0]
        candles = _candles(closes)
        i = len(closes)
        trend = TrendVol(50, 20, 0.30).weight_at(candles, i)
        rev = MeanReversionZ(window=30, entry_z=-0.75, exit_z=0.50,
                             trend_filter=200, vol_window=20, target_vol=1.0).weight_at(candles, i)
        assert trend == 0.0
        assert rev > 0.0
        # They must not merely differ in magnitude — the trend rule has to be
        # completely absent at the exact bar the reversion rule enters.
        assert trend != rev


# --------------------------------------------------------------------------
# Blend
# --------------------------------------------------------------------------
class TestBlendValidation:
    def test_rejects_empty_members(self):
        with pytest.raises(ValueError, match="at least one member"):
            Blend([])

    def test_rejects_weight_length_mismatch(self):
        with pytest.raises(ValueError, match="must match members"):
            Blend([BuyHold(), BuyHold()], [0.5])

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="non-negative"):
            Blend([BuyHold(), BuyHold()], [1.5, -0.5])

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError, match="positive number"):
            Blend([BuyHold(), BuyHold()], [0.0, 0.0])

    def test_default_weights_are_equal_weight(self):
        b = Blend([BuyHold(), TrendVol(50, 20, 0.30), ChannelBreakout()])
        assert sum(b.weights) == pytest.approx(1.0)
        assert b.weights == pytest.approx([1 / 3] * 3)


class TestBlendSemantics:
    def test_weight_at_is_the_convex_combination(self):
        candles = _candles(_random_walk(300, seed=9))
        held, cash = BuyHold(), TrendVol(50, 20, 0.30)
        b = Blend([held, cash], [0.25, 0.75])
        for i in (10, 120, 299):
            want = 0.25 * held.weight_at(candles, i) + 0.75 * cash.weight_at(candles, i)
            assert b.weight_at(candles, i) == pytest.approx(want)

    def test_equal_weights_reproduce_ensemble(self):
        """Ensemble is the equal-weight special case, so the two must agree
        member-for-member — otherwise Blend is not a generalisation."""
        candles = _candles(_random_walk(300, seed=10))
        members = [BuyHold(), TrendVol(50, 20, 0.30)]
        b = Blend(list(members))
        e = Ensemble(list(members))
        for i in (10, 120, 299):
            assert b.weight_at(candles, i) == pytest.approx(e.weight_at(candles, i))

    def test_weights_are_normalised_to_unlevered(self):
        b = Blend([BuyHold(), BuyHold()], [3.0, 1.0])
        assert b.weights == pytest.approx([0.75, 0.25])

    def test_output_clamped_to_unit_range(self):
        """A blend of long-only members can never exceed 1, but the clamp is
        asserted explicitly because a bad weight vector is exactly how a
        leverage bug would enter."""
        candles = _candles(_random_walk(300, seed=12))
        b = Blend([BuyHold(), BuyHold()], [0.5, 0.5])
        for i in range(len(candles)):
            assert 0.0 <= b.weight_at(candles, i) <= 1.0


class TestBlendFreezeRoundTrip:
    """The freeze pins a strategy by content. A Blend that came back from a
    round-trip with equal weights would be a DIFFERENT strategy wearing the
    same repr — the shrinkage would be undone silently, at the exact moment
    the freeze is supposed to guarantee reproducibility."""

    def test_weights_survive_a_round_trip(self):
        b = Blend([TrendVol(50, 20, 0.30), BuyHold()], [0.4, 0.6])
        back = strategy_from_spec(strategy_to_spec(b))
        assert isinstance(back, Blend)
        assert back.weights == pytest.approx([0.4, 0.6])
        assert repr(back) == repr(b)

    def test_unnormalised_weights_survive(self):
        b = Blend([BuyHold(), BuyHold()], [3.0, 1.0])
        back = strategy_from_spec(strategy_to_spec(b))
        assert back.weights == pytest.approx([0.75, 0.25])

    def test_nested_containers_round_trip(self):
        inner = Blend([TrendVol(50, 20, 0.30), BuyHold()], [0.3, 0.7])
        outer = Ensemble([inner, ChannelBreakout(20, 0.8, 0.4)])
        back = strategy_from_spec(strategy_to_spec(outer))
        assert isinstance(back, Ensemble)
        assert isinstance(back.members[0], Blend)
        assert back.members[0].weights == pytest.approx([0.3, 0.7])

    def test_round_trip_preserves_behaviour_not_just_data(self):
        candles = _candles(_random_walk(320, seed=13))
        b = Blend([TrendVol(50, 20, 0.30), ChannelBreakout()], [0.35, 0.65])
        back = strategy_from_spec(strategy_to_spec(b))
        for i in (150, 250, 319):
            assert back.weight_at(candles, i) == pytest.approx(b.weight_at(candles, i))

    def test_ensemble_without_weights_still_round_trips(self):
        """Regression guard: the generic container walk must not have broken
        plain Ensembles on the way to supporting Blend."""
        e = Ensemble([TrendVol(25, 20, 0.30), TrendVol(100, 20, 0.30)])
        back = strategy_from_spec(strategy_to_spec(e))
        assert isinstance(back, Ensemble)
        assert not isinstance(back, Blend)
        assert repr(back) == repr(e)


# --------------------------------------------------------------------------
# Pool composition
# --------------------------------------------------------------------------
class TestPoolComposition:
    def test_default_pool_is_unchanged(self):
        """The default pool size and its every member are part of the frozen
        candidate_pool_version. Changing them would invalidate every
        existing backtest, so this is a hard pin."""
        assert len(build_candidates()) == 85

    def test_extended_pool_is_a_strict_superset(self):
        default = [repr(c) for c in build_candidates()]
        extended = [repr(c) for c in build_candidates(extended=True)]
        assert len(extended) > len(default)
        assert set(default) <= set(extended)
        assert len(set(extended)) == len(extended)   # no duplicates

    def test_every_extended_candidate_is_instantiable_and_bounded(self):
        candles = _candles(_random_walk(700, seed=14))
        for cand in build_candidates(extended=True):
            for i in (400, 550, 699):
                w = cand.weight_at(candles, i)
                assert 0.0 <= w <= 1.0, repr(cand)
                assert w == w, f"{cand!r} produced NaN at bar {i}"

    def test_new_families_are_absent_from_the_default_pool(self):
        kinds = {type(c).__name__ for c in build_candidates()}
        assert "ChannelBreakout" not in kinds
        assert "MeanReversionZ" not in kinds
        kinds_ext = {type(c).__name__ for c in build_candidates(extended=True)}
        assert {"ChannelBreakout", "MeanReversionZ"} <= kinds_ext
