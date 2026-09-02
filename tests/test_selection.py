"""Robust selection: the estimators, the rule, and its auditability.

The headline claim of `bot/selection.py` is that replacing
"argmax in-sample Sharpe" with a robustness-aware rule changes which
strategy gets picked, in the direction of the one whose record does not
depend on a single lucky regime. That claim is tested directly below by
constructing two candidates where argmax and the robust rule disagree,
and asserting each picks the one it is supposed to.
"""
import math
import random

import pytest

from bot.metrics import sharpe
from bot.selection import (
    DEGENERATE_SHARPE,
    credibility_weight,
    evaluate_candidates,
    make_one_se_selection_fn,
    make_robust_selection_fn,
    sharpe_standard_error,
    subwindow_sharpes,
    worst_subwindow_sharpe,
)
from bot.strategy import Blend, BuyHold, TrendVol, build_candidates

PPY = 365

# Candidate A: a spectacular first third, a catastrophic middle third, a
# flat last third. Highest OVERALL Sharpe of the pair.
SPIKY = [0.0035] * 100 + [-0.0030] * 100 + [0.0006] * 100
# Candidate B: modest but positive in every third. NOTE: the Random instance
# must be created ONCE outside the comprehension — re-seeding it inside would
# produce 300 identical "random" draws, zero variance, and a Sharpe of
# exactly 0.0, which silently breaks every test below.
_STEADY_RNG = random.Random(3)
STEADY = [0.0009 + _STEADY_RNG.gauss(0.0, 0.004) for _ in range(300)]


def _candles(n: int = 300, seed: int = 1, mu: float = 0.0004) -> list[dict]:
    """A real candle list, for the paths that are NOT mocked.

    Shrinkage backtests the prior for real, so a test that passed an empty
    slice would silently take the "prior unmeasurable" branch and never
    construct a Blend at all.

    `mu` is the per-bar drift. It is a parameter because shrinkage compares
    the pick against a REAL backtest of the prior, so whatever this series
    happens to do decides the sign of the gap — and a default-drift path
    is perfectly capable of making buy-and-hold look better than the pick.
    Tests that need the gap to be positive say so explicitly.
    """
    rng = random.Random(seed)
    px = [100.0]
    for _ in range(n - 1):
        px.append(px[-1] * (1 + rng.gauss(mu, 0.02)))
    return [{"open_time": i * 86_400_000, "open": p, "close": p} for i, p in enumerate(px)]


def _records(pairs):
    """Build synthetic candidate records shaped like `evaluate_candidates`
    output. Each pair is (name, sharpe, returns, turnover)."""
    return [
        {"strategy": name, "sharpe": s, "returns": r, "turnover": t}
        for (name, s, r, t) in pairs
    ]


class TestSharpeStandardError:
    def test_matches_the_one_over_sqrt_years_rule_of_thumb(self):
        # 3 years of daily data, SR ~ 0: SE should be about 1/sqrt(3)
        se = sharpe_standard_error(0.0, 3 * 365, PPY)
        assert se == pytest.approx(1 / math.sqrt(3), abs=0.01)

    def test_falls_with_more_observations(self):
        a = sharpe_standard_error(1.0, 365, PPY)
        b = sharpe_standard_error(1.0, 4 * 365, PPY)
        assert a > b > 0

    def test_rises_with_sharpe_magnitude(self):
        # the (1 + SR^2/2) term: a bigger estimated Sharpe is less certain
        assert sharpe_standard_error(2.0, 1095, PPY) > sharpe_standard_error(0.5, 1095, PPY)

    def test_too_few_observations_is_infinite_not_zero(self):
        # returning 0 would make an unmeasurable Sharpe look certain
        assert sharpe_standard_error(1.0, 3, PPY) == float("inf")

    def test_three_years_cannot_distinguish_one_from_zero(self):
        # the practical point: SR=1.0 on 3y of daily data is ~1.7 SE from zero
        se = sharpe_standard_error(1.0, 1095, PPY)
        assert 1.0 / se < 2.0


class TestSubwindowSharpes:
    def test_splits_into_contiguous_thirds(self):
        out = subwindow_sharpes([0.001] * 300, 3, PPY)
        assert len(out) == 3

    def test_worst_is_the_minimum(self):
        out = subwindow_sharpes(STEADY, 3, PPY)
        assert worst_subwindow_sharpe(STEADY, 3, PPY) == min(out)

    def test_a_steady_loss_scores_lower_than_a_volatile_winner(self):
        """The degenerate case that motivated DEGENERATE_SHARPE.

        A block that loses the same amount every single day has zero
        variance, so a naive Sharpe scores it 0.0 — ranking the worst
        possible regime above a merely noisy profitable one.
        """
        steady_loss = [-0.003] * 100
        assert subwindow_sharpes(steady_loss, 1, PPY)[0] == -DEGENERATE_SHARPE

    def test_a_flat_book_stays_neutral_even_with_a_cash_rate(self):
        # zero returns and zero variance must not be punished as a loss
        assert subwindow_sharpes([0.0] * 300, 3, PPY, 0.03) == [0.0, 0.0, 0.0]

    def test_no_sharpe_explodes_from_floating_point_dust(self):
        # a near-constant block must not produce a 1e17-style value
        vals = subwindow_sharpes([0.0035] * 100, 1, PPY)
        assert vals[0] == DEGENERATE_SHARPE


class TestCredibilityWeight:
    def test_no_gap_means_trust_the_prior(self):
        assert credibility_weight(0.0, 0.5) == 0.0

    def test_gap_equal_to_noise_is_the_break_even(self):
        assert credibility_weight(0.5, 0.5) == pytest.approx(0.5)

    def test_a_large_gap_earns_almost_full_weight(self):
        assert credibility_weight(10.0, 0.5) > 0.95

    def test_monotonic_in_the_gap(self):
        ws = [credibility_weight(g, 0.5) for g in (0.1, 0.5, 1.0, 2.0)]
        assert ws == sorted(ws) and len(set(ws)) == 4

    def test_unmeasurable_noise_means_trust_the_prior(self):
        assert credibility_weight(5.0, float("inf")) == 0.0

    def test_negative_gap_is_never_rewarded(self):
        assert credibility_weight(-1.0, 0.5) == 0.0


class TestRuleDisagrees:
    """argmax and the robust rule must pick different winners here, and the
    robust rule must pick the one that did not depend on one regime."""

    def test_preconditions_hold(self):
        assert sharpe(SPIKY, PPY) > sharpe(STEADY, PPY)          # argmax prefers SPIKY
        assert worst_subwindow_sharpe(STEADY, 3, PPY) > worst_subwindow_sharpe(SPIKY, 3, PPY)
        se = sharpe_standard_error(sharpe(SPIKY, PPY), len(SPIKY), PPY)
        assert sharpe(STEADY, PPY) >= sharpe(SPIKY, PPY) - se    # and it is within 1 SE

    def test_robust_rule_picks_the_steady_candidate(self, monkeypatch):
        import bot.selection as sel

        recs = _records([
            ("spiky", sharpe(SPIKY, PPY), SPIKY, 5.0),
            ("steady", sharpe(STEADY, PPY), STEADY, 5.0),
        ])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn()
        picked = select(["spiky", "steady"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert picked == "steady"

    def test_plain_argmax_would_have_picked_the_spiky_candidate(self, monkeypatch):
        """The control: on the same records, argmax picks the other one. If
        both rules agreed, the whole module would be decorative."""
        import bot.selection as sel

        recs = _records([
            ("spiky", sharpe(SPIKY, PPY), SPIKY, 5.0),
            ("steady", sharpe(STEADY, PPY), STEADY, 5.0),
        ])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        best = max(recs, key=lambda r: r["sharpe"])
        assert best["strategy"] == "spiky"

    def test_one_se_widens_the_eligible_set(self, monkeypatch):
        import bot.selection as sel

        recs = _records([
            ("spiky", sharpe(SPIKY, PPY), SPIKY, 5.0),
            ("steady", sharpe(STEADY, PPY), STEADY, 5.0),
        ])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_one_se_selection_fn(tie_break="worst_subwindow")
        select(["spiky", "steady"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert select.last_diagnostics["n_eligible"] == 2
        assert select.last_diagnostics["n_trials"] == 2

    def test_lowest_turnover_tie_break_prefers_the_cheaper_candidate(self, monkeypatch):
        import bot.selection as sel

        # identical Sharpes, so the whole eligible set is tied and only the
        # tie-break decides
        recs = _records([("expensive", 1.0, STEADY, 9.0), ("cheap", 1.0, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_one_se_selection_fn(tie_break="lowest_turnover")
        picked = select(["expensive", "cheap"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert picked == "cheap"


class TestAuditability:
    """A selection rule that cannot be deflated must not be usable."""

    def test_selector_publishes_every_trial_sharpe(self, monkeypatch):
        import bot.selection as sel

        recs = _records([("a", 1.0, STEADY, 1.0), ("b", 0.5, STEADY, 1.0), ("c", 0.2, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn()
        select(["a", "b", "c"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert select.last_trial_sharpes == [1.0, 0.5, 0.2]

    def test_diagnostics_record_the_rule_actually_applied(self, monkeypatch):
        import bot.selection as sel

        recs = _records([("a", 1.0, STEADY, 1.0), ("b", 0.9, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn()
        select(["a", "b"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        d = select.last_diagnostics
        assert d["rule"] == "robust"
        assert d["n_trials"] == 2 and d["n_ran"] == 2
        assert d["sharpe_se"] > 0
        assert d["threshold"] == pytest.approx(d["best_train_sharpe"] - d["sharpe_se"])

    def test_no_candidates_returns_none_not_a_crash(self, monkeypatch):
        import bot.selection as sel

        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: [])
        select = make_robust_selection_fn()
        assert select([], [], {"periods_per_year": PPY, "risk_free_annual": 0.0}) is None
        assert select.last_diagnostics["n_trials"] == 0


class TestWalkForwardIntegration:
    def test_trial_sharpes_are_recorded_for_a_custom_selector(self):
        """Without this, a robust-selection run would report an empty trial
        record and its Sharpe could not be deflated at all."""
        from bot.selection import make_one_se_selection_fn
        from bot.walkforward import walk_forward

        rng = random.Random(5)
        px = [100.0]
        for _ in range(1500):
            px.append(px[-1] * (1 + rng.gauss(0.0004, 0.02)))
        candles = [{"open_time": i * 86_400_000, "open": p, "close": p} for i, p in enumerate(px)]

        wf = walk_forward(
            candles,
            candidates=[BuyHold(), TrendVol(50, 20, 0.30), TrendVol(100, 20, 0.30)],
            train_days=730,
            test_days=180,
            selection_fn=make_one_se_selection_fn(),
        )
        # 2+ folds x 3 candidates: the trials must all be visible
        assert len(wf["trial_sharpes"]) >= 2 * 3
        assert all(isinstance(s, float) for s in wf["trial_sharpes"])

    def test_builtin_argmax_behaviour_is_unchanged(self):
        """The default path must not move: the robust rules are opt-in."""
        from bot.walkforward import walk_forward

        rng = random.Random(5)
        px = [100.0]
        for _ in range(1500):
            px.append(px[-1] * (1 + rng.gauss(0.0004, 0.02)))
        candles = [{"open_time": i * 86_400_000, "open": p, "close": p} for i, p in enumerate(px)]
        cands = [BuyHold(), TrendVol(50, 20, 0.30), TrendVol(100, 20, 0.30)]

        a = walk_forward(candles, candidates=cands, train_days=730, test_days=180)
        b = walk_forward(candles, candidates=cands, train_days=730, test_days=180)
        assert a["daily"] == b["daily"]          # deterministic
        assert len(a["trial_sharpes"]) == len(b["trial_sharpes"])


class TestEvaluateCandidates:
    def test_runs_every_candidate_and_returns_returns(self):
        rng = random.Random(9)
        px = [100.0]
        for _ in range(600):
            px.append(px[-1] * (1 + rng.gauss(0.0004, 0.02)))
        candles = [{"open_time": i * 86_400_000, "open": p, "close": p} for i, p in enumerate(px)]

        recs = evaluate_candidates([BuyHold(), TrendVol(50, 20, 0.30)], candles,
                                   {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert len(recs) == 2
        assert all(len(r["returns"]) > 2 for r in recs)
        assert all(isinstance(r["sharpe"], float) for r in recs)

    def test_the_full_pool_evaluates_without_error(self):
        rng = random.Random(9)
        px = [100.0]
        for _ in range(400):
            px.append(px[-1] * (1 + rng.gauss(0.0004, 0.02)))
        candles = [{"open_time": i * 86_400_000, "open": p, "close": p} for i, p in enumerate(px)]
        recs = evaluate_candidates(build_candidates(extended=True), candles,
                                   {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert len(recs) > 0


class TestBlendShrinkage:
    def test_shrinkage_returns_a_blend_of_pick_and_prior(self, monkeypatch):
        import bot.selection as sel

        # The prior must be a real strategy: the rule backtests it on the
        # training window to measure the gap, so a stand-in object without
        # `weight_at` would only test that the code crashes.
        prior = BuyHold()
        # The reported Sharpe is deliberately decoupled from `returns` here:
        # `returns` drives the standard error, the scalar drives the gap, and
        # keeping them independent lets the test set each on its own.
        recs = _records([("pick", 1.0, STEADY, 1.0), ("other", 0.95, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)
        monkeypatch.setattr(sel, "risk_ensemble", lambda: prior)

        select = make_robust_selection_fn(shrink=True, prior=prior)
        # A declining market, so buy-and-hold is measurably worse than the
        # pick. On a rising path the prior wins, the gap goes negative, and
        # alpha collapses to 0.0 — which is correct behaviour, just not the
        # behaviour this test is here to pin down.
        train = _candles(mu=-0.002)
        picked = select(["pick", "other"], train, {"periods_per_year": PPY, "risk_free_annual": 0.0})

        diag = select.last_diagnostics
        assert isinstance(picked, Blend)
        assert picked.members[1] is prior
        # The prior must actually have been measured. Without this, a
        # regression that makes the prior unmeasurable would report
        # alpha == 0.0 and silently pass the bounds check below.
        assert diag["prior_train_sharpe"] is not None
        assert diag["shrink_gap"] > 0.0
        assert 0.0 < diag["shrink_alpha"] < 1.0
        assert sum(picked.weights) == pytest.approx(1.0)
        assert picked.weights[0] == pytest.approx(diag["shrink_alpha"])

    def test_prior_beating_the_pick_shrinks_to_the_prior_entirely(self, monkeypatch):
        """alpha == 0.0 is a real outcome, not a failure.

        Credibility is gap / (gap + se) with the gap floored at zero: if the
        pick cannot beat buy-and-hold on the training window, there is no
        evidence it has skill, so it gets no weight. The important property
        is that this is distinguishable from "the prior could not be
        measured" — hence the prior's Sharpe and the gap in the diagnostics.
        """
        import bot.selection as sel

        prior = BuyHold()
        recs = _records([("pick", 0.1, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn(shrink=True, prior=prior)
        # A strongly rising market: buy-and-hold crushes the mocked pick.
        picked = select(["pick"], _candles(seed=7, mu=0.004),
                        {"periods_per_year": PPY, "risk_free_annual": 0.0})

        diag = select.last_diagnostics
        assert isinstance(picked, Blend)
        assert diag["prior_train_sharpe"] > 0.1
        assert diag["shrink_gap"] < 0.0
        assert diag["shrink_alpha"] == 0.0
        # Every unit of weight sits on the prior.
        assert picked.weights == pytest.approx([0.0, 1.0])

    def test_unmeasurable_prior_is_reported_as_none_not_zero(self, monkeypatch):
        """A prior that cannot be backtested must not masquerade as a
        measured prior that merely scored badly."""
        import bot.selection as sel

        class Unusable:
            def weight_at(self, candles, i):
                raise ValueError("no data")

        recs = _records([("pick", 1.0, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn(shrink=True, prior=Unusable())
        picked = select(["pick"], _candles(), {"periods_per_year": PPY, "risk_free_annual": 0.0})

        assert not isinstance(picked, Blend)
        assert select.last_diagnostics["prior_train_sharpe"] is None
        assert select.last_diagnostics["shrink_alpha"] == 1.0

    def test_no_shrinkage_returns_the_bare_pick(self, monkeypatch):
        import bot.selection as sel

        recs = _records([("a", 1.0, STEADY, 1.0), ("b", 0.95, STEADY, 1.0)])
        monkeypatch.setattr(sel, "evaluate_candidates", lambda *a, **k: recs)

        select = make_robust_selection_fn(shrink=False)
        picked = select(["a", "b"], [], {"periods_per_year": PPY, "risk_free_annual": 0.0})
        assert not isinstance(picked, Blend)
        assert select.last_diagnostics["shrink_alpha"] == 1.0

    def test_unknown_tie_break_is_rejected(self):
        with pytest.raises(ValueError, match="unknown tie_break"):
            make_one_se_selection_fn(tie_break="vibes")
