"""LEAKAGE CANARIES — proving a timestamp cannot access later observations.

One fixed price history. For a sweep of decision bars i, CORRUPT every bar
j > i (garbage prices) and require that anything decided AT i is unchanged
versus the clean series.

Covered paths:
  1. engine.run_strategy     — per-bar weights & daily returns
  2. walk-forward geometry   — folds/OOS stream from prefixes only
  3. strategy pool           — weight_at(i) for every candidate family
  4. prospective.run_step    — live forward runner, including the strict rule
                               that strictly-FUTURE-dated prints are ignored

If any canary goes red, historical results AND the forward tape are both
contaminated by future information. Do not ship past a red canary.
"""
from datetime import UTC, datetime, timedelta

import pytest

from bot.algorithm import build_algorithm
from bot.engine import run_strategy
from bot.prospective import create_freeze, load_log, run_step
from bot.strategy import (
    Blend,
    BuyHold,
    ChannelBreakout,
    MacdTrend,
    MeanReversionZ,
    RsiDipBuy,
    SmaCrossover,
    TrendVol,
    build_candidates,
    risk_ensemble,
)


def _candles(n=260):
    closes = []
    px = 100.0
    for i in range(n):
        px *= 1 + ((i * 37 % 11) - 5) / 900.0  # deterministic wobble
        closes.append(px)
    return [
        {"open_time": 1_700_000_000_000 + i * 86_400_000,
         "open": closes[i - 1] if i else closes[0],
         "high": max(closes[i], closes[i - 1] if i else closes[0]) * 1.001,
         "low": min(closes[i], closes[i - 1] if i else closes[0]) * 0.999,
         "close": closes[i],
         "volume": 1000.0}
        for i in range(n)
    ]


def _poison_bar(price, open_time):
    """A plausible-but-wrong future print (3x true price)."""
    return {"open_time": open_time, "open": price * 0.999, "high": price * 1.001,
            "low": price * 0.998, "close": price, "volume": 1000.0}


def _corrupt_after(candles, last_valid_index):
    """Replace every bar AFTER `last_valid_index` with a plausible-but-wrong
    print (3x the true price) — valid enough for any indicator to compute,
    wrong enough that any leak changes results."""
    out = [dict(c) for c in candles]
    for j in range(last_valid_index + 1, len(out)):
        base = out[j]["close"] * 3.0
        out[j] = {"open_time": out[j]["open_time"], "open": base * 0.999,
                  "high": base * 1.001, "low": base * 0.998,
                  "close": base, "volume": 1000.0}
    return out


# The two families added for the extended pool are listed explicitly rather
# than relied on appearing in `build_candidates()`: they are deliberately NOT
# in the default pool (so the shipped candidate_pool_version is unchanged),
# which means a canary that only swept the default pool would silently stop
# covering them the moment they were selected via `extended=True`.
CANDIDATES = [BuyHold(), TrendVol(50, 20, 0.3), SmaCrossover(10, 40),
              RsiDipBuy(2, 60, 150), MacdTrend(),
              ChannelBreakout(), MeanReversionZ(),
              # A Blend is what a shrunk selection actually returns, so its
              # own weight_at path needs the same guarantee.
              Blend([TrendVol(50, 20, 0.3), BuyHold()], [0.4, 0.6]),
              risk_ensemble()]


class TestEngineNoLookahead:
    def test_prefix_truncation_equals_full_run(self):
        """Decisions at bar i from a truncated feed == decisions at bar i from
        the full feed: later observations cannot reach backward."""
        data = _candles()
        clean = run_strategy(data, TrendVol(50, 20, 0.3).weight_at,
                             fee=0.001, spread_bps=5.0, execution="next_open")
        for i in (60, 120, 200):
            prefix = run_strategy(data[: i + 1], TrendVol(50, 20, 0.3).weight_at,
                                  fee=0.001, spread_bps=5.0, execution="next_open")
            assert prefix["returns"] == pytest.approx(clean["returns"][: len(prefix["returns"])])
            assert prefix["weights"] == clean["weights"][: len(prefix["weights"])]

    def test_corrupted_future_bars_cannot_change_past_bars(self):
        """Even when the caller hands back a series whose LATER bars were
        replaced by poison, the prefix results stay identical."""
        data = _candles()
        clean = run_strategy(data, TrendVol(50, 20, 0.3).weight_at,
                             fee=0.001, spread_bps=5.0, execution="next_open")
        cut = 150
        poisoned_series = _corrupt_after(data, cut)
        res = run_strategy(poisoned_series, TrendVol(50, 20, 0.3).weight_at,
                           fee=0.001, spread_bps=5.0, execution="next_open")
        # bars up to `cut` are untouched, so their returns must match exactly;
        # the poisoned region produces its own (garbage-in) values afterwards
        assert res["returns"][:cut] == pytest.approx(clean["returns"][:cut])
        assert res["weights"][: cut + 1] == clean["weights"][: cut + 1]


class TestStrategyPool:
    @pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: repr(c))
    def test_weight_at_i_ignores_future_bars(self, candidate):
        data = _candles(300)
        for i in (60, 120, 200, 299):
            w_clean = candidate.weight_at(data, i)
            poisoned = _corrupt_after(data, i)
            assert candidate.weight_at(poisoned, i) == pytest.approx(w_clean), repr(candidate)

    @pytest.mark.parametrize("extended", [False, True], ids=["default_pool", "extended_pool"])
    def test_full_pool_prefix_equivalence(self, extended):
        """Every selectable candidate: weight_at(i) on data[:i+1] equals
        weight_at(i) on the full series.

        Swept for BOTH pools. The extended pool is the one a robustness run
        actually searches, and it is not a subset of the default pool, so
        covering only the default would leave the new families unchecked.
        """
        data = _candles(280)
        for cand in build_candidates(extended=extended):
            for i in (100, 200, 279):
                assert cand.weight_at(data[: i + 1], i) == pytest.approx(
                    cand.weight_at(data, i)
                ), repr(cand)


class TestWalkForward:
    def test_fold_geometry_from_prefix_only(self):
        from bot.walkforward import absolute_folds

        data = _candles(1400)
        full = absolute_folds(data, train_days=365, test_days=180)
        partial = absolute_folds(data[:1100], train_days=365, test_days=180)
        assert partial == full[: len(partial)]

    def test_oos_stream_is_prefix_stable(self):
        from bot.strategy import risk_ensemble
        from bot.walkforward import absolute_folds, walk_forward_at

        data = _candles(1400)
        kw = dict(candidates=[risk_ensemble()], fee=0.001, execution="next_open")
        short = walk_forward_at(data, absolute_folds(data, 365, 180)[:2], **kw)
        longr = walk_forward_at(data, absolute_folds(data, 365, 180)[:4], **kw)
        days_s = sorted(short["daily"])
        days_l = sorted(longr["daily"])
        assert days_s == days_l[: len(days_s)]
        for d in days_s:
            assert short["daily"][d] == pytest.approx(longr["daily"][d])


class TestForwardRunner:
    def _manifest(self, tmp_path):
        return create_freeze(
            assets=[{"symbol": "AAA", "source": "test", "periods_per_year": 365,
                     "session": "continuous", "strategy": TrendVol(50, 20, 0.3)}],
            frictions={"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0,
                       "execution": "next_open", "risk_free_annual": 0.03},
            algorithm=build_algorithm(rebalance_band=0.0, with_pool_version=False),
            path=tmp_path / "f.json",
            git_commit="canary",
        )

    def _now_for(self, day_index):
        base = datetime.fromtimestamp(1_700_000_000_000 / 1000, tz=UTC)
        return base + timedelta(days=day_index, hours=12)

    def test_day_p_decision_invariant_under_appended_future_rows(self, tmp_path, monkeypatch):
        """Feed contains FUTURE-DATED poison rows after today. The runner must
        ignore strictly-future prints and produce the identical day-p entry."""
        monkeypatch.chdir(tmp_path)
        manifest = self._manifest(tmp_path)
        data = _candles(200)

        def run_once(feed):
            log = tmp_path / "l.jsonl"
            if log.exists():
                log.unlink()
            run_step(manifest, lambda s, src: (feed, None),
                     now=self._now_for(90), log_path=log)
            return load_log(log)[-1]

        today_last = [dict(c) for c in data[:91]]           # today = bar 90
        with_future = today_last + [_poison_bar(today_last[-1]["close"] * 3.0,
                                    today_last[-1]["open_time"] + k * 86_400_000)
                                    for k in range(1, 6)]   # five future-dated rows
        e_clean = run_once(today_last)["assets"]["AAA"]
        e_poison = run_once(with_future)["assets"]["AAA"]
        assert e_clean["weight"] == e_poison["weight"]
        assert e_clean["sleeve_ret"] == pytest.approx(e_poison["sleeve_ret"], abs=1e-12)
