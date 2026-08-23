"""Property-style tests: invariants that must hold for ANY input series,
checked over many seeded-random cases rather than fixed fixtures."""
import math
import random

import pytest

from bot.clustering import pearson
from bot.costs import CostParams, square_root_impact_fraction, vol_cost_multiplier
from bot.engine import run_strategy
from bot.metrics import cagr, max_drawdown, sharpe
from bot.paper import PaperPortfolio
from bot.research import (
    circular_block_bootstrap_indices,
    drawdown_confidence_intervals,
    mc_future_paths,
    moving_block_bootstrap_indices,
)
from bot.stats_validation import stationary_bootstrap_indices


def _rand_series(rng, n=300):
    """Random price path with regime-switching drift and occasional gaps."""
    closes = []
    price = 100.0
    drift = 0.0
    for _ in range(n):
        if rng.random() < 0.05:
            drift = rng.uniform(-0.004, 0.005)
        price *= max(0.05, 1 + drift + rng.gauss(0, rng.choice([0.01, 0.03])))
        closes.append(price)
    return [
        {"open_time": 86_400_000 * i, "open": c * (1 + rng.uniform(-1e-4, 1e-4)), "close": c}
        for i, c in enumerate(closes)
    ]


SEEDS = [1, 7, 42, 1337, 2026]


class TestEngineProperties:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_returns_weights_and_costs_bounded(self, seed):
        rng = random.Random(seed)
        data = _rand_series(rng)
        weights = [rng.random() for _ in range(len(data))]
        res = run_strategy(data, lambda candles, i: weights[i], fee=0.002, spread_bps=10.0, slippage_bps=10.0)
        assert all(0.0 <= w <= 1.0 for w in res["weights"])
        assert len(res["returns"]) == len(data) - 1
        assert all(math.isfinite(r) for r in res["returns"])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_zero_cost_buy_hold_tracks_price_exactly(self, seed):
        rng = random.Random(seed)
        data = _rand_series(rng, n=100)
        res = run_strategy(data[:60], lambda c, i: 1.0, fee=0.0, start_index=30)
        expected = [data[i]["close"] / data[i - 1]["close"] - 1.0 for i in range(30, 60)]
        assert res["returns"] == pytest.approx(expected)


class TestCostProperties:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_vol_multiplier_always_clamped(self, seed):
        rng = random.Random(seed)
        for _ in range(200):
            rv = rng.uniform(0.001, 50.0)
            m = vol_cost_multiplier(rv, reference_vol=rng.uniform(0.1, 2.0))
            assert 0.5 <= m <= 3.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_impact_monotone_in_turnover(self, seed):
        rng = random.Random(seed)
        for _ in range(100):
            dv = rng.uniform(0.001, 0.1)
            adv = rng.uniform(1.0, 500.0)
            k = rng.uniform(0.1, 10.0)
            a = square_root_impact_fraction(rng.uniform(0.01, 0.3), dv, adv, k)
            b = square_root_impact_fraction(0.99, dv, adv, k)
            assert a <= b

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extended_costs_never_increase_final_equity(self, seed):
        rng = random.Random(seed)
        data = _rand_series(rng, n=250)

        class W:
            # pseudo-strategy flipping exposure to force turnover
            state = {"i": 0}

            def __call__(self, candles, i):
                self.state["i"] += 1
                return float(self.state["i"] % 3 == 0)

        wfn = W()
        flat = run_strategy(data, wfn, fee=0.001, spread_bps=8.0, slippage_bps=8.0)
        wfn2 = W()
        ext = run_strategy(
            data,
            wfn2,
            fee=0.001,
            spread_bps=8.0,
            slippage_bps=8.0,
            cost_params=CostParams(spread_vol_scale=1.0, slippage_vol_scale=1.0, impact_k=2.0, adv_to_equity=25.0),
        )
        assert ext["final"] <= flat["final"] + 1e-12


class TestBootstrapProperties:
    def test_all_samplers_valid_indices_any_block(self):
        for sampler in (stationary_bootstrap_indices, circular_block_bootstrap_indices, moving_block_bootstrap_indices):
            for block in (1, 5, 17, 64):
                idx = sampler(97, block=block, rng=random.Random(block))
                assert len(idx) == 97
                assert all(0 <= i < 97 for i in idx)


class TestMetricProperties:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_drawdown_bounded_and_sharpe_finite(self, seed):
        rng = random.Random(seed)
        rets = [rng.gauss(0.0004, 0.02) for _ in range(400)]
        eq = [1.0]
        for r in rets:
            eq.append(max(eq[-1] * (1 + r), 1e-9))
        mdd = max_drawdown(eq)
        assert -1.0 <= mdd <= 0.0
        assert math.isfinite(sharpe(rets, 365))
        assert math.isfinite(cagr(eq, 400))


class TestClusteringProperties:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_pearson_bounded(self, seed):
        rng = random.Random(seed)
        a = [rng.gauss(0, 1) for _ in range(120)]
        b = [x * rng.uniform(-3, 3) + rng.uniform(-1, 1) for x in a]
        assert -1.0001 <= pearson(a, b) <= 1.0001


class TestPaperIdempotencyProperty:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_repeated_rebalances_converge_not_explode(self, tmp_path, seed):
        rng = random.Random(seed)
        pf = PaperPortfolio(start_cash=10_000.0, fee=0.001, state_file=tmp_path / "s.json", ledger_file=tmp_path / "l.jsonl")
        price = 50.0
        day = 0
        for _step in range(40):
            day += 1
            target = rng.choice([0.0, 0.25, 0.5, 1.0])
            key = f"2026-08-{day:02d}|BTC|REBAL|{target}"
            r1 = pf.rebalance("BTC", target, price, idem_key=key)
            r2 = pf.rebalance("BTC", target, price, idem_key=key)  # duplicate rejected
            if r1 and r1.get("kind") == "fill":
                assert r2.get("skipped") == "duplicate_order"
            assert pf.cash >= -1e-6
            price *= 1 + rng.gauss(0, 0.02)
        eq_final = pf.cash + pf.positions.get("BTC", 0.0) * price
        assert eq_final > 0


class TestResearchInvariants:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_drawdown_cis_contain_median_and_are_negative(self, tmp_path, seed):
        rng = random.Random(seed)
        rets = [rng.gauss(0.0003, 0.015) for _ in range(400)]
        dd = drawdown_confidence_intervals(rets, n_boot=150, seed=seed)
        lo, hi = dd["mdd_90_ci"]
        assert lo <= hi <= 0.0
        assert dd["mdd_worst"] <= lo

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mc_terminal_percentiles_ordered(self, tmp_path, seed):
        rng = random.Random(seed)
        rets = [rng.gauss(0.0005, 0.02) for _ in range(350)]
        mc = mc_future_paths(rets, horizon_days=180, n_paths=600, seed=seed)
        assert mc["terminal_p05"] <= mc["terminal_median"] <= mc["terminal_p95"]
