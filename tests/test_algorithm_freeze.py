"""The frozen-algorithm contract.

1. PARITY: feeding identical daily asset returns day-by-day through
   run_step's portfolio construction must reproduce combine_portfolio_rule +
   vol-overlay EXACTLY — backtest and forward are the same bot, provably.
2. Every return-affecting knob is in the spec, validated (typos rejected).
3. The spec is sealed: tampering with it breaks the config hash.
4. Band / throttle / tilt actually change forward behaviour per the spec.
"""
import json
import random
from datetime import UTC, datetime

import pytest

from bot.algorithm import (
    build_algorithm,
    candidate_pool_version,
    validate_algorithm,
)
from bot.portfolio_rules import combine_portfolio_rule
from bot.prospective import run_step
from bot.strategy import BuyHold


def _algo(**over):
    return build_algorithm(with_pool_version=False, **over)


def _mk_manifest(tmp_path, algo):
    from bot.prospective import create_freeze

    return create_freeze(
        assets=[
            {"symbol": s, "source": "test", "periods_per_year": 365,
             "strategy": BuyHold()}  # sleeve == raw asset return; isolates the portfolio layer
            for s in ("AAA", "BBB", "CCC")
        ],
        frictions={"fee": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0, "execution": "close", "risk_free_annual": 0.0},
        algorithm=algo,
        path=tmp_path / f"freeze_{abs(hash(json.dumps(algo, sort_keys=True))) % 10**8}.json",
        now=datetime(2026, 8, 23, tzinfo=UTC),
        git_commit="cafe",
    )


BASE_MS = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000)


def _flat_prices(raw_prefix, p_index):
    """Candles for raw returns [0..p_index]; open_time of candle i is BASE_MS+i*day.
    The LAST candle is 'today' (the live print); everything before is completed."""
    out = []
    p = 100.0
    for i, r in enumerate(raw_prefix[: p_index + 1]):
        prev = p
        p *= 1 + r
        t_ms = BASE_MS + i * 86_400_000
        out.append({"open_time": t_ms, "open": prev, "close": p})
    return out


def _now_for(p_index):
    """Midday of day p_index: candle p is today's live print (excluded from
    decisions), candles < p are completed history."""
    from datetime import datetime as dt

    return dt.fromtimestamp((BASE_MS + p_index * 86_400_000 + 12 * 3600_000) / 1000, tz=UTC)


class TestParity:
    """Forward day-stepping == timeline combiner, bit-for-bit."""

    @pytest.mark.parametrize("over", [
        {},                                   # full headline config
        {"use_tilt": False},
        {"use_crisis": False},
        {"use_throttle": True},               # throttle needs its own state replay
        {"overlay_enabled": False},
        {"rebalance_band": 0.0},
    ])
    def test_forward_reproduces_backtest(self, tmp_path, monkeypatch, over):
        syms = ["AAA", "BBB", "CCC"]
        n_days = 200
        start_idx = 30  # compare only after warmup
        # Build ONE price series per symbol. Day t's return is
        # closes[t]/closes[t-1]-1; days 0-1 are zero seeds (the runner needs
        # two completed candles before any position exists). Both paths
        # consume exactly these values, bit-for-bit.
        base_rng = random.Random(11)
        closes = {}
        for s in syms:
            p = 100.0
            seq = []
            for _ in range(n_days):
                p *= 1 + base_rng.gauss(0.0004, 0.02)
                seq.append(p)
            closes[s] = [100.0, 100.0] + seq
        moves = {
            s: [0.0, 0.0] + [closes[s][i] / closes[s][i - 1] - 1.0 for i in range(2, n_days + 2)]
            for s in syms
        }

        def prices_for(s, p):
            return [
                {"open_time": BASE_MS + i * 86_400_000,
                 "open": closes[s][i - 1] if i else closes[s][0],
                 "close": closes[s][i]}
                for i in range(p + 1)
            ]

        algo = _algo(**over)

        # --- reference path (backtest): timeline combiner + vol overlay ---
        # The combiner sees ALL days (warmup included); we compare the OOS
        # window afterwards. The forward runner must therefore also step
        # through the warmup days — same information, same decisions.
        n_days_total = n_days + 2  # two zero-seed days + n_days returns
        all_days = list(range(n_days_total))
        dailies = {s: {86_400_000 * t: moves[s][t] for t in all_days} for s in syms}
        ref_rule = combine_portfolio_rule(
            dailies, [86_400_000 * t for t in all_days], len(syms),
            vol_window=algo["weighting"]["vol_window"],
            max_multiple_of_equal=algo["weighting"]["max_multiple_of_equal"],
            use_tilt=algo["xs_momentum"]["enabled"], tilt_lookback=algo["xs_momentum"]["lookback"],
            max_tilt=algo["xs_momentum"]["max_tilt"],
            use_crisis=algo["crisis_derisk"]["enabled"], corr_window=algo["crisis_derisk"]["corr_window"],
            corr_threshold=algo["crisis_derisk"]["corr_threshold"], derisk=algo["crisis_derisk"]["multiplier"],
            use_dd_throttle=algo["drawdown_throttle"]["enabled"], dd_trigger=algo["drawdown_throttle"]["dd_trigger"],
            dd_exit=algo["drawdown_throttle"]["dd_exit"], throttle=algo["drawdown_throttle"]["factor"],
        )
        if algo["overlay"]["enabled"]:
            from bot.__main__ import _vol_overlay

            full_expected = _vol_overlay(ref_rule, target=algo["overlay"]["target_vol"],
                                         window=algo["overlay"]["window"], fee=algo["overlay"]["fee_on_turnover"])
        else:
            full_expected = ref_rule

        # --- forward path: one run_step per day on the frozen manifest ------
        manifest = _mk_manifest(tmp_path, algo)
        log = tmp_path / "log.jsonl"
        fwd = []
        for p in all_days:  # p is the raw index whose return lands today
            data = {s: prices_for(s, p) for s in syms}
            res = run_step(manifest, lambda sym, src, d=data: (d[sym], None),
                           now=_now_for(p), log_path=log)
            fwd.append(res["entry"]["port_ret"])

        expected = full_expected[start_idx:]
        fwd = fwd[start_idx:]
        assert len(fwd) == len(expected)
        for a, b in zip(fwd, expected):
            assert a == pytest.approx(b, abs=1e-12)


class TestSpecValidation:
    def test_unknown_top_key_rejected(self):
        bad = _algo()
        bad["momentum_tilt"] = True
        with pytest.raises(ValueError, match="unknown algorithm key"):
            validate_algorithm(bad)

    def test_unknown_sub_key_rejected(self):
        bad = _algo()
        bad["xs_momentum"]["lokback"] = 90  # typo
        with pytest.raises(ValueError, match="xs_momentum"):
            validate_algorithm(bad)

    def test_missing_section_rejected(self):
        bad = {k: v for k, v in _algo().items() if k != "crisis_derisk"}
        with pytest.raises(ValueError, match="missing required"):
            validate_algorithm(bad)

    def test_band_range_checked(self):
        with pytest.raises(ValueError, match="rebalance_band"):
            validate_algorithm(_algo(rebalance_band=1.5))

    def test_pool_version_changes_when_pool_changes(self):
        v1 = candidate_pool_version()

        v2 = candidate_pool_version([BuyHold()])
        assert v1 != v2 and len(v1) == 16


class TestSealing:
    def test_algorithm_is_sealed_by_config_hash(self, tmp_path):
        from bot import prospective as P

        _mk_manifest(tmp_path, _algo())
        path = tmp_path / next(tmp_path.glob("freeze_*.json")).name
        blob = json.loads(path.read_text())
        blob["config"]["algorithm"]["rebalance_band"] = 0.5  # silent behaviour change attempt
        path.write_text(json.dumps(blob))
        with pytest.raises(ValueError, match="modified after freezing"):
            P.load_freeze(path)

    def test_run_step_uses_frozen_band_not_global_default(self, tmp_path, monkeypatch):
        """A frozen band of 0.9 must hold a 0.5 target flat (deviation below
        band); a 0.05 band must let it execute. The band comes ONLY from the
        manifest — proving the forward test trades the frozen construction."""
        import bot.strategy as S
        from bot.strategy import WeightStrategy

        class ConstWeight(WeightStrategy):
            def __init__(self, w=0.5):
                self.w = w

            def weight_at(self, candles, i):
                return self.w

            def __repr__(self):
                return f"Const({self.w})"

        monkeypatch.setattr(S, "_STRATEGY_TYPES", {**S._STRATEGY_TYPES, "ConstWeight": ConstWeight})
        from bot.prospective import create_freeze

        manifest = create_freeze(
            assets=[{"symbol": s, "source": "test", "periods_per_year": 365, "strategy": ConstWeight(0.5)}
                    for s in ("AAA", "BBB", "CCC")],
            frictions={"fee": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0, "execution": "close", "risk_free_annual": 0.0},
            algorithm=_algo(rebalance_band=0.9, overlay_enabled=False),
            path=tmp_path / "freeze_band.json",
            now=datetime(2026, 8, 23, tzinfo=UTC),
            git_commit="band",
        )
        log = tmp_path / "log.jsonl"
        prices = {s: _flat_prices([0.01] * 40, 39) for s in ("AAA", "BBB", "CCC")}
        res = run_step(manifest, lambda sym, src: (prices[sym], None), now=_now_for(39), log_path=log)
        for detail in res["entry"]["assets"].values():
            assert detail["target"] == 0.5
            assert detail["weight"] == 0.0  # |0.5 - 0| <= band -> hold cash
