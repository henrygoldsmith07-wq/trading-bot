"""Cost calibration: predicted (V1 model) vs observed (paper tape) frictions."""
import json
from datetime import UTC, datetime

import pytest

from bot.cost_calibration import (
    build_observation,
    calibrate,
    format_report,
    load_observations,
)


def _obs(**over):
    base = dict(
        symbol="BTC",
        side="BUY",
        target_weight=1.0,
        previous_weight=0.0,
        decision_close=100.0,
        exec_price=100.05,       # +5bp drift, adverse for a buy
        mark_price=100.10,
        bid=None,
        ask=None,
        predicted_cost_bps=15.0,  # V1: fee 10 + spread 5? -> passed explicitly
        realized_vol_annual=0.55,
        adv30_usd=8e9,
        day_volume_base=1234.0,
    )
    base.update(over)
    return build_observation(**base)


class TestBuildObservation:
    def test_drift_signed_by_side(self):
        buy = _obs(side="BUY")
        sell = _obs(side="SELL", exec_price=99.95)
        assert buy["observed_cost_proxy_bps"] == pytest.approx(5.0)   # buy filled higher = cost
        assert sell["observed_cost_proxy_bps"] == pytest.approx(5.0)  # sell filled lower = cost

    def test_quotes_produce_spread_and_mid(self):
        o = _obs(bid=100.00, ask=100.10)
        assert o["mid"] == pytest.approx(100.05)
        assert o["quoted_spread_bps"] == pytest.approx((0.10 / 100.05) * 1e4)
        assert o["fill_vs_mid_bps"] == pytest.approx((100.05 / 100.05 - 1) * 1e4)

    def test_null_fields_survive(self):
        o = _obs(bid=None, ask=None)
        assert o["mid"] is None and o["quoted_spread_bps"] is None and o["fill_vs_mid_bps"] is None

    def test_flat_turnover_zero_proxy(self):
        o = _obs(target_weight=0.5, previous_weight=0.5, side="FLAT")
        assert o["turnover"] == pytest.approx(0.0)
        assert o["observed_cost_proxy_bps"] is None


class TestCalibrate:
    def _observations(self, n, drift_bps=6.0):
        px = 100.0
        return [_obs(exec_price=px * (1 + drift_bps / 1e4)) for _ in range(n)]

    def test_error_is_observed_minus_predicted(self):
        rep = calibrate(self._observations(50), v1_frictions={"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0})
        assert rep["sufficient"]
        assert rep["predicted_cost_bps"] == pytest.approx(20.0)   # fee 10 + 5 + 5
        assert rep["observed_cost_proxy_bps"] == pytest.approx(6.0)
        assert rep["error_bp"] == pytest.approx(-14.0)
        assert rep["v2_proposal"]["status"] == "proposed"

    def test_insufficient_sample_blocks_v2(self):
        rep = calibrate(self._observations(5), v1_frictions={"fee": 0.001})
        assert not rep["sufficient"]
        assert rep["v2_proposal"]["status"] == "insufficient_data"

    def test_non_turnover_rows_excluded(self):
        rows = self._observations(40)
        flat = _obs(target_weight=0.5, previous_weight=0.5, side="FLAT")
        rep = calibrate(rows + [flat], v1_frictions={"fee": 0.001})
        assert rep["n_turnover_events"] == 40

    def test_v2_suggestion_uses_observed_when_positive(self):
        rep = calibrate(self._observations(40, drift_bps=8.0), v1_frictions={"fee": 0.001})
        assert rep["v2_proposal"]["effective_spread_plus_slippage_bps"] == pytest.approx(8.0)
        assert rep["v2_proposal"]["fee"] == pytest.approx(0.001)

    def test_negative_drift_clamped_in_suggestion(self):
        # fills LUCKIER than the decision: do not propose negative frictions
        rep = calibrate(self._observations(40, drift_bps=-3.0), v1_frictions={"fee": 0.001})
        assert rep["observed_cost_proxy_bps"] == pytest.approx(-3.0)  # reported honestly
        assert rep["v2_proposal"]["effective_spread_plus_slippage_bps"] == pytest.approx(0.0)


class TestRoundtripAndFormat:
    def test_load_roundtrip(self, tmp_path):
        p = tmp_path / "cost_observations.jsonl"
        from bot.cost_calibration import append_observation

        append_observation(_obs(), path=p)
        append_observation(_obs(exec_price=101.0), path=p)
        loaded = load_observations(p)
        assert len(loaded) == 2
        assert loaded[1]["exec_price"] == 101.0

    def test_format_report_contains_key_lines(self):
        rows = [_obs(exec_price=100.06) for _ in range(35)]
        rep = calibrate(rows, v1_frictions={"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0})
        text = format_report(rep)
        assert "predicted trading cost" in text
        assert "observed slippage proxy" in text
        assert "V2 proposal" in text

    def test_json_serializable(self):
        rows = [_obs() for _ in range(31)]
        rep = calibrate(rows, v1_frictions={"fee": 0.001})
        assert isinstance(json.dumps(rep), str)


class TestPersistenceAndFormatEdges:
    def test_write_load_calibration_roundtrip(self, tmp_path):
        from bot.cost_calibration import load_calibration, write_calibration

        rows = [_obs(exec_price=100.06) for _ in range(35)]
        rep = calibrate(rows, v1_frictions={"fee": 0.001})
        path = write_calibration(rep, path=tmp_path / "cost_calibration.json")
        loaded = load_calibration(path)
        assert loaded is not None
        assert loaded["n_turnover_events"] == 35

    def test_load_calibration_missing_or_corrupt(self, tmp_path):
        from bot.cost_calibration import load_calibration

        assert load_calibration(tmp_path / "missing.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{oops")
        assert load_calibration(bad) is None

    def test_format_zero_events(self):
        from bot.cost_calibration import format_report

        text = format_report(calibrate([], v1_frictions={"fee": 0.001}))
        assert "no turnover events" in text

    def test_mean_quoted_spread_line_when_present(self):
        rows = [_obs(bid=100.0, ask=100.10) for _ in range(31)]
        text = format_report(calibrate(rows, v1_frictions={"fee": 0.001}))
        assert "mean quoted spread" in text

    def test_load_observations_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert load_observations(p) == []


class TestVolatilityContextUnits:
    """ADV units regression: Binance quote_volume is ALREADY USD turnover —
    multiplying by close produced price x USD nonsense (e.g. BTC $2B volume
    at $60k recorded as $120T ADV)."""

    def test_adv_is_mean_of_quote_volumes_not_scaled_by_price(self):
        from bot.prospective import _volatility_context

        closes = []
        px = 60_000.0
        for _ in range(30):
            px *= 1.001
            closes.append(px)
        candles = [
            {"open_time": i * 86_400_000,
             "close": closes[i],
             "volume": 1000.0,
             "quote_volume": 2_000_000_000.0}  # $2B/day, constant
            for i in range(30)
        ]
        _rv, adv, _day = _volatility_context(candles)        assert adv == pytest.approx(2_000_000_000.0)          # exact mean
        assert adv < 1e10                                      # not $120T nonsense

    def test_mixed_quote_volumes_average(self):
        from bot.prospective import _volatility_context

        candles = [
            {"close": 50.0 + i, "quote_volume": 3e6 + i * 1e5} for i in range(30)
        ]
        _, adv, _ = _volatility_context(candles)
        assert adv == pytest.approx(sum(3e6 + i * 1e5 for i in range(30)) / 30)

    def test_missing_quote_volume_gives_none_adv_but_rv_still_computed(self):
        from bot.prospective import _volatility_context

        candles = [
            {"close": 100.0 * (1.004 if i % 2 else 0.997)} for i in range(40)
        ]
        rv, adv, day = _volatility_context(candles)
        assert adv is None
        assert rv is not None and rv > 0


class TestRunStepIntegration:
    def test_turnover_writes_observation_with_quotes_and_context(self, tmp_path, monkeypatch):
        from datetime import UTC, datetime

        from bot.algorithm import build_algorithm
        from bot.prospective import create_freeze, run_step
        from bot.strategy import BuyHold

        manifest = create_freeze(
            assets=[{"symbol": "AAA", "source": "binance", "periods_per_year": 365,
                     "strategy": BuyHold()}],
            frictions={"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0,
                       "execution": "next_open", "risk_free_annual": 0.03},
            algorithm=build_algorithm(rebalance_band=0.0, overlay_enabled=False,
                                      use_tilt=False, use_crisis=False),
            path=tmp_path / "freeze.json",
            now=datetime(2026, 8, 23, tzinfo=UTC),
            git_commit="cal",
        )
        obs_path = tmp_path / "cost_observations.jsonl"
        # deterministic but non-degenerate closes: alternating growth so
        # realized variance is strictly positive
        closes = [100.0]
        for i in range(1, 30):
            closes.append(closes[-1] * (1.008 if i % 2 else 1.012))
        candles = [
            {"open_time": T_BASE_MS + i * 86_400_000,
             "open": closes[i - 1] if i else 100.0, "close": closes[i],
             "volume": 1000.0, "quote_volume": closes[i] * 1000.0}
            for i in range(30)
        ]

        quotes_seen = {}

        def fake_quote(sym):
            quotes_seen[sym] = {"bid": closes[-1] - 0.05, "ask": closes[-1] + 0.05}
            return quotes_seen[sym]

        res = run_step(
            manifest,
            lambda sym, src: (candles, None),
            now=datetime(2026, 8, 23, tzinfo=UTC).replace(hour=12),
            log_path=tmp_path / "log.jsonl",
            kwargs_quote_fetcher=fake_quote,
            cost_observation_path=obs_path,
        )
        assert res["status"] == "logged"
        obs = load_observations(obs_path)
        assert len(obs) == 1
        o = obs[0]
        assert o["symbol"] == "AAA"
        assert o["side"] == "BUY"
        assert o["turnover"] == pytest.approx(1.0)
        assert o["bid"] is not None and o["ask"] is not None
        assert o["quoted_spread_bps"] > 0
        assert o["predicted_cost_bps"] == pytest.approx(20.0)
        assert o["realized_vol_annual"] is not None and o["realized_vol_annual"] > 0
        # UNITS: adv30_usd must be the mean of quote_volumes (already USD),
        # never quote_volume x close (which would inflate by ~price)
        expected_adv = sum(c["quote_volume"] for c in candles) / len(candles)
        assert o["adv30_usd"] == pytest.approx(expected_adv, rel=1e-9)


T_BASE_MS = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000)
