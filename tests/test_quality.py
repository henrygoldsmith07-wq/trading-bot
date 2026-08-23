
import pytest

from bot.data import clean_candles, is_stale
from bot.engine import run_strategy
from bot.metrics import sharpe
from bot.regimes import label_regimes, segment, segment_metrics, stress_mask
from bot.walkforward import combine_portfolio


def _candles(rows, start_ms=1_600_000_000_000):
    """rows: list of (open, close) or just close floats."""
    out = []
    for i, row in enumerate(rows):
        if isinstance(row, (tuple, list)):
            o, c = row
        else:
            o = c = row
        out.append({"open_time": start_ms + i * 86_400_000, "open": o, "close": c})
    return out


# ---------- engine: execution realism ----------

def test_next_open_splits_overnight_and_intraday():
    # day 1: overnight gap +2% on old (zero) position, intraday +10% on new full position
    candles = _candles([(100, 100), (102, 112.2)])
    res = run_strategy(candles, lambda c, i: 1.0, fee=0.0, execution="next_open")
    assert res["returns"][0] == pytest.approx(0.10)


def test_close_execution_ignores_open():
    candles = _candles([(100, 100), (50, 110)])
    res = run_strategy(candles, lambda c, i: 1.0, fee=0.0, execution="close")
    assert res["returns"][0] == pytest.approx(0.10)


def test_latency_delays_signal():
    calls = []

    def w(candles, i):
        calls.append(i)
        return 0.0 if i <= 5 else 1.0

    candles = _candles([100.0] * 12)
    res = run_strategy(candles, w, fee=0.0, latency_days=2)
    # the signal for day i is what the strategy said at day i-2
    assert res["weights"][6] == pytest.approx(0.0)  # signal at day 4
    assert res["weights"][8] == pytest.approx(1.0)  # signal at day 6
    assert res["weights"][:2] == [0.0, 0.0]


def test_risk_free_accrues_on_idle_cash():
    candles = _candles([100.0] * 10)
    res = run_strategy(candles, lambda c, i: 0.0, fee=0.0, risk_free_annual=0.10)
    expected_daily = 0.10 / 365
    for r in res["returns"]:
        assert r == pytest.approx(expected_daily)


def test_spread_and_slippage_increase_cost():
    candles = _candles([100.0] * 6)
    base = run_strategy(candles, lambda c, i: 1.0 if i == 3 else 0.0, fee=0.001)
    extra = run_strategy(candles, lambda c, i: 1.0 if i == 3 else 0.0, fee=0.001, spread_bps=10, slippage_bps=10)
    assert extra["final"] < base["final"]
    # one 0->1->0 round trip = 2 units of turnover at 20 extra bps
    assert base["final"] - extra["final"] == pytest.approx(2 * 0.0020, rel=0.05)


def test_invalid_execution_mode_rejected():
    with pytest.raises(ValueError):
        run_strategy(_candles([1.0, 2.0]), lambda c, i: 1.0, execution="vwap")


def test_open_missing_falls_back_to_previous_close():
    candles = [
        {"open_time": 0, "close": 100.0},
        {"open_time": 86_400_000, "close": 110.0},  # no open field
    ]
    res = run_strategy(candles, lambda c, i: 1.0, fee=0.0, execution="next_open")
    assert res["returns"][0] == pytest.approx(0.10)


# ---------- data hygiene ----------

def test_clean_candles_dedupes_drops_and_sorts():
    raw = [
        {"open_time": 3, "close": 103.0, "volume": 5.0},
        {"open_time": 1, "close": 101.0, "volume": 1.0},
        {"open_time": 1, "close": 101.5, "volume": 2.0},  # higher-volume dup wins
        {"open_time": 2, "close": float("nan")},
        {"open_time": 2, "close": 0.0},  # non-positive
        {"open_time": 4, "close": 104.0, "volume": 0.0},
    ]
    cleaned = clean_candles(raw)
    assert [(c["open_time"], c["close"]) for c in cleaned] == [(1, 101.5), (3, 103.0), (4, 104.0)]


def test_is_stale_detects_dead_history():
    now = 100 * 86_400_000
    fresh = [{"open_time": now - 86_400_000, "close": 1.0}]
    dead = [{"open_time": now - 90 * 86_400_000, "close": 1.0}]
    assert not is_stale(fresh, now_ms=now)
    assert is_stale(dead, now_ms=now)
    assert is_stale([], now_ms=now)


# ---------- portfolio: survivorship control ----------

def test_combine_portfolio_fixed_denominator():
    a = {0: 0.10, 86_400_000: 0.10}
    b = {0: 0.30}  # missing day 2: sits in cash, does not hand capital to A
    timeline = [0, 86_400_000]
    out = combine_portfolio({"A": a, "B": b}, timeline, n_assets=2)
    assert out[0] == pytest.approx(0.20)
    assert out[1] == pytest.approx(0.05)  # (0.10 + 0.0) / 2


# ---------- metrics: excess Sharpe ----------

def test_sharpe_excess_return_sign():
    base = [0.0002, 0.0003, 0.0001, 0.0002]  # ~7.6% annualized
    rets = base * 12
    assert sharpe(rets, 365, risk_free_annual=0.10) < 0  # below cash
    assert sharpe(rets, 365, risk_free_annual=0.01) > 0  # above cash


# ---------- regimes ----------

def _btc_regime_candles():
    # 400 flat days, then a strong 150-day rally, then a steep 100-day crash
    closes = [100.0] * 400
    for _ in range(150):
        closes.append(closes[-1] * 1.005)
    for _ in range(100):
        closes.append(closes[-1] * 0.985)
    return [
        {"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def test_label_regimes_bull_bear_sideways():
    btc = _btc_regime_candles()
    timeline = [c["open_time"] for c in btc]
    labels = label_regimes(btc, timeline)
    assert labels[timeline[100]] == "sideways"
    assert labels[timeline[500]] == "bull"
    assert labels[timeline[-1]] == "bear"


def test_segment_groups_consecutive_labels():
    d = 86_400_000
    labels = {0: "bull", d: "bull", 2 * d: "bear", 3 * d: "bear", 4 * d: "bull"}
    segs = segment(labels, [0, d, 2 * d, 3 * d, 4 * d], min_days=0)
    assert [(s["label"], s["start"], s["end"]) for s in segs] == [
        ("bull", 0, d),
        ("bear", 2 * d, 3 * d),
        ("bull", 4 * d, 4 * d),
    ]


def test_segment_min_length_filters_noise():
    d = 86_400_000
    labels = {0: "bull", d: "bear", 2 * d: "bear", 3 * d: "bear"}
    segs = segment(labels, [0, d, 2 * d, 3 * d], min_days=2)
    assert [s["label"] for s in segs] == ["bear"]


def test_stress_mask_flags_crash_window():
    btc = _btc_regime_candles()
    timeline = [c["open_time"] for c in btc]
    mask = stress_mask(btc, timeline)
    assert any(mask.values())
    # the crash tail must be stressed
    assert mask[timeline[-1]]


def test_segment_metrics_on_known_returns():
    m = segment_metrics({0: 0.1, 1: 0.1, 2: 0.1}, [0, 1, 2], 0, 2)
    assert m["final"] == pytest.approx(1.331)  # all three days fall in [0, 2]
    assert m["cagr"] > 1.0
    assert m["max_drawdown"] == 0.0


# ---------- sensitivity ----------

def test_cost_sweep_penalizes_higher_costs():
    from bot.sensitivity import cost_sweep
    from bot.strategy import TrendVol

    closes = [100.0 * (1.003 ** max(0, i - 300)) if i > 300 else 100.0 for i in range(1200)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    from bot.walkforward import absolute_folds

    folds = absolute_folds(candles, train_days=365, test_days=180)
    sweep = cost_sweep(candles, folds, total_bps=(5, 100), candidates=[TrendVol(50, 20, 0.4)])
    assert sweep[5]["final"] >= sweep[100]["final"]


def test_latency_sweep_runs():
    from bot.sensitivity import latency_sweep
    from bot.strategy import TrendVol
    from bot.walkforward import absolute_folds

    closes = [100.0 * (1.003 ** max(0, i - 300)) if i > 300 else 100.0 for i in range(1200)]
    candles = [{"open_time": i * 86_400_000, "close": c} for i, c in enumerate(closes)]
    folds = absolute_folds(candles, train_days=365, test_days=180)
    out = latency_sweep(candles, folds, latencies=(0, 1), candidates=[TrendVol(50, 20, 0.4)])
    assert set(out) == {0, 1}
