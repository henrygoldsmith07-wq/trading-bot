"""Grid property tests: every candidate strategy must be bounded, deterministic,
and exception-free on a variety of synthetic regimes — including pathological
windows (short history, flat prices, gaps)."""
import pytest

from bot.strategy import build_candidates

CANDIDATES = build_candidates()


def _candles(closes, start_ms=1_500_000_000_000):
    return [
        {"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def up_trend():
    return _candles([100.0 * 1.004 ** i for i in range(400)])


def down_trend():
    return _candles([100.0 * 0.996 ** i for i in range(400)])


def choppy():
    out = []
    px = 100.0
    for i in range(400):
        px *= 1.02 if i % 3 == 0 else (0.985 if i % 3 == 1 else 0.995)
        out.append(px)
    return _candles(out)


def high_vol_uptrend():
    out = []
    px = 100.0
    for i in range(400):
        px *= 1.09 if i % 2 == 0 else 0.965
        out.append(px)
    return _candles(out)


def flat():
    return _candles([100.0] * 400)


def short_history():
    return _candles([100.0, 101.0, 99.5, 100.2])


SERIES = {
    "up": up_trend(),
    "down": down_trend(),
    "choppy": choppy(),
    "highvol": high_vol_uptrend(),
    "flat": flat(),
    "short": short_history(),
}


@pytest.mark.parametrize("series_name", list(SERIES))
@pytest.mark.parametrize("candidate", CANDIDATES, ids=repr)
def test_candidate_weight_bounded_and_deterministic(series_name, candidate):
    candles = SERIES[series_name]
    w1 = candidate.weight(candles)
    w2 = candidate.weight(candles)
    assert 0.0 <= w1 <= 1.0
    assert w1 == w2


@pytest.mark.parametrize("series_name", list(SERIES))
@pytest.mark.parametrize("candidate", CANDIDATES, ids=repr)
def test_candidate_weight_at_partial_windows_safe(series_name, candidate):
    """Every prefix window (including tiny ones) must yield a valid weight."""
    candles = SERIES[series_name]
    for i in (1, 5, 30, 120, len(candles)):
        if 0 < i <= len(candles):
            w = candidate.weight_at(candles, i)
            assert 0.0 <= w <= 1.0


@pytest.mark.parametrize("series_name", list(SERIES))
@pytest.mark.parametrize("candidate", CANDIDATES, ids=repr)
def test_candidate_backtest_runs(series_name, candidate):
    from bot.engine import run_strategy

    res = run_strategy(SERIES[series_name], candidate.weight_at, fee=0.001)
    assert res["final"] > 0.0
    assert len(res["returns"]) == len(SERIES[series_name]) - 1


def test_pool_size_at_least_70():
    assert len(CANDIDATES) >= 70


def test_pool_all_distinct():
    reprs = [repr(c) for c in CANDIDATES]
    assert len(reprs) == len(set(reprs))
