import pytest

from bot.strategy import BuyHold, TrendVol
from bot.walkforward import (
    _fold_boundaries,
    combine_portfolio,
    combine_portfolio_invvol,
    walk_forward,
)


def _candles(closes, start_ms=0):
    return [
        {"open_time": start_ms + i * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def test_fold_boundaries_contiguous_non_overlapping():
    candles = _candles([100.0] * 800)
    folds = _fold_boundaries(candles, train_days=365, test_days=182)
    assert len(folds) >= 1
    for i, (start, train_end, test_end) in enumerate(folds):
        assert start == 0  # expanding window always trains from day 0
        assert start < train_end < test_end
        assert train_end - start >= 360  # ~a year of daily rows
        if i > 0:
            prev = folds[i - 1]
            assert train_end == prev[2]  # test windows tile the timeline with no gaps


def test_walk_forward_runs_and_reports():
    # synthetic trending market: the bot should end above 1.0
    closes = [100.0 * (1.003 ** i) for i in range(1200)]
    candles = _candles(closes)
    res = walk_forward(candles, candidates=[BuyHold(), TrendVol(100, 20, 0.4)], train_days=365, test_days=180)
    assert res["n_folds"] >= 1
    assert res["equity"] > 1.0
    assert res["cagr"] > 0.0
    assert len(res["folds"]) == res["n_folds"]
    for f in res["folds"]:
        assert "strategy" in f


def test_walk_forward_insufficient_data_raises():
    with pytest.raises(ValueError):
        walk_forward(_candles([1.0, 2.0, 3.0]), train_days=365, test_days=365)


def test_inverse_vol_combiner_enforces_hard_cap_after_redistribution():
    timeline = list(range(40))
    streams = {
        "LOW": {t: (0.00001 if t % 2 else -0.00001) for t in timeline},
        "MID": {t: (0.01 if t % 2 else -0.01) for t in timeline},
        "HIGH": {t: (0.03 if t % 2 else -0.03) for t in timeline},
    }
    # With three assets and max_multiple=1.2, no normalized sleeve may exceed
    # 40%. The old one-pass redistribution could breach this after clipping.
    out = combine_portfolio_invvol(
        streams,
        timeline,
        n_assets=3,
        window=10,
        max_multiple_of_equal=1.2,
    )
    assert len(out) == len(timeline)
    assert all(abs(r) <= 0.03 + 1e-12 for r in out)


def test_equal_weight_combiner_rejects_denominator_smaller_than_present_assets():
    timeline = [1]
    streams = {"A": {1: 0.01}, "B": {1: 0.02}}
    with pytest.raises(ValueError, match="exceeds denominator"):
        combine_portfolio(streams, timeline, n_assets=2, denominator_by_day={1: 1})


def test_inverse_vol_combiner_rejects_denominator_smaller_than_present_assets():
    timeline = list(range(25))
    streams = {
        "A": {t: 0.001 for t in timeline},
        "B": {t: -0.001 for t in timeline},
    }
    with pytest.raises(ValueError, match="exceeds denominator"):
        combine_portfolio_invvol(
            streams,
            timeline,
            n_assets=2,
            window=5,
            denominator_by_day={t: 1 for t in timeline},
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_assets": 0}, "positive integer"),
        ({"n_assets": 2, "window": 1}, "window"),
        ({"n_assets": 2, "max_multiple_of_equal": 0.5}, "max_multiple_of_equal"),
    ],
)
def test_inverse_vol_combiner_rejects_invalid_configuration(kwargs, match):
    timeline = [1, 2]
    streams = {"A": {1: 0.01, 2: 0.01}, "B": {1: 0.02, 2: 0.02}}
    with pytest.raises(ValueError, match=match):
        combine_portfolio_invvol(streams, timeline, **kwargs)
