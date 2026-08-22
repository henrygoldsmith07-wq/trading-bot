from bot.walkforward import _fold_boundaries, walk_forward
from bot.strategy import BuyHold, TrendVol


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
    import pytest

    with pytest.raises(ValueError):
        walk_forward(_candles([1.0, 2.0, 3.0]), train_days=365, test_days=365)
