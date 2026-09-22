import pytest

from bot.metrics import cagr, max_drawdown, sharpe, volatility


def test_max_drawdown_simple():
    assert max_drawdown([1.0, 2.0, 1.0, 1.5]) == -0.5


def test_max_drawdown_no_drawdown():
    assert max_drawdown([1.0, 1.1, 1.2]) == 0.0


def test_cagr_doubling_in_one_year():
    assert abs(cagr([1.0, 2.0], 365) - 1.0) < 1e-9


def test_cagr_zero_days_is_zero():
    assert cagr([1.0, 2.0], 0) == 0.0


def test_sharpe_zero_for_constant_returns():
    assert sharpe([0.01, 0.01, 0.01], 365) == 0.0


def test_sharpe_positive_for_positive_returns():
    assert sharpe([0.01, 0.02, 0.015, 0.005], 365) > 0


def test_volatility_scales_with_period_count():
    rets = [0.01, -0.02, 0.03, 0.01]
    v1 = volatility(rets, 1)
    v4 = volatility(rets, 4)
    assert abs(v4 - v1 * 2) < 1e-9


def test_total_loss_reports_minus_one_cagr_not_zero():
    assert cagr([1.0, 0.0], 365) == -1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_returns_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        sharpe([0.01, bad, 0.02], 365)
    with pytest.raises(ValueError, match="finite"):
        volatility([0.01, bad], 365)


def test_negative_equity_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        max_drawdown([1.0, -0.1])


@pytest.mark.parametrize("periods", [0, -1])
def test_invalid_periods_per_year_rejected(periods):
    with pytest.raises(ValueError, match="positive integer"):
        sharpe([0.01, 0.02], periods)
