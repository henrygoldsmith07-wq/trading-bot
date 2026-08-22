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
