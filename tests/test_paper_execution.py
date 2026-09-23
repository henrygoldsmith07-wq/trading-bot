import math

import pytest

from bot.paper_execution import PaperExecutionModel


def test_buy_fill_is_adverse_to_mid():
    model = PaperExecutionModel()
    assert model.fill_price(100.0, "BUY", 100.0) > 100.0


def test_sell_fill_is_adverse_to_mid():
    model = PaperExecutionModel()
    assert model.fill_price(100.0, "SELL", 100.0) < 100.0


def test_larger_participation_has_more_market_impact():
    model = PaperExecutionModel()
    small = model.fill_price(100.0, "BUY", 100.0, 100_000.0)
    large = model.fill_price(100.0, "BUY", 10_000.0, 100_000.0)
    assert large > small


@pytest.mark.parametrize(
    "price,notional,quote",
    [
        (math.nan, 100.0, None),
        (0.0, 100.0, None),
        (100.0, math.inf, None),
        (100.0, -1.0, None),
        (100.0, 100.0, 0.0),
        (100.0, 100.0, math.nan),
    ],
)
def test_malformed_execution_inputs_fail_closed(price, notional, quote):
    model = PaperExecutionModel()
    with pytest.raises(ValueError):
        model.fill_price(price, "BUY", notional, quote)


def test_invalid_side_fails_closed():
    with pytest.raises(ValueError):
        PaperExecutionModel().fill_price(100.0, "HOLD", 100.0)


def test_slippage_is_capped():
    model = PaperExecutionModel(max_slippage_bps=25.0)
    fill = model.fill_price(100.0, "BUY", 1_000_000.0, 1.0)
    assert fill == pytest.approx(100.25)
