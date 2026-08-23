"""Position transition accounting — ONE implementation for backtest and forward.

`calculate_transition` is the only place in the repo that converts a weight
change plus three prices into a period return. The backtest engine calls it
per bar; the prospective paper runner calls it per live day; the parity test
suite proves the two produce identical numbers. If this function is wrong,
both are wrong together — which is exactly the point.

next_open semantics (the realistic convention):
    previous close ──overnight──▶ execution price   (OLD position earns this)
    execution price ──intraday──▶ closing price     (NEW position earns this)
Close-mode (the optimistic baseline) is the degenerate case where execution
happens AT the previous close: zero-length overnight, the whole move accrues
to the new position.
"""
from __future__ import annotations


def open_of(candle: dict, fallback: float) -> float:
    """The candle's open print, or `fallback` when the feed omits/invalidates
    it (some Yahoo rows carry no open). Same rule the engine has always used."""
    o = candle.get("open")
    if o is None or o <= 0:
        return fallback
    return o


def calculate_transition(
    previous_position: float,
    target_position: float,
    previous_close: float,
    execution_price: float,
    closing_price: float,
    costs: float = 0.0,
    cash_rate_period: float = 0.0,
    cash_basis: str = "previous",
) -> dict:
    """One rebalance day's return from prices and positions.

    previous_position : weight held coming into the day (decided yesterday)
    target_position   : weight after today's trade (band already applied)
    previous_close    : decision-time close (yesterday's close)
    execution_price   : the price the trade fills at (today's open for
                        next_open mode; previous_close itself for close mode)
    closing_price     : the price at the END of the accounting window (the
                        bar close in backtests, the latest live print in the
                        forward runner's snapshot)
    costs             : fractional cost per unit of turnover
    cash_rate_period  : cash yield accrued over THIS period on the uninvested
                        fraction (`cash_basis` picks whether that fraction
                        follows the previous or target position; the engine
                        uses "previous" for next_open and "target" for close)

    Returns {return, overnight, intraday, turnover, cost, cash} so callers
    can log the decomposition, not just the total.
    """
    for name, px in (("previous_close", previous_close), ("execution_price", execution_price), ("closing_price", closing_price)):
        if px <= 0:
            raise ValueError(f"{name} must be positive (got {px})")
    if cash_basis not in ("previous", "target"):
        raise ValueError("cash_basis must be 'previous' or 'target'")
    overnight = previous_position * (execution_price / previous_close - 1.0)
    intraday = target_position * (closing_price / execution_price - 1.0)
    turnover = abs(target_position - previous_position)
    cost = costs * turnover
    held = previous_position if cash_basis == "previous" else target_position
    cash = (1.0 - held) * cash_rate_period
    total = overnight + intraday + cash - cost
    return {
        "return": total,
        "overnight": overnight,
        "intraday": intraday,
        "turnover": turnover,
        "cost": cost,
        "cash": cash,
    }
