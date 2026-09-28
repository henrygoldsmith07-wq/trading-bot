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

EXACT WEALTH ACCOUNTING (not an additive approximation)
------------------------------------------------------
The session return is built by evolving actual wealth through the session, not
by adding per-leg simple returns. The previous implementation returned

    previous_weight * (open/prev_close - 1) + target_weight * (close/open - 1)

which is a first-order approximation. Across a 5% overnight gap it drops the
cross term w_prev*w_tgt*gap*intraday, systematically UNDERSTATING the return
for any position held through both legs:

    prev_close=100, open=105, close=110, held 1.0, zero fees
    exact close-to-close      = 110/100 - 1        = +10.0000%
    previous additive result  = 0.05 + 0.047619    = +9.7619%   <-- wrong

`calculate_transition` now performs the full self-financing sequence:

    1. wealth_open_from_previous_close = 1.0            (unit wealth, scaled)
    2. overnight P&L on the PREVIOUS position
    3. opening portfolio wealth  W_open
    4. opening asset / cash split at the execution price
    5. rebalance to the target weight AT the execution price
    6. turnover measured on the DRIFTED opening allocation
    7. costs charged against post-trade wealth (they are wealth, not returns)
    8. post-trade position / cash
    9. intraday P&L on the NEW position
   10. closing wealth  W_close, and the exact session return W_close/W_start - 1

The unit of account is wealth starting at 1.0, so the returned `return` is the
exact multiplicative session return. `overnight` / `intraday` / `cost` / `cash`
remain available as the *wealth-denominated contributions* to that return, so
the decomposition still adds up — but as a sum of wealth deltas divided by the
starting wealth, which is what "adding up" was always supposed to mean.
"""
from __future__ import annotations

import math


def open_of(candle: dict, fallback: float) -> float:
    """Return a valid open print, falling back only to a valid prior mark."""
    if not math.isfinite(float(fallback)) or float(fallback) <= 0.0:
        raise ValueError("fallback open price must be positive and finite")
    o = candle.get("open")
    if o is None:
        return float(fallback)
    try:
        value = float(o)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(value) or value <= 0.0:
        return float(fallback)
    return value


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
    for name, position in (("previous_position", previous_position), ("target_position", target_position)):
        if not math.isfinite(float(position)) or not 0.0 <= float(position) <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1] (got {position})")
    for name, px in (("previous_close", previous_close), ("execution_price", execution_price), ("closing_price", closing_price)):
        if not math.isfinite(float(px)) or float(px) <= 0.0:
            raise ValueError(f"{name} must be positive and finite (got {px})")
    if not math.isfinite(float(costs)) or float(costs) < 0.0:
        raise ValueError("costs must be finite and non-negative")
    if not math.isfinite(float(cash_rate_period)) or float(cash_rate_period) <= -1.0:
        raise ValueError("cash_rate_period must be finite and greater than -1")
    if cash_basis not in ("previous", "target"):
        raise ValueError("cash_basis must be 'previous' or 'target'")

    w_prev = float(previous_position)
    w_target = float(target_position)
    p_close = float(previous_close)
    p_exec = float(execution_price)
    p_close_end = float(closing_price)
    cost_rate = float(costs)
    cash_rate = float(cash_rate_period)

    # (1) Unit wealth carried in from the previous session's close. Working in
    # units of starting wealth is what makes every later quantity a true
    # fraction of the portfolio rather than an approximation of one.
    wealth_start = 1.0

    # (2) Overnight P&L on the position actually held overnight, and (3) the
    # opening portfolio wealth. Cash is the uninvested remainder; under the
    # "previous" basis it accrued the idle yield overnight (it was held
    # through the gap), under "target" the idle fraction follows the new
    # allocation instead and accrues nothing here.
    asset_value_open = w_prev * (p_exec / p_close)
    cash_fraction = 1.0 - w_prev
    cash_value_open = cash_fraction * (1.0 + (cash_rate if cash_basis == "previous" else 0.0))
    wealth_open = asset_value_open + cash_value_open
    # Wealth-denominated contributions, so overnight + cost + intraday + cash
    # reproduces `return` exactly instead of approximately.
    overnight = asset_value_open - w_prev
    cash_leg = cash_value_open - cash_fraction

    # (4) Opening allocation DRIFTED to the execution price. A position sized
    # at yesterday's close is no longer that weight after a gap, so turnover
    # and the rebalance must be measured against the drifted weight. Using
    # the undrifted target-previous difference mis-charges cost in proportion
    # to the gap.
    drifted_weight = asset_value_open / wealth_open

    # (5) Rebalance at the execution price to the target weight, and (6) the
    # turnover that requires — the trade is the change in drifted weight.
    turnover = abs(w_target - drifted_weight)
    traded_notional = turnover * wealth_open

    # (7) Costs are paid out of wealth at the execution price: selling frees
    # cash, buying consumes it, and either way the fee reduces the account.
    # Charging it against opening wealth (rather than subtracting it from a
    # return) is what keeps the next steps exactly self-financing.
    wealth_after_cost = wealth_open - cost_rate * traded_notional
    # `cost` stays a POSITIVE magnitude (fee charged) to match its long-standing
    # meaning for callers; it is subtracted from the wealth walk above.
    cost = cost_rate * traded_notional

    # (8) Post-trade position/cash at the execution price, now expressed as a
    # fraction of the post-cost wealth so the remaining legs compound on the
    # wealth that actually survived the fee.
    if wealth_after_cost <= 0.0 or not math.isfinite(wealth_after_cost):
        raise ValueError("costs and turnover exhausted portfolio wealth")
    asset_post = w_target * wealth_after_cost
    cash_post = (1.0 - w_target) * wealth_after_cost

    # (9) Intraday P&L on the new position, plus cash accrual on whichever
    # basis was requested.
    asset_value_close = asset_post * (p_close_end / p_exec)
    cash_value_close = cash_post * (1.0 + (cash_rate if cash_basis == "target" else 0.0))
    wealth_close = asset_value_close + cash_value_close
    intraday = (asset_value_close - asset_post) + (cash_value_close - cash_post)
    if cash_basis == "target":
        cash_leg += cash_value_close - cash_post

    # (10) The exact session return: closing wealth over opening wealth.
    total = wealth_close / wealth_start - 1.0

    return {
        "return": total,
        "overnight": overnight,
        "intraday": intraday,
        "turnover": turnover,
        "cost": cost,
        "cash": cash_leg,
        # Full wealth walk — lets callers audit the accounting rather than
        # trust the scalar. Kept flat (no nesting) for JSONL round-tripping.
        "wealth_start": wealth_start,
        "wealth_open": wealth_open,
        "wealth_after_cost": wealth_after_cost,
        "wealth_close": wealth_close,
        "drifted_weight": drifted_weight,
        "asset_value_open": asset_value_open,
        "asset_value_close": asset_value_close,
    }
