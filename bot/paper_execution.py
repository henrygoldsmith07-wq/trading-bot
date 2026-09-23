"""Deterministic execution-quality model for paper trading.

This module deliberately contains no broker client and no network order path.
It makes simulated fills less optimistic by modelling spread and market impact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperExecutionModel:
    """Conservative paper fill model using spread plus size-dependent impact."""

    half_spread_bps: float = 2.5
    impact_bps: float = 5.0
    participation_reference: float = 0.01
    max_slippage_bps: float = 100.0

    def __post_init__(self) -> None:
        values = (
            self.half_spread_bps,
            self.impact_bps,
            self.participation_reference,
            self.max_slippage_bps,
        )
        if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in values):
            raise ValueError("execution model parameters must be finite and non-negative")
        if self.participation_reference <= 0.0:
            raise ValueError("participation_reference must be positive")

    def fill_price(
        self,
        mid_price: float,
        side: str,
        order_notional: float,
        recent_quote_notional: float | None = None,
    ) -> float:
        """Return an adverse simulated fill price; malformed inputs fail closed."""
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if not math.isfinite(float(mid_price)) or float(mid_price) <= 0.0:
            raise ValueError("mid_price must be positive and finite")
        if not math.isfinite(float(order_notional)) or float(order_notional) < 0.0:
            raise ValueError("order_notional must be finite and non-negative")

        participation = 0.0
        if recent_quote_notional is not None:
            if (
                not math.isfinite(float(recent_quote_notional))
                or float(recent_quote_notional) <= 0.0
            ):
                raise ValueError("recent_quote_notional must be positive and finite")
            participation = float(order_notional) / float(recent_quote_notional)

        impact = self.impact_bps * math.sqrt(
            max(0.0, participation / self.participation_reference)
        )
        slippage_bps = min(
            self.max_slippage_bps,
            self.half_spread_bps + impact,
        )
        adverse = slippage_bps / 10_000.0
        multiplier = 1.0 + adverse if side == "BUY" else 1.0 - adverse
        fill = float(mid_price) * multiplier
        if not math.isfinite(fill) or fill <= 0.0:
            raise ValueError("execution model produced an invalid fill price")
        return fill
