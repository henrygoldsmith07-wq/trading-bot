"""Transaction-cost models beyond flat bps.

The engine's default cost is a flat rate per unit of turnover
(`fee + spread_bps + slippage_bps`). These models refine that:

- Vol-dependent spread & slippage: real spreads widen when realized
  volatility rises; both are scaled by (trailing_vol / reference_vol)^0.5,
  clamped to `[floor_mult, cap_mult]`.
- Square-root market impact (Almgren-style approximation): trading a
  fraction `u` of average daily volume costs about
  `k * daily_vol * sqrt(u)` in fractional terms. The portfolio's size
  enters through `adv_to_equity` = (avg daily quote volume / account equity):
  a $10k account trading BTC has essentially no impact; the same account
  in an illiquid alt has some; a fund has more everywhere.
- Tiered fees (maker/taker by 30d volume) for the live paper broker, which
  tracks notional and can therefore charge account-level fees.
- Liquidity filter: skip assets whose median daily quote volume cannot
  support the intended turnover without persistent impact.
"""
from __future__ import annotations

import math

DEFAULT_VOL_FLOOR = 0.5
DEFAULT_VOL_CAP = 3.0


def realized_vol_series(closes: list[float], window: int = 20, periods_per_year: int = 365) -> list[float]:
    """Annualized trailing volatility ending at each index (index i uses
    closes up to i-1, matching the engine's decision timing). Values before
    enough history are NaN so callers can fall back to the flat model."""
    n = len(closes)
    out = [float("nan")] * n
    pr = [0.0] * (n + 1)
    pr2 = [0.0] * (n + 1)
    for j in range(1, n):
        r = math.log(closes[j] / closes[j - 1])
        pr[j + 1] = pr[j] + r
        pr2[j + 1] = pr2[j] + r * r
    for i in range(window + 1, n + 1):
        s = pr[i] - pr[i - window]
        s2 = pr2[i] - pr2[i - window]
        var = (s2 - s * s / window) / (window - 1)
        out[i - 1] = math.sqrt(max(var, 0.0) * periods_per_year)
    return out


def vol_cost_multiplier(
    trailing_ann_vol: float,
    reference_vol: float = 0.60,
    floor_mult: float = DEFAULT_VOL_FLOOR,
    cap_mult: float = DEFAULT_VOL_CAP,
) -> float:
    """Cost scaling factor from realized volatility (sqrt law, clamped)."""
    if not math.isfinite(trailing_ann_vol) or trailing_ann_vol <= 0 or reference_vol <= 0:
        return 1.0
    m = math.sqrt(trailing_ann_vol / reference_vol)
    return max(floor_mult, min(cap_mult, m))


def square_root_impact_fraction(
    turnover_fraction: float,
    daily_vol: float,
    adv_to_equity: float,
    impact_k: float = 1.0,
) -> float:
    """Fractional one-sided impact cost for this bar's turnover.

    `turnover_fraction` is |Δweight| of equity; `daily_vol` is the day's
    (not annualized) volatility as a fraction; `adv_to_equity` is average
    daily quote volume divided by account equity, so participation is
    turnover / adv_to_equity. Zero when inputs degenerate or the vol
    estimate is not yet available (NaN warmup).
    """
    if adv_to_equity <= 0 or turnover_fraction <= 0 or daily_vol <= 0:
        return 0.0
    if not (math.isfinite(turnover_fraction) and math.isfinite(daily_vol)):
        return 0.0  # insufficient history for a vol estimate: charge no impact
    participation = turnover_fraction / adv_to_equity
    return impact_k * daily_vol * math.sqrt(participation)


def tiered_taker_fee(
    quote_volume_30d: float,
    tiers: list[tuple[float, float]] | None = None,
) -> float:
    """Taker fee fraction for a 30-day quote-volume tier.

    `tiers` is [(volume_floor_usd, taker_fee)] sorted ascending; the first
    tier whose floor is met wins. Defaults mirror a typical retail schedule.
    """
    if tiers is None:
        tiers = [(0.0, 0.0010), (50_000.0, 0.0009), (100_000.0, 0.0008), (250_000.0, 0.0007), (1_000_000.0, 0.0005)]
    rate = tiers[0][1]
    for floor, fee in tiers:
        if quote_volume_30d >= floor:
            rate = fee
    return rate


def median_daily_quote_volume(candles: list[dict], lookback: int = 30) -> float | None:
    """Median daily notional traded (quote currency) over the last `lookback`
    candles. Prefers Binance's exact `quote_volume` field; falls back to
    base-volume x close when only base units are present. None when the feed
    carries neither (Yahoo candles may omit both), so callers can distinguish
    'illiquid' from 'unknown'."""
    vols = []
    for c in candles[-lookback:]:
        qv = c.get("quote_volume")
        if qv is not None:
            vols.append(abs(float(qv)))
            continue
        v = c.get("volume")
        close = c.get("close")
        if v is None or close is None:
            return None
        vols.append(abs(float(v) * float(close)))
    if not vols:
        return None
    vols.sort()
    mid = len(vols) // 2
    if len(vols) % 2:
        return vols[mid]
    return (vols[mid - 1] + vols[mid]) / 2.0


def passes_liquidity_filter(
    candles: list[dict],
    min_median_quote_volume: float = 5_000_000.0,
    lookback: int = 30,
) -> tuple[bool, str]:
    """(passes, reason). Assets with no volume data pass vacuously (cannot
    judge); assets below the floor are rejected with the measured value."""
    med = median_daily_quote_volume(candles, lookback)
    if med is None:
        return True, "no volume data (filter not applicable)"
    if med < min_median_quote_volume:
        return False, f"median daily volume ${med:,.0f} < ${min_median_quote_volume:,.0f}"
    return True, f"median daily volume ${med:,.0f}"


class CostParams:
    """Bundle of extended-cost parameters accepted by `engine.run_strategy`.

    Everything defaults to the historical flat model, so existing results
    are unchanged unless explicitly opted in.
    """

    __slots__ = (
        "spread_vol_scale",
        "slippage_vol_scale",
        "vol_window",
        "reference_vol",
        "vol_floor_mult",
        "vol_cap_mult",
        "impact_k",
        "adv_to_equity",
    )

    def __init__(
        self,
        spread_vol_scale: float = 0.0,
        slippage_vol_scale: float = 0.0,
        vol_window: int = 20,
        reference_vol: float = 0.60,
        vol_floor_mult: float = DEFAULT_VOL_FLOOR,
        vol_cap_mult: float = DEFAULT_VOL_CAP,
        impact_k: float = 0.0,
        adv_to_equity: float | None = None,
    ):
        self.spread_vol_scale = spread_vol_scale
        self.slippage_vol_scale = slippage_vol_scale
        self.vol_window = vol_window
        self.reference_vol = reference_vol
        self.vol_floor_mult = vol_floor_mult
        self.vol_cap_mult = vol_cap_mult
        self.impact_k = impact_k
        self.adv_to_equity = adv_to_equity

    def active(self) -> bool:
        return (
            self.spread_vol_scale > 0
            or self.slippage_vol_scale > 0
            or (self.impact_k > 0 and self.adv_to_equity is not None)
        )

    def to_kwargs(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}
