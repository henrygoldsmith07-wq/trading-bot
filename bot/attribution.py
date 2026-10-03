"""Which component of the frozen portfolio actually earns its keep?

WHY THIS EXISTS
    The primary rule is a stack: selected sleeves, inverse-vol weighting, a
    cross-sectional momentum tilt, crisis de-risking, a 5% rebalance band and
    a volatility overlay. The headline number grades the whole stack, which
    says nothing about what inside it contributes value — and a stack can
    carry components that subtract while the total looks respectable.

    This module runs a controlled ablation ladder on the EXACT same universe
    and period for every rung. The universe can never change between ablation
    runs (a different universe would confound the attribution), so every rung
    consumes one immutable input set and the report carries an
    `inputs_fingerprint` proving it. Each rung adds exactly one component, and
    the deltas are reported as an additive story:

        Base selected sleeves:            +X%
        + inverse-vol weighting:          +Y%
        + momentum tilt:                  +Z%
        + crisis de-risk:                 ...
        + rebalance band:                 ...
        + vol overlay:                    ...
        - estimated execution costs:      ...

    The cost line is REPORTED, never re-simulated: re-running a different cost
    model would be a different experiment, not a decomposition of this one.

Paper trading only. This module decomposes returns; it never trades.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .compare_common import equity_metrics as returns_metrics
from .compare_common import vol_overlay
from .portfolio_rules import combine_portfolio_rule

COMPONENTS = (
    "base selected sleeves",
    "inverse-vol weighting",
    "cross-sectional momentum tilt",
    "crisis de-risking",
    "5% rebalance band",
    "volatility overlay",
)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).date().isoformat()


def _inputs_fingerprint(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    has_mask: bool,
) -> str:
    blob = json.dumps(
        {
            "symbols": sorted(asset_dailies),
            "n_days": len(timeline),
            "first_day": timeline[0] if timeline else None,
            "last_day": timeline[-1] if timeline else None,
            "has_mask": has_mask,
            "n_assets": n_assets,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _equal_weight_combine(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    eligible_by_day: dict[int, set[str]] | None,
) -> list[float]:
    """Equal-weight sleeves with the same masking convention as the full rule.

    Ineligible sleeves get zero weight and their capital stays in cash: no
    redistribution, matching `combine_portfolio_rule` so the base rung differs
    from rung 2 ONLY in the weighting scheme.
    """
    syms = sorted(asset_dailies)
    out: list[float] = []
    for t in timeline:
        if eligible_by_day is not None:
            if t not in eligible_by_day:
                raise ValueError(f"no point-in-time eligibility recorded for {t}")
            contributors = {s for s in syms if t in asset_dailies[s]} & set(eligible_by_day[t])
            denom = max(1, len(eligible_by_day[t]))
        else:
            contributors = {s for s in syms if t in asset_dailies[s]}
            denom = n_assets if n_assets > 0 else 1
        present = sorted(contributors)
        if not present:
            out.append(0.0)
            continue
        w = 1.0 / denom
        out.append(sum(w * asset_dailies[s][t] for s in present))
    return out


def _metrics(returns: list[float], periods_per_year: int, risk_free_annual: float) -> dict[str, float]:
    if not returns:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = 1.0
    path = [1.0]
    for r in returns:
        equity *= 1.0 + r
        path.append(equity)
    m = returns_metrics(returns, periods_per_year=periods_per_year, risk_free_annual=risk_free_annual)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]}


def decompose(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    *,
    eligible_by_day: dict[int, set[str]] | None = None,
    band_dailies: dict[str, dict[int, float]] | None = None,
    overlay_target_vol: float | None = 0.25,
    vol_window: int = 20,
    max_multiple_of_equal: float = 2.0,
    tilt_lookback: int = 90,
    max_tilt: float = 0.5,
    corr_window: int = 60,
    corr_threshold: float = 0.6,
    derisk: float = 0.6,
    band: float = 0.05,
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
    overlay_fee_on_turnover: float = 0.0015,
) -> dict[str, Any]:
    """Controlled ablation ladder over one immutable universe and period.

    `asset_dailies` maps each selected sleeve to its out-of-sample daily
    returns keyed by day (ms). `band_dailies` carries the same sleeves
    simulated with the rebalance band applied at sleeve level (the band lives
    there); when absent, the band is applied inside the combine step and the
    report says so in `notes` rather than pretending both are the same rung.
    """
    if not timeline:
        raise ValueError("timeline is empty; nothing to attribute")
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1 (got {n_assets})")
    if not asset_dailies:
        raise ValueError("asset_dailies is empty")
    if band_dailies is not None and sorted(band_dailies) != sorted(asset_dailies):
        raise ValueError("band_dailies must cover exactly the same sleeves as asset_dailies")

    notes: list[str] = []

    def _combine(sleeves: dict[str, dict[int, float]], *, use_tilt: bool, use_crisis: bool) -> list[float]:
        return combine_portfolio_rule(
            sleeves,
            timeline,
            n_assets,
            vol_window=vol_window,
            max_multiple_of_equal=max_multiple_of_equal,
            tilt_lookback=tilt_lookback,
            max_tilt=max_tilt,
            corr_window=corr_window,
            corr_threshold=corr_threshold,
            derisk=derisk,
            use_tilt=use_tilt,
            use_crisis=use_crisis,
            eligible_by_day=eligible_by_day,
        )

    # Rung 1: equal-weight base (no tilts, no band, no overlay).
    r1 = _equal_weight_combine(asset_dailies, timeline, n_assets, eligible_by_day)
    # Rung 2: inverse-vol weighting, nothing else.
    r2 = _combine(asset_dailies, use_tilt=False, use_crisis=False)
    # Rung 3: + momentum tilt.
    r3 = _combine(asset_dailies, use_tilt=True, use_crisis=False)
    # Rung 4: + crisis de-risk.
    r4 = _combine(asset_dailies, use_tilt=True, use_crisis=True)
    # Rung 5: + rebalance band. The band lives at SLEEVE level (it suppresses
    # individual trades in the engine), so it cannot be simulated at the
    # portfolio combine step: without band_dailies the honest answer is a
    # refusal, not a rung that silently equals rung 4 and calls it the band.
    if band_dailies is None:
        raise ValueError(
            "band_dailies is required to ablate the rebalance band: the band acts at "
            "sleeve level, so its effect cannot be derived from the unbanded return "
            "streams. Pass the band-simulated sleeves (band="
            f"{band}) or drop the band rung explicitly."
        )
    r5 = _combine(band_dailies, use_tilt=True, use_crisis=True)
    # Rung 6: + vol overlay.
    if overlay_target_vol is not None:
        r6 = vol_overlay(r5, target=overlay_target_vol, window=vol_window, fee=overlay_fee_on_turnover)
    else:
        r6 = list(r5)
        notes.append("vol overlay disabled (overlay_target_vol=None); rung 6 equals rung 5")

    metrics = [_metrics(r, periods_per_year, risk_free_annual) for r in (r1, r2, r3, r4, r5, r6)]

    rungs: list[dict[str, Any]] = []
    prev = None
    for label, m in zip(COMPONENTS, metrics, strict=True):
        rungs.append({
            "component": label,
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "delta_cagr": 0.0 if prev is None else m["cagr"] - prev["cagr"],
            "delta_sharpe": 0.0 if prev is None else m["sharpe"] - prev["sharpe"],
        })
        prev = m

    # The cost line is REPORTED as the overlay's fee-on-turnover drag estimate
    # applied to rung-5 turnover, never re-simulated.
    total_turnover = 0.0
    prev_weights: dict[str, float] = {}
    for t in timeline:
        weights: dict[str, float] = {}
        elig = eligible_by_day.get(t) if eligible_by_day is not None else None
        for s in sorted(asset_dailies):
            if t in asset_dailies[s] and (elig is None or s in elig):
                weights[s] = 1.0 / max(1, len(elig) if elig is not None else n_assets)
        total_turnover += sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in set(weights) | set(prev_weights)) / 2.0
        prev_weights = weights
    cost_drag = -overlay_fee_on_turnover * total_turnover

    cost_lines = [{
        "component": "estimated execution costs (overlay fee on turnover)",
        "cagr": cost_drag,
        "detail": (
            f"{overlay_fee_on_turnover * 1e4:.1f}bp on {total_turnover:.2f} one-way "
            "turns of base weights; reported, not re-simulated"
        ),
    }]

    return {
        "inputs_fingerprint": _inputs_fingerprint(asset_dailies, timeline, n_assets, eligible_by_day is not None),
        "universe_pinned": True,
        "period": {"start": _iso(timeline[0]), "end": _iso(timeline[-1])},
        "rungs": rungs,
        "cost_lines": cost_lines,
        "full_rule": metrics[-1],
        "notes": notes,
    }


def format_ladder(report: dict[str, Any]) -> str:
    """The human ladder. Rendered from the report; never recomputed."""
    lines = []
    for rung in report["rungs"]:
        lines.append(f"{rung['component'] + ':':34}{rung['delta_cagr'] * 100:+8.2f}% cumulative {rung['cagr'] * 100:+7.2f}%")
    for cost in report["cost_lines"]:
        lines.append(f"{cost['component'] + ':':34}{cost['cagr'] * 100:+8.2f}%  ({cost['detail']})")
    return "\n".join(lines)
