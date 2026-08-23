"""Fixed portfolio-level rules (no selection, no tuning on forward data).

These are a-priori overlays applied on top of per-asset strategy streams:

- Cross-sectional momentum tilt: rank assets by trailing `tilt_lookback`
  return and tilt sleeves within a capped band (top-ranked asset gets
  (1+max_tilt)x its base weight, bottom gets (1-max_tilt)x, linear in
  between, renormalized). Cross-sectional momentum is a long-documented
  anomaly; the caps bound how much the tilt can do when it misfires.
- Crisis de-risking: when the average pairwise correlation of asset returns
  over the trailing window exceeds `corr_threshold`, diversification has
  broken down and total exposure is scaled by `derisk`.
- Both use strictly trailing data — no lookahead by construction.
"""
from __future__ import annotations

import math


def _trailing_cum_ret(hist: list[float], lookback: int) -> float | None:
    if len(hist) < lookback:
        return None
    eq = 1.0
    for r in hist[-lookback:]:
        eq *= 1.0 + r
    return eq - 1.0


def tilt_multipliers(
    ranks: dict[str, float | None],
    max_tilt: float = 0.5,
) -> dict[str, float]:
    """Linear-in-rank tilt multipliers in [1-max_tilt, 1+max_tilt].

    Assets without enough history get multiplier 1.0 (neutral). Multipliers
    over the assets that have ranks average to 1.0 so gross exposure is
    preserved before the survivorship scaling is applied.
    """
    ranked = sorted((v, s) for s, v in ranks.items() if v is not None)
    k = len(ranked)
    out = {s: 1.0 for s in ranks}
    if k < 2:
        return out
    for i, (_, s) in enumerate(ranked):
        out[s] = 1.0 - max_tilt + 2.0 * max_tilt * (i / (k - 1))
    mean_m = sum(out[s] for s in out if ranks[s] is not None) / k
    for s in out:
        if ranks[s] is not None:
            out[s] /= mean_m
    return out


def avg_pairwise_corr(hists: dict[str, list[float]], window: int) -> float | None:
    """Average pairwise correlation over the trailing `window` returns."""
    usable = {s: h[-window:] for s, h in hists.items() if len(h) >= window}
    syms = list(usable)
    if len(syms) < 2:
        return None
    corrs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a = usable[syms[i]]
            b = usable[syms[j]]
            ma = sum(a) / len(a)
            mb = sum(b) / len(b)
            cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=False))
            va = sum((x - ma) ** 2 for x in a)
            vb = sum((y - mb) ** 2 for y in b)
            if va > 0 and vb > 0:
                corrs.append(cov / math.sqrt(va * vb))
    if not corrs:
        return None
    return sum(corrs) / len(corrs)


def combine_portfolio_rule(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    vol_window: int = 20,
    max_multiple_of_equal: float = 2.0,
    tilt_lookback: int = 90,
    max_tilt: float = 0.5,
    corr_window: int = 60,
    corr_threshold: float = 0.60,
    derisk: float = 0.60,
    use_tilt: bool = True,
    use_crisis: bool = True,
    use_dd_throttle: bool = False,
    dd_trigger: float = -0.10,
    dd_exit: float = -0.05,
    throttle: float = 0.5,
) -> list[float]:
    """Fixed composite rule: inverse-vol base + XS-momentum tilt + crisis
    de-risking + optional drawdown throttle. Same survivorship convention as
    the other combiners: a missing asset's sleeve sits in cash (exposure
    shrinks, never reflows).

    The drawdown throttle cuts exposure to `throttle` once the portfolio's
    own drawdown breaches `dd_trigger`, restoring it only after the drawdown
    recovers above `dd_exit` (hysteresis prevents whipsaw). It reads the
    rule's own realized equity — strictly past data.
    """
    syms = list(asset_dailies)
    hist: dict[str, list[float]] = {s: [] for s in syms}
    out = []
    equity = 1.0
    peak = 1.0
    throttled = False
    for t in timeline:
        present = [s for s in syms if t in asset_dailies[s]]
        gross = 0.0
        if present:
            # base weights: inverse vol (capped), same as combine_portfolio_invvol
            if any(len(hist[s]) >= vol_window for s in present):
                raw = {}
                for s in present:
                    h = hist[s][-vol_window:]
                    if len(h) < 2:
                        raw[s] = 1.0
                        continue
                    m = sum(h) / len(h)
                    var = sum((x - m) ** 2 for x in h) / (len(h) - 1)
                    vol = math.sqrt(max(var, 0.0) * 365)
                    raw[s] = 1.0 / max(vol, 1e-6)
                cap = max_multiple_of_equal / len(present)
                total_raw = sum(raw.values())
                weights = {s: min(cap, raw[s] / total_raw) for s in present}
                free = [s for s in present if weights[s] < cap]
                slack = 1.0 - sum(weights.values())
                free_total = sum(raw[s] for s in free)
                if free and free_total > 0 and slack > 0:
                    for s in free:
                        weights[s] += slack * raw[s] / free_total
            else:
                eq = 1.0 / len(present)
                weights = {s: eq for s in present}

            # cross-sectional momentum tilt (neutral until every present asset
            # has enough history; conservative warmup)
            if use_tilt:
                ranks = {s: _trailing_cum_ret(hist[s], tilt_lookback) for s in present}
                if all(v is not None for v in ranks.values()) and len(present) >= 2:
                    mult = tilt_multipliers(ranks, max_tilt)
                    for s in present:
                        weights[s] *= mult[s]

            gross = sum(weights[s] * asset_dailies[s][t] for s in present)

        exposure = len(present) / n_assets
        if use_crisis:
            corr = avg_pairwise_corr({s: hist[s] for s in present}, corr_window) if present else None
            if corr is not None and corr > corr_threshold:
                exposure *= derisk
        if use_dd_throttle:
            dd = equity / peak - 1.0
            if throttled:
                if dd > dd_exit:
                    throttled = False
            elif dd <= dd_trigger:
                throttled = True
            if throttled:
                exposure *= throttle

        ret = gross * exposure
        out.append(ret)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        for s in present:
            hist[s].append(asset_dailies[s][t])
    return out
