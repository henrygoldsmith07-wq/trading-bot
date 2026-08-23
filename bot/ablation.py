"""Ablation studies: what is each component actually contributing?

- Strategy-family ablation: drop one candidate family at a time from the
  walk-forward pool and re-run selection. A family whose removal *helps*
  OOS was pure noise fit; a family whose removal hurts was carrying edge.
- Overlay ablation: toggle each fixed portfolio rule (inverse-vol sizing,
  cross-sectional tilt, crisis de-risking, volatility targeting) so every
  headline claim can be attributed to a specific mechanism rather than
  asserted wholesale.
"""
from __future__ import annotations

from .metrics import cagr, max_drawdown, sharpe
from .walkforward import combine_portfolio, combine_portfolio_invvol, walk_forward_at


def family_of(candidate) -> str:
    """Family label from the candidate's repr prefix ('TrendVol(50,0.30)' ->
    'TrendVol'; Ensembles collapse to one 'Ensemble' family)."""
    return repr(candidate).split("(", 1)[0]


def family_ablation(
    candles: list[dict],
    folds: list[tuple[int, int]],
    candidates: list,
    periods_per_year: int = 365,
    **engine_kwargs,
) -> list[dict]:
    """Re-run walk-forward selection once per omitted family.

    Returns rows sorted by damage inflicted (most-negative Sharpe delta
    first = most valuable family). Includes the full-pool baseline row.
    """
    def _oos(pool):
        if not pool:
            return None
        wf = walk_forward_at(candles, folds, candidates=pool, periods_per_year=periods_per_year, **engine_kwargs)
        days = sorted(wf["daily"])
        rets = [wf["daily"][t] for t in days]
        return {
            "cagr": wf["cagr"],
            "sharpe": wf["sharpe"],
            "max_drawdown": wf["max_drawdown"],
            "returns": rets,
        }

    baseline = _oos(candidates)
    families = sorted({family_of(c) for c in candidates})
    rows = []
    for fam in families:
        reduced = [c for c in candidates if family_of(c) != fam]
        res = _oos(reduced)
        rows.append(
            {
                "omitted_family": fam,
                "n_remaining": len(reduced),
                "sharpe_delta": (res["sharpe"] - baseline["sharpe"]) if res else float("nan"),
                "cagr_delta": (res["cagr"] - baseline["cagr"]) if res else float("nan"),
                **({"sharpe": res["sharpe"], "cagr": res["cagr"]} if res else {"sharpe": float("nan"), "cagr": float("nan")}),
            }
        )
    rows.sort(key=lambda r: r["sharpe_delta"])
    return [{"omitted_family": "(none — full pool)", "n_remaining": len(candidates), "sharpe": baseline["sharpe"], "cagr": baseline["cagr"], "sharpe_delta": 0.0, "cagr_delta": 0.0}] + rows


def overlay_ablation(
    asset_dailies: dict[str, dict[int, float]],
    timeline: list[int],
    n_assets: int,
    target_vol: float = 0.25,
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
) -> dict[str, dict]:
    """Metrics for every on/off combination of the fixed portfolio rules.

    Variants: equal-weight base; inverse-vol sizing; +XS-momentum tilt;
    +crisis de-risking; each additionally run through the trailing-vol
    targeting overlay (`_vol_target`) so raw and risk-managed numbers are
    separable.
    """
    from .portfolio_rules import combine_portfolio_rule

    equal = combine_portfolio(asset_dailies, timeline, n_assets)
    iv = combine_portfolio_invvol(asset_dailies, timeline, n_assets)
    tilt = combine_portfolio_rule(asset_dailies, timeline, n_assets, use_tilt=True, use_crisis=False)
    crisis = combine_portfolio_rule(asset_dailies, timeline, n_assets, use_tilt=True, use_crisis=True)

    out = {}
    for name, rets in (
        ("equal_weight", equal),
        ("inv_vol", iv),
        ("inv_vol+tilt", tilt),
        ("inv_vol+tilt+crisis", crisis),
    ):
        out[name] = _summarize(rets, periods_per_year, risk_free_annual)
        out[name + "|vol_targeted"] = _summarize(_vol_target(rets, target_vol), periods_per_year, risk_free_annual)
    return out


def _vol_target(returns: list[float], target: float, window: int = 20, fee: float = 0.0015) -> list[float]:
    """Trailing-vol exposure scaling (same convention as the CLI overlay).
    Warmup stays fully invested with no phantom transition fee."""
    import math

    out = []
    w = 1.0
    for i, r in enumerate(returns):
        if i >= window:
            hist = returns[i - window : i]
            m = sum(hist) / window
            var = sum((x - m) ** 2 for x in hist) / (window - 1)
            rv = math.sqrt(max(var, 0.0) * 365)
            w_new = min(1.0, target / rv) if rv > 0 else 1.0
        else:
            w_new = 1.0
        out.append(w_new * r - fee * abs(w_new - w))
        w = w_new
    return out


def _summarize(returns: list[float], periods_per_year: int, risk_free_annual: float) -> dict:
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = max(len(returns), 1)
    c = cagr(equity, days)
    mdd = max_drawdown(equity)
    return {
        "cagr": c,
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "max_drawdown": mdd,
        "calmar": c / abs(mdd) if mdd < 0 else 0.0,
        "final": equity[-1],
    }
