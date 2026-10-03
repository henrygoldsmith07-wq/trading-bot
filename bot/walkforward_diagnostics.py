"""Is the strategy consistent, or merely profitable on average?

WHY THIS EXISTS
    Aggregate profitability hides dependence. A portfolio can post a fine
    overall CAGR while one asset, one year, or one exceptional month carries
    all of it — and the next regime pays for the illusion. The canonical
    research report used to show the aggregate and the per-asset table, but
    nothing answered "where did the performance actually come from?".

    This module answers that with fold-by-fold diagnostics, cross-asset
    dispersion, and contribution concentration. It is pure analysis over
    already-computed returns: no network, no file IO, no hidden state, and the
    same inputs always produce the same report. It never decides what the
    strategy is; it only describes how the evidence is distributed.

The concentration flags exist because "one asset made 60% of the money" is a
research finding that must be printed, not smoothed away (see spec: never
hide weak folds, never optimise for flattering numbers).

Paper trading only.
"""
from __future__ import annotations

import statistics
from typing import Any

# A single name accounting for more than half of all positive contribution is
# concentration the reader must see. Defined once; every flag reads this.
CONCENTRATION_FLAG_SHARE = 0.5


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"stdev": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _fold_metrics(daily_returns: list[float], periods_per_year: int, risk_free_annual: float) -> tuple[float, float, float]:
    from .metrics import max_drawdown, sharpe

    if not daily_returns:
        return 0.0, 0.0, 0.0
    equity = 1.0
    path = [1.0]
    for r in daily_returns:
        equity *= 1.0 + float(r)
        path.append(equity)
    return (
        equity - 1.0,
        sharpe([float(r) for r in daily_returns], periods_per_year, risk_free_annual),
        max_drawdown(path),
    )


def analyse_folds(
    folds: list[dict[str, Any]],
    *,
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
) -> dict[str, Any]:
    """Fold-by-fold return, Sharpe and drawdown plus the consistency summary.

    Each input fold names a walk-forward test window with its own out-of-sample
    daily returns. The output shows every fold — worst ones included, always —
    because hiding weak folds is exactly the failure mode this module exists
    to prevent.
    """
    rows: list[dict[str, Any]] = []
    for fold in folds:
        ret, sr, dd = _fold_metrics(
            [float(r) for r in fold.get("daily_returns") or []],
            periods_per_year,
            risk_free_annual,
        )
        rows.append({
            "label": str(fold.get("label") or ""),
            "start": str(fold.get("start") or ""),
            "end": str(fold.get("end") or ""),
            "n_days": len(fold.get("daily_returns") or []),
            "return": ret,
            "sharpe": sr,
            "max_drawdown": dd,
            "strategy": fold.get("strategy"),
        })

    notes: list[str] = []
    if not rows:
        notes.append("no fold results supplied; consistency cannot be assessed")
        return {
            "n_folds": 0,
            "folds": [],
            "median_fold_return": 0.0,
            "median_fold_sharpe": 0.0,
            "worst_fold": None,
            "pct_profitable_folds": 0.0,
            "fold_return_dispersion": {"stdev": 0.0, "min": 0.0, "max": 0.0},
            "notes": notes,
        }

    returns = [r["return"] for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    profitable = sum(1 for r in returns if r > 0)
    worst = min(rows, key=lambda r: (r["return"], r["sharpe"]))

    if len(rows) == 1:
        notes.append("a single fold cannot demonstrate cross-period consistency")

    return {
        "n_folds": len(rows),
        "folds": rows,
        "median_fold_return": statistics.median(returns),
        "median_fold_sharpe": statistics.median(sharpes),
        "worst_fold": {
            "label": worst["label"],
            "return": worst["return"],
            "sharpe": worst["sharpe"],
            "max_drawdown": worst["max_drawdown"],
        },
        "pct_profitable_folds": profitable / len(rows),
        "fold_return_dispersion": {"stdev": _summary(returns)["stdev"], "min": min(returns), "max": max(returns)},
        "notes": notes,
    }


def analyse_asset_dispersion(per_asset: list[dict[str, Any]]) -> dict[str, Any]:
    """How spread out is performance across the assets actually traded?"""
    notes: list[str] = []
    if not per_asset:
        return {
            "assets": 0,
            "positive_sharpe_assets": 0,
            "pct_positive_sharpe": 0.0,
            "sharpe_dispersion": {"stdev": 0.0, "min": 0.0, "median": 0.0, "max": 0.0},
            "cagr_dispersion": {"stdev": 0.0, "min": 0.0, "median": 0.0, "max": 0.0},
            "notes": ["no per-asset results supplied"],
        }

    sharpes = [float(a.get("sharpe", 0.0)) for a in per_asset]
    cagrs = [float(a.get("cagr", 0.0)) for a in per_asset]
    positive = sum(1 for s in sharpes if s > 0)
    if len(per_asset) < 3:
        notes.append("very few assets; dispersion statistics are fragile")
    return {
        "assets": len(per_asset),
        "positive_sharpe_assets": positive,
        "pct_positive_sharpe": positive / len(per_asset),
        "sharpe_dispersion": _summary(sharpes),
        "cagr_dispersion": _summary(cagrs),
        "notes": notes,
    }


def _largest(contributions: dict[str, float] | None) -> tuple[dict[str, Any] | None, list[str], float | None]:
    warnings: list[str] = []
    if not contributions:
        return None, warnings, None
    positives = {k: v for k, v in contributions.items() if v > 0}
    total_positive = sum(positives.values())
    if total_positive <= 0:
        warnings.append("no positive contributions recorded; concentration share undefined")
        name = max(sorted(contributions), key=lambda k: contributions[k])
        return {"name": name, "value": contributions[name], "share_of_positive": 0.0}, warnings, 0.0
    name = max(sorted(positives), key=lambda k: positives[k])
    share = positives[name] / total_positive
    if share > CONCENTRATION_FLAG_SHARE:
        warnings.append(
            f"{name} accounts for {share:.0%} of positive contributions "
            f"(> {CONCENTRATION_FLAG_SHARE:.0%}) — performance is concentrated"
        )
    return {"name": name, "value": positives[name], "share_of_positive": share}, warnings, share


def analyse_concentration(
    *,
    asset_contributions: dict[str, float] | None = None,
    fold_contributions: dict[str, float] | None = None,
    regime_contributions: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Where does the performance come from — one asset, one fold, one regime?

    Contributions are ADDITIVE return contributions (weight x return summed
    over days), never standalone CAGRs: standalone figures cannot be summed
    and would make the shares meaningless. Callers holding only standalone
    metrics must not call this.
    """
    warnings: list[str] = []
    largest_asset, w, asset_share = _largest(asset_contributions)
    warnings.extend(w)
    largest_fold, w, _ = _largest(fold_contributions)
    warnings.extend(w)
    largest_regime, w, _ = _largest(regime_contributions)
    warnings.extend(w)

    herfindahl: float | None = None
    if asset_contributions:
        positives = [v for v in asset_contributions.values() if v > 0]
        total = sum(positives)
        if total > 0:
            herfindahl = sum((v / total) ** 2 for v in positives)

    return {
        "largest_asset_contribution": largest_asset,
        "largest_fold_contribution": largest_fold,
        "largest_regime_contribution": largest_regime,
        "herfindahl_asset": herfindahl,
        "warnings": warnings,
        "_largest_asset_share": asset_share,
    }


def consistency_report(
    folds: dict[str, Any],
    dispersion: dict[str, Any],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    """One deterministic verdict on where performance is coming from.

    Precedence for `dominant_source`, documented here because the choice is
    substantive: insufficient data first (an honest "cannot tell"), then
    one_fold (time dependence is the most dangerous — it means the edge may be
    a single regime), then one_asset (idiosyncratic, not a strategy), then
    one_regime, else spread. Flags mirror the same facts for machine use.
    """
    pct_positive = float(dispersion.get("pct_positive_sharpe", 0.0))
    n_assets = int(dispersion.get("assets", 0))
    n_folds = int(folds.get("n_folds", 0))

    asset_share = concentration.get("_largest_asset_share")
    fold_largest = concentration.get("largest_fold_contribution")
    regime_largest = concentration.get("largest_regime_contribution")

    one_asset = bool(asset_share is not None and asset_share > CONCENTRATION_FLAG_SHARE)
    one_fold = bool(fold_largest and fold_largest.get("share_of_positive", 0) > CONCENTRATION_FLAG_SHARE)
    one_regime = bool(regime_largest and regime_largest.get("share_of_positive", 0) > CONCENTRATION_FLAG_SHARE)
    minority_positive = bool(n_assets > 0 and pct_positive < 0.5)

    warnings = list(concentration.get("warnings", []))
    if minority_positive:
        warnings.append(f"only {pct_positive:.0%} of assets show positive OOS Sharpe")

    if n_assets == 0 or n_folds == 0:
        dominant = "insufficient_data"
        summary = "Not enough fold or asset evidence to say where performance comes from."
    elif one_fold:
        dominant = "one_fold"
        summary = "Most performance comes from a single walk-forward fold; the result is period-dependent."
    elif one_asset:
        dominant = "one_asset"
        summary = "Most performance comes from a single asset; this is closer to a position than a strategy."
    elif one_regime:
        dominant = "one_regime"
        summary = "Most performance comes from one market regime; expect mean reversion in the others."
    else:
        dominant = "spread_across_assets_and_folds"
        summary = "Performance is spread across assets and folds with no single dominant contributor."

    return {
        "dominant_source": dominant,
        "warnings": warnings,
        "flags": {
            "one_asset": one_asset,
            "one_fold": one_fold,
            "one_regime": one_regime,
            "minority_of_assets_positive": minority_positive,
        },
        "summary": summary,
    }
