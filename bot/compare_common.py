"""Helpers shared by the CLI commands and the canonical compare pipeline.

These were private functions on the CLI entry point. They are used by two
modules now, so they live here under public names rather than one module
reaching into the other's privates.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

DAY_MS = 86_400_000

# Identifier of the source-fingerprint algorithm, defined in exactly one
# place so a run record, a freeze manifest and the verifier cannot disagree.
CODE_FP_ALGO = "sha256-lf-v1"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def equity_metrics(returns: list[float], periods_per_year: int = 365, risk_free_annual: float = 0.0) -> dict:
    from .metrics import cagr, calmar, expected_shortfall, max_drawdown, sharpe, sortino, var_hist, volatility

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    days = len(returns)  # daily returns: one calendar day each
    mdd = max_drawdown(equity)
    cagr_v = cagr(equity, days)
    return {
        "final": equity[-1],
        "cagr": cagr_v,
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free_annual),
        "sortino": sortino(returns, periods_per_year, risk_free_annual),
        "calmar": calmar(cagr_v, mdd),
        "max_drawdown": mdd,
        "var95": var_hist(returns, 0.95),
        "es95": expected_shortfall(returns, 0.95),
    }


def vol_overlay(returns: list[float], target: float = 0.25, window: int = 20, fee: float = 0.0015) -> list[float]:
    """Scale portfolio exposure to a trailing-vol target.

    The weight for day t uses only returns up to t-1 - no lookahead. This is
    the risk-management layer that de-risks a crypto portfolio to equity-like
    volatility. During warmup the overlay stays fully invested (w=1.0) and
    charges nothing: there is no transition to pay for.
    """
    out = []
    w = 1.0
    for i, r in enumerate(returns):
        if i >= window:
            hist = returns[i - window : i]
            mean = sum(hist) / window
            var = sum((x - mean) ** 2 for x in hist) / (window - 1)
            rv = math.sqrt(max(var, 0.0) * 365)
            w_new = min(1.0, target / rv) if rv > 0 else 1.0
        else:
            w_new = 1.0
        out.append(w_new * r - fee * abs(w_new - w))
        w = w_new
    return out


def as_date(ms):
    return datetime.fromtimestamp(ms / 1000, tz=UTC).date()


def print_regimes(btc, timeline, port_returns, sp_window, args) -> None:
    from .regimes import label_regimes, segment, segment_metrics, stress_mask

    labels = label_regimes(btc, timeline)
    segments = segment(labels, timeline)
    port_by_day = {t: r for t, r in zip(timeline, port_returns, strict=True)}

    sp_times = [
        int(datetime(r["date"].year, r["date"].month, r["date"].day, tzinfo=UTC).timestamp() * 1000)
        for r in sp_window
    ]
    sp_closes = [r["close"] for r in sp_window]
    spx_by_day = {sp_times[k]: sp_closes[k] / sp_closes[k - 1] - 1.0 for k in range(1, len(sp_window))}
    spx_days = [t for t in timeline if t in spx_by_day]

    print("\nRegime analysis (BTC-proxy trailing 180d return labels):")
    print(f"  {'regime':10}{'period':>15}{'days':>7}{'bot CAGR':>11}{'S&P CAGR':>11}")
    for seg in segments:
        bot_m = segment_metrics(port_by_day, timeline, seg["start"], seg["end"])
        sp_m = segment_metrics(spx_by_day, spx_days, seg["start"], seg["end"])
        span = f"{as_date(seg['start'])}..{str(as_date(seg['end']))[5:]}"
        print(f"  {seg['label']:10}{span:>15}{bot_m['days']:>7}{fmt_pct(bot_m['cagr']):>11}{fmt_pct(sp_m['cagr']):>11}")

    mask = stress_mask(btc, timeline)
    stress_days = [t for t in timeline if mask.get(t)]
    if stress_days:
        sm = segment_metrics(port_by_day, timeline, stress_days[0], stress_days[-1])
        ssm = segment_metrics(spx_by_day, spx_days, stress_days[0], stress_days[-1])
        print(f"\nStress windows (30d crash <=-20% or vol in top decile): {len(stress_days)} days")
        print(f"  bot over stressed span: CAGR {fmt_pct(sm['cagr'])}, maxDD {fmt_pct(sm['max_drawdown'])}")
        print(f"  S&P over stressed span: CAGR {fmt_pct(ssm['cagr'])}, maxDD {fmt_pct(ssm['max_drawdown'])}")
