"""Which strategy is THE canonical one, and proving every surface agrees.

This module exists because the repository used to grade one strategy and
display another. The verdict string and the exit code were computed from
"equal_rm" (the risk-managed EQUAL-WEIGHT portfolio), while the frozen
forward experiment executes inverse-vol weighting with a cross-sectional
momentum tilt, crisis de-risking, a 5% rebalance band and a volatility
overlay. Those are different portfolios with very different results: the
equal-weight book reported about 25% CAGR against an S&P 500 near 15%, while
the rule actually under test returned about 12% and trailed the index.

So the headline advertised a win the frozen strategy never achieved.

There is exactly one primary rule identity. Every headline surface -- the
canonical verdict, the CLI exit code, the README block, the dashboard, the
API summary, the freeze manifest and the reproduction tooling -- resolves it
from here, so the displayed claim and the tested strategy cannot drift apart
without a test failing.

Everything else (equal weight, raw equal weight, plain inverse-vol, the
drawdown-throttled and fully-fixed variants) is a COMPARATOR: reported for
context, never used to decide whether the canonical claim passed.

Paper trading only.
"""
from __future__ import annotations

from typing import Any

# --- the one canonical identity ---------------------------------------------

PRIMARY_RULE_ID = "banded_rm"
PRIMARY_RULE_STAT_NAME = "+ tilt + crisis, banded 5% rebalance"
PRIMARY_RULE_DESCRIPTION = (
    "inverse-vol weighting + cross-sectional momentum tilt + crisis de-risk "
    "+ 5% rebalance band + volatility overlay"
)

COMPARATOR_RULE_IDS = (
    "equal_raw",
    "inv_vol_rm",
    "equal_rm",
    "full_rm",
    "throttle_rm",
    "fixed_rm",
    "spx",
    "btc_bh",
)

BENCHMARK_RULE_ID = "spx"

# The selection mode the freeze CLI records for the primary rule.
PRIMARY_SELECTION_MODE = "walk_forward_selected"


class PrimaryRuleMismatch(RuntimeError):
    """The frozen configuration is not the rule the headline claims to grade."""


def _flag(container: Any, key: str, default: bool) -> bool:
    if not isinstance(container, dict):
        return default
    return bool(container.get(key, default))


def rule_id_for_algorithm(algorithm: dict[str, Any]) -> str:
    """Map a frozen algorithm block onto the backtest rule that reproduces it.

The mapping is exact because bot/__main__.py builds each comparator from the
same primitives: full_rm applies tilt and crisis over the plain sleeves,
banded_rm applies them over sleeves carrying a rebalance band, throttle_rm
adds the drawdown throttle, and inv_vol_rm uses inverse-vol weights alone.

Raising on an unrecognised combination is deliberate. If a new freeze
configuration cannot be mapped to a graded rule, the honest answer is
unknown, not a silent fallback onto whichever rule happens to exist.
"""
    if not isinstance(algorithm, dict):
        raise PrimaryRuleMismatch("frozen configuration has no algorithm block")
    mode = str(algorithm.get("selection_mode") or "")
    if mode not in ("", PRIMARY_SELECTION_MODE):
        raise PrimaryRuleMismatch(f"frozen selection_mode {mode!r} is not a graded portfolio rule")
    band = algorithm.get("rebalance_band", 0.0)
    try:
        banded = float(band) > 0.0
    except (TypeError, ValueError) as exc:
        raise PrimaryRuleMismatch(f"rebalance_band is not a number: {band!r}") from exc
    tilt = _flag(algorithm.get("xs_momentum"), "enabled", True)
    crisis = _flag(algorithm.get("crisis_derisk"), "enabled", True)
    throttle = _flag(algorithm.get("drawdown_throttle"), "enabled", False)
    weighting = algorithm.get("weighting")
    weight_mode = str(weighting.get("mode", "inverse_vol")) if isinstance(weighting, dict) else "inverse_vol"
    if weight_mode != "inverse_vol":
        raise PrimaryRuleMismatch(f"weighting mode {weight_mode!r} is not a graded portfolio rule")
    if not tilt and not crisis and not throttle:
        return "inv_vol_rm"
    if tilt and crisis and throttle:
        return "throttle_rm"
    if tilt and crisis and banded:
        return "banded_rm"
    if tilt and crisis and not banded:
        return "full_rm"
    raise PrimaryRuleMismatch(
        f"frozen overlays (tilt={tilt}, crisis={crisis}, throttle={throttle}, "
        f"banded={banded}) do not correspond to any graded portfolio rule"
    )


def assert_primary_rule_is_frozen(algorithm: dict[str, Any]) -> str:
    """Fail loudly when the headline rule and the frozen rule disagree."""
    actual = rule_id_for_algorithm(algorithm)
    if actual != PRIMARY_RULE_ID:
        raise PrimaryRuleMismatch(
            "the frozen configuration does not implement the canonical primary rule.\n"
            f"  canonical primary : {PRIMARY_RULE_ID} ({PRIMARY_RULE_DESCRIPTION})\n"
            f"  frozen config maps: {actual}\n"
            "Freezing this configuration would grade a headline that the forward "
            "experiment never trades. Freeze the canonical rule, or change "
            "PRIMARY_RULE_ID deliberately in bot/canonical_identity.py.",
        )
    return actual


def resolve_primary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """The primary rule metrics block from a canonical run record."""
    if PRIMARY_RULE_ID not in metrics:
        available = sorted(metrics)
        raise PrimaryRuleMismatch(f"canonical record has no {PRIMARY_RULE_ID} metrics block; available: {available}")
    block = metrics[PRIMARY_RULE_ID]
    if not isinstance(block, dict):
        raise PrimaryRuleMismatch(f"{PRIMARY_RULE_ID} metrics block is not an object")
    return block


def resolve_primary_stat(rule_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """The primary rule statistical row from a canonical run record."""
    names = [str(r.get("name", "")) for r in rule_stats]
    matches = [r for r in rule_stats if str(r.get("name", "")) == PRIMARY_RULE_STAT_NAME]
    if not matches:
        raise PrimaryRuleMismatch(f"canonical record has no rule row named {PRIMARY_RULE_STAT_NAME!r}; available: {names}")
    return matches[0]


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_primary_verdict(primary: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    """The canonical headline, built from the primary rule ONLY.

    Kept here rather than inline in the CLI so the claim is testable without
    running a full walk-forward, and so there is exactly one place that can
    decide whether the canonical rule beat the benchmark.

The exit code follows the primary rule. A comparator that beats the index
does not make the canonical claim true, and must never make the build pass.
    """
    beats_cagr = float(primary["cagr"]) > float(benchmark["cagr"])
    beats_sharpe = float(primary["sharpe"]) > float(benchmark["sharpe"])
    beats_mdd = float(primary["max_drawdown"]) > float(benchmark["max_drawdown"])
    text = (
        f"primary rule [{PRIMARY_RULE_ID}] OOS CAGR {'BEATS' if beats_cagr else 'TRAILS'} S&P 500 "
        f"({_fmt_pct(float(primary['cagr']))} vs {_fmt_pct(float(benchmark['cagr']))}); "
        f"Sharpe {'beats' if beats_sharpe else 'trails'} "
        f"({float(primary['sharpe']):.2f} vs {float(benchmark['sharpe']):.2f}); "
        f"max drawdown {'better' if beats_mdd else 'worse'} "
        f"({_fmt_pct(float(primary['max_drawdown']))} vs {_fmt_pct(float(benchmark['max_drawdown']))})"
    )
    return {
        "verdict": text,
        "beats_cagr": beats_cagr,
        "beats_sharpe": beats_sharpe,
        "beats_max_drawdown": beats_mdd,
        "exit_code": 0 if (beats_cagr and beats_sharpe) else 1,
        "primary_rule_id": PRIMARY_RULE_ID,
    }

def comparator_rows(rule_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every non-primary rule row, for context tables."""
    return [r for r in rule_stats if str(r.get("name", "")) != PRIMARY_RULE_STAT_NAME]
