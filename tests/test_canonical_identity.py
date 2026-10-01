"""The canonical headline must describe the strategy that is actually frozen.

These tests exist because the repository once graded the risk-managed
EQUAL-WEIGHT portfolio while freezing and trading inverse-vol + tilt + crisis
+ 5% band + vol overlay. The two have very different returns, so the headline
claimed a benchmark win the frozen rule never earned.

Any divergence between the displayed claim and the frozen rule must fail here.
"""
from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

import pytest

from bot.canonical_identity import (
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
    PRIMARY_RULE_STAT_NAME,
    PrimaryRuleMismatch,
    assert_primary_rule_is_frozen,
    build_primary_verdict,
    resolve_primary_metrics,
    resolve_primary_stat,
    rule_id_for_algorithm,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_algorithm() -> dict:
    return _load("freeze.json")["config"]["algorithm"]


@pytest.fixture(scope="module")
def canonical_v2() -> dict:
    return _load("runs/canonical-v2/run.json")


# ---------------------------------------------------------------------------
# identity: one rule, and the freeze agrees with it
# ---------------------------------------------------------------------------

def test_committed_freeze_implements_the_primary_rule(frozen_algorithm: dict) -> None:
    """The shipped freeze must execute the rule the headline grades."""
    assert assert_primary_rule_is_frozen(frozen_algorithm) == PRIMARY_RULE_ID


def test_primary_rule_is_not_one_of_the_comparators() -> None:
    assert PRIMARY_RULE_ID not in ("equal_rm", "equal_raw", "inv_vol_rm", "full_rm", "fixed_rm")


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"rebalance_band": 0.05, "xs_momentum": {"enabled": True}, "crisis_derisk": {"enabled": True}}, "banded_rm"),
        ({"rebalance_band": 0.0, "xs_momentum": {"enabled": True}, "crisis_derisk": {"enabled": True}}, "full_rm"),
        ({"rebalance_band": 0.0, "xs_momentum": {"enabled": True}, "crisis_derisk": {"enabled": True},
          "drawdown_throttle": {"enabled": True}}, "throttle_rm"),
        ({"rebalance_band": 0.05, "xs_momentum": {"enabled": False}, "crisis_derisk": {"enabled": False}}, "inv_vol_rm"),
    ],
)
def test_frozen_algorithm_maps_to_the_right_rule(overrides: dict, expected: str) -> None:
    algo = {"weighting": {"mode": "inverse_vol"}, **overrides}
    assert rule_id_for_algorithm(algo) == expected


def test_a_freeze_that_is_not_the_primary_rule_is_refused() -> None:
    algo = {"weighting": {"mode": "inverse_vol"}, "rebalance_band": 0.0,
            "xs_momentum": {"enabled": True}, "crisis_derisk": {"enabled": True}}
    with pytest.raises(PrimaryRuleMismatch) as exc:
        assert_primary_rule_is_frozen(algo)
    assert "full_rm" in str(exc.value)


def test_unknown_frozen_overlay_combination_is_refused() -> None:
    algo = {"weighting": {"mode": "inverse_vol"}, "rebalance_band": 0.05,
            "xs_momentum": {"enabled": False}, "crisis_derisk": {"enabled": True}}
    with pytest.raises(PrimaryRuleMismatch):
        rule_id_for_algorithm(algo)


def test_non_inverse_vol_weighting_is_refused() -> None:
    with pytest.raises(PrimaryRuleMismatch):
        rule_id_for_algorithm({"weighting": {"mode": "equal"}, "rebalance_band": 0.05})


def test_primary_resolves_in_every_committed_canonical_record() -> None:
    for name in ("canonical-v1", "canonical-v2"):
        results = _load(f"runs/{name}/run.json")["results"]
        metrics = results["metrics"]
        primary = resolve_primary_metrics(metrics)
        stat = resolve_primary_stat(metrics["rules"])
        assert stat["name"] == PRIMARY_RULE_STAT_NAME
        assert primary["cagr"] == pytest.approx(stat["cagr"]), f"{name}: metrics and stat rows disagree"


def test_primary_block_is_present_in_both_canonical_records() -> None:
    for name in ("canonical-v1", "canonical-v2"):
        metrics = _load(f"runs/{name}/run.json")["results"]["metrics"]
        assert PRIMARY_RULE_ID in metrics


# ---------------------------------------------------------------------------
# the headline follows the primary rule, not a stronger comparator
# ---------------------------------------------------------------------------

def _mk(cagr: float, sharpe: float, mdd: float = -0.1) -> dict:
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd}


def test_headline_reports_trailing_when_the_primary_rule_trails() -> None:
    v = build_primary_verdict(_mk(0.117, 0.74), _mk(0.149, 0.74))
    assert "TRAILS" in v["verdict"]
    assert "11.7%" in v["verdict"]
    assert PRIMARY_RULE_ID in v["verdict"]


def test_a_beating_comparator_cannot_make_the_canonical_claim_pass() -> None:
    """The exact bug: equal-weight beat the index while the frozen rule did not."""
    spx = _mk(0.149, 0.74, -0.254)
    equal_weight = _mk(0.254, 1.05, -0.238)
    primary = _mk(0.117, 0.74, -0.098)
    assert equal_weight["cagr"] > spx["cagr"], "precondition: the comparator wins"
    assert primary["cagr"] < spx["cagr"], "precondition: the frozen rule loses"
    v = build_primary_verdict(primary, spx)
    assert "TRAILS" in v["verdict"]
    assert v["exit_code"] == 1
    # the headline must compare the PRIMARY against the index, never the
    # comparator against the index
    assert "11.7% vs 14.9%" in v["verdict"]
    assert "25.4% vs 14.9%" not in v["verdict"]


def test_exit_code_follows_the_primary_rule_only() -> None:
    assert build_primary_verdict(_mk(0.20, 1.2), _mk(0.149, 0.74))["exit_code"] == 0
    assert build_primary_verdict(_mk(0.10, 1.2), _mk(0.149, 0.74))["exit_code"] == 1
    assert build_primary_verdict(_mk(0.20, 0.50), _mk(0.149, 0.74))["exit_code"] == 1


def test_verdict_module_grades_the_primary_rule_by_default() -> None:
    from bot.verdict import build_verdict

    results = _load("runs/canonical-v2/run.json")["results"]
    v = build_verdict(
        canonical_rule_stats=results["metrics"]["rules"],
        canonical_per_asset=results["per_asset"],
        canonical_n_folds=results["n_folds"],
        pool_size=len(results["picks_counter"]),
        ledger_search_n=35,
        cost_report=None,
        forward=None,
    )
    graded = v["details"]["historical"]["inputs"]
    assert graded["rule"] == PRIMARY_RULE_STAT_NAME


def test_readme_block_headlines_the_primary_rule_not_a_comparator(canonical_v2: dict) -> None:
    sys_path_backup = list(_sys.path)
    _sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import derive_canonical_readme as dcr

        rendered = dcr.render(canonical_v2)
    finally:
        _sys.path[:] = sys_path_backup

    metrics = canonical_v2["results"]["metrics"]
    primary_pct = f"{metrics[PRIMARY_RULE_ID]['cagr'] * 100:.1f}%"
    equal_pct = f"{metrics['equal_rm']['cagr'] * 100:.1f}%"
    assert PRIMARY_RULE_ID in rendered
    assert PRIMARY_RULE_DESCRIPTION in rendered
    # the canonical summary table must state the PRIMARY figure next to the index
    summary_line = next(ln for ln in rendered.splitlines() if ln.startswith("| OOS CAGR"))
    assert primary_pct in summary_line
    assert equal_pct not in summary_line


def test_readme_renderer_tolerates_an_unavailable_dsr(canonical_v2: dict) -> None:
    _sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import derive_canonical_readme as dcr

        rec = json.loads(json.dumps(canonical_v2))
        for row in rec["results"]["metrics"]["rules"]:
            row["dsr"] = None
            row["dsr_available"] = False
            row["dsr_unavailable_reason"] = "no search history recorded"
        rendered = dcr.render(rec)
    finally:
        _sys.path.pop(0)
    assert "n/a" in rendered
