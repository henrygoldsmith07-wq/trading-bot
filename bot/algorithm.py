"""The frozen algorithm: every quantity that can affect a return.

A freeze used to record strategies + frictions + a vol target, while the
headline backtest also ran inverse-vol weighting, an XS-momentum tilt,
crisis de-risking, a rebalance band and (optionally) a drawdown throttle.
That meant backtest and forward test could be different bots wearing the
same name. This module defines the COMPLETE portfolio algorithm as one
validated, hashable spec that:

- `compare`/`freeze` build and seal into the manifest,
- `prospective.run_step` consumes day by day in the forward test,
- `portfolio_rules.combine_portfolio_rule` consumes across whole timelines,

so backtest and forward are provably the same construction. Unknown keys
are rejected: a typo must never silently become "default".
"""
from __future__ import annotations

import hashlib
import json

# --- defaults: the headline configuration ----------------------------------

ALGORITHM_DEFAULTS: dict = {
    "selection_mode": "walk_forward_selected",  # or "fixed_risk_ensemble"
    "candidate_pool_version": None,  # sha256 over sorted candidate reprs
    "universe_selection_rule": "top-N crypto pairs by 24h quote volume (+SPY/GLD/TLT ETFs); today's list, survivorship disclosed",
    "weighting": {
        "mode": "inverse_vol",
        "vol_window": 20,
        "max_multiple_of_equal": 2.0,
    },
    "xs_momentum": {
        "enabled": True,
        "lookback": 90,
        "max_tilt": 0.5,
    },
    "crisis_derisk": {
        "enabled": True,
        "corr_window": 60,
        "corr_threshold": 0.60,
        "multiplier": 0.60,
    },
    "rebalance_band": 0.05,
    "drawdown_throttle": {
        "enabled": False,
        "dd_trigger": -0.10,
        "dd_exit": -0.05,
        "factor": 0.5,
    },
    "overlay": {
        "enabled": True,
        "target_vol": 0.25,
        "window": 20,
        "fee_on_turnover": 0.0015,
    },
}

# Sections whose sub-keys are validated; scalars validated separately.
_SECTIONS = {"weighting", "xs_momentum", "crisis_derisk", "drawdown_throttle", "overlay"}
_SCALARS = {"selection_mode", "candidate_pool_version", "universe_selection_rule", "rebalance_band"}


def candidate_pool_version(candidates: list | None = None) -> str:
    """sha256 over the sorted reprs of the candidate pool: any change to the
    pool (a new grid point, a changed default) changes the version."""
    if candidates is None:
        from .strategy import build_candidates

        candidates = build_candidates()
    blob = "\n".join(sorted(repr(c) for c in candidates)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def validate_algorithm(spec: dict) -> None:
    """Reject unknown keys and bad types. Typos are behaviour changes."""
    unknown_top = set(spec) - set(ALGORITHM_DEFAULTS)
    if unknown_top:
        raise ValueError(f"unknown algorithm key(s): {sorted(unknown_top)}")
    missing = [k for k in _SCALARS | _SECTIONS if k not in spec]
    if missing:
        raise ValueError(f"algorithm missing required key(s): {sorted(missing)}")
    for section in _SECTIONS:
        sub = spec[section]
        if not isinstance(sub, dict):
            raise ValueError(f"algorithm.{section} must be a mapping")
        unknown_sub = set(sub) - set(ALGORITHM_DEFAULTS[section])
        if unknown_sub:
            raise ValueError(f"unknown key(s) in algorithm.{section}: {sorted(unknown_sub)}")
    band = spec["rebalance_band"]
    if not isinstance(band, (int, float)) or not 0 <= band < 1:
        raise ValueError("rebalance_band must be a fraction in [0, 1)")
    if spec["selection_mode"] not in ("walk_forward_selected", "fixed_risk_ensemble"):
        raise ValueError("selection_mode must be 'walk_forward_selected' or 'fixed_risk_ensemble'")
    xs = spec["xs_momentum"]
    if not isinstance(xs.get("enabled"), bool):
        raise ValueError("xs_momentum.enabled must be bool")
    cd = spec["crisis_derisk"]
    if not isinstance(cd.get("enabled"), bool):
        raise ValueError("crisis_derisk.enabled must be bool")
    th = spec["drawdown_throttle"]
    if not isinstance(th.get("enabled"), bool):
        raise ValueError("drawdown_throttle.enabled must be bool")


def build_algorithm(
    *,
    selection_mode: str = "walk_forward_selected",
    universe_selection_rule: str | None = None,
    rebalance_band: float = ALGORITHM_DEFAULTS["rebalance_band"],
    use_tilt: bool = True,
    tilt_lookback: int = 90,
    max_tilt: float = 0.5,
    use_crisis: bool = True,
    corr_window: int = 60,
    corr_threshold: float = 0.60,
    derisk_factor: float = 0.60,
    use_throttle: bool = False,
    dd_trigger: float = -0.10,
    dd_exit: float = -0.05,
    throttle_factor: float = 0.5,
    vol_window: int = 20,
    max_multiple_of_equal: float = 2.0,
    overlay_enabled: bool = True,
    target_vol: float = 0.25,
    overlay_window: int = 20,
    overlay_fee: float = 0.0015,
    with_pool_version: bool = True,
) -> dict:
    """Full spec with every knob explicit. `with_pool_version=False` exists
    only for tests constructing specs without importing strategy data."""
    spec: dict = {
        "selection_mode": selection_mode,
        "candidate_pool_version": candidate_pool_version() if with_pool_version else "test-pool",
        "universe_selection_rule": universe_selection_rule or ALGORITHM_DEFAULTS["universe_selection_rule"],
        "weighting": {
            "mode": "inverse_vol",
            "vol_window": vol_window,
            "max_multiple_of_equal": max_multiple_of_equal,
        },
        "xs_momentum": {"enabled": use_tilt, "lookback": tilt_lookback, "max_tilt": max_tilt},
        "crisis_derisk": {
            "enabled": use_crisis,
            "corr_window": corr_window,
            "corr_threshold": corr_threshold,
            "multiplier": derisk_factor,
        },
        "rebalance_band": rebalance_band,
        "drawdown_throttle": {
            "enabled": use_throttle,
            "dd_trigger": dd_trigger,
            "dd_exit": dd_exit,
            "factor": throttle_factor,
        },
        "overlay": {
            "enabled": overlay_enabled,
            "target_vol": target_vol,
            "window": overlay_window,
            "fee_on_turnover": overlay_fee,
        },
    }
    validate_algorithm(spec)
    return spec


def algorithm_fingerprint(algorithm: dict) -> str:
    """sha256 of the canonical algorithm JSON — printed alongside config/code hashes."""
    blob = json.dumps(algorithm, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
