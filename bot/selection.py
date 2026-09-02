"""Robust strategy selection: replacing the argmax-in-sample-Sharpe rule.

## Why this module exists

`walk_forward_at` historically picked the candidate with the highest
in-sample Sharpe ratio. That number is the *maximum* of ~85 noisy
statistics, so it is biased upward by construction: the winner is the
candidate that drew the luckiest sample, not the one with the most edge.
Picking the argmax of a noisy statistic is exactly the procedure the
deflated Sharpe ratio exists to punish, and the repo measures the damage —
single-asset DSR collapses to ~0.11 against a 74-85 trial count.

## What replaces it

Three *pre-registered* rules. "Pre-registered" matters: every constant
below comes from statistical theory, not from a sweep over out-of-sample
results. Tuning these on OOS data would re-create the very overfitting
they are meant to prevent, so they are fixed at their textbook values and
reported, not optimised.

1. **Lo (2002) standard error.** A Sharpe ratio estimated from T years of
   data carries a standard error of roughly 1/sqrt(T). A candidate that
   wins by less than that margin has not won at all.
2. **Breiman's one-standard-error rule** (CART pruning, 1984). Restrict to
   the candidates statistically indistinguishable from the winner, then
   choose among *those* by a robustness criterion instead of by raw
   performance. The eligible set is typically large, which is the point:
   it converts "find the best" into "find the best of the good".
3. **Worst-subwindow (minimax) tie-break.** Among the eligible set, prefer
   the candidate whose *worst* contiguous sub-period is best. This is the
   direct antidote to the failure mode the README documents: trimming 180
   days off the start of the window cut single-asset CAGR from 26.5% to
   7.3%, i.e. the headline result was one regime wearing a trenchcoat.
   A candidate cannot win here on the strength of one lucky episode.

Optionally, a fourth:

4. **Shrinkage toward a fixed prior.** Blend the pick with the a-priori
   `risk_ensemble()` at a credibility weight derived from how far the pick
   beat the prior *relative to the noise*: alpha = gap / (gap + SE). When
   the search found nothing beyond noise, alpha -> 0 and the bot trades
   the prior; when it found something real, alpha -> 1.

Honest caveat, stated up front: shrinkage reduces the *variance* of the
pick, it does not reduce the *trial count*. DSR is a function of the
trial count, so this module does not make DSR arithmetically better. What
it can do is raise genuine out-of-sample Sharpe by picking better
strategies, which is a different and more defensible claim — and it is
the one the measurement in `bot/research.py` actually tests.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .engine import run_strategy
from .metrics import kurtosis, skewness
from .strategy import Blend, WeightStrategy, risk_ensemble

# Textbook constants, deliberately NOT fitted on out-of-sample data:
DEFAULT_SUBWINDOWS = 3       # split the training window into thirds
DEFAULT_SE_MULTIPLIER = 1.0  # Breiman's "one standard error"
MIN_OBS_FOR_SE = 8           # below this a Sharpe SE is not meaningful

# Score given to a sub-window with NO volatility but a non-zero excess
# return. `metrics.sharpe` returns 0.0 there (0/0), which is the right
# convention for a flat book but badly wrong for the minimax rule: a
# stretch that loses a steady 0.3%/day every single day is the worst
# possible regime, and scoring it 0.0 would rank it above a volatile but
# profitable stretch. That is precisely the failure this module exists to
# catch, so the degenerate case is scored explicitly instead. 10.0 is far
# outside any plausible annualised Sharpe, so it only ever decides
# ordering, never magnitude.
DEGENERATE_SHARPE = 10.0


class SelectionFn:
    """A `selection_fn` for `walk_forward_at` that publishes its diagnostics.

    The rule factories below need to expose the full trial-Sharpe record
    after they run, because without it a robust-selection run has an empty
    trial record and cannot be deflated at all — it would be unauditable by
    construction. Hanging those attributes off a closure works at runtime
    but is invisible to the type checker, so every read would need a
    `# type: ignore`. A callable object carries them properly instead.

    Diagnostics are published on every call, including the early "nothing
    ran" exit, so a caller reading `last_diagnostics` always sees the state
    of the most recent selection rather than a stale one.
    """

    def __init__(
        self,
        rule: str,
        worker: Callable[[list, list[dict], dict], tuple[WeightStrategy | None, list[float], dict[str, Any]]],
    ) -> None:
        self.rule = rule
        self._worker = worker
        self.last_trial_sharpes: list[float] = []
        self.last_diagnostics: dict[str, Any] = {"rule": rule, "n_trials": 0, "calls": 0}

    def __call__(self, candidates: list, train_slice: list[dict], engine_kwargs: dict):
        chosen, trial_sharpes, diagnostics = self._worker(candidates, train_slice, engine_kwargs)
        self.last_trial_sharpes = trial_sharpes
        self.last_diagnostics = diagnostics
        return chosen


def sharpe_standard_error(
    sharpe_annual: float,
    n_obs: int,
    periods_per_year: int = 365,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Lo (2002) asymptotic standard error of an ANNUALISED Sharpe ratio.

    For IID returns the per-period Sharpe estimator has asymptotic variance
    (1 + SR_p^2 / 2) / n; the general form adds the skew and kurtosis
    corrections. Annualising multiplies the standard error by
    sqrt(periods_per_year), which is why a daily-sampled Sharpe is better
    determined than an annual-sampled one over the same calendar span.

    Sanity check: 3 years of daily data (n=1095, ppy=365) gives
    SE ~ 0.58 for SR = 1.0 — the familiar 1/sqrt(years) rule of thumb.
    A Sharpe below ~1.1 is therefore not distinguishable from zero at
    95% confidence on three years of data.
    """
    if n_obs < MIN_OBS_FOR_SE:
        return float("inf")
    sr_p = sharpe_annual / math.sqrt(periods_per_year)
    var_p = (1.0 - skew * sr_p + 0.25 * (kurt - 1.0) * sr_p * sr_p) / n_obs
    if var_p <= 0:
        return 0.0
    return math.sqrt(var_p) * math.sqrt(periods_per_year)


def _chunk_sharpe(chunk: list[float], periods_per_year: int, risk_free_annual: float) -> float:
    """Excess-return Sharpe of one sub-window, with the zero-variance case
    resolved by sign rather than collapsed to zero (see DEGENERATE_SHARPE)."""
    m = sum(chunk) / len(chunk)
    var = sum((x - m) ** 2 for x in chunk) / (len(chunk) - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        # No dispersion. Score by the sign of the RAW return rather than
        # the excess return: a flat book (every return exactly 0) is
        # genuinely neutral and must keep scoring 0.0 whatever the cash
        # rate, whereas a book that loses the same amount day after day is
        # the worst regime there is and must not score 0. The absolute
        # tolerance also stops a near-constant block from producing the
        # 1e17-style Sharpe that floating-point dust in the variance
        # would otherwise generate.
        if m > 0:
            return DEGENERATE_SHARPE
        if m < 0:
            return -DEGENERATE_SHARPE
        return 0.0
    excess = m - risk_free_annual / periods_per_year
    return excess / sd * math.sqrt(periods_per_year)


def subwindow_sharpes(
    returns: list[float],
    n_windows: int = DEFAULT_SUBWINDOWS,
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
) -> list[float]:
    """Sharpe of each contiguous `n_windows`-way split of the return series."""
    n = len(returns)
    k = max(1, min(n_windows, n))
    size = n // k
    if size < 2:
        return []
    out = []
    for j in range(k):
        lo = j * size
        hi = n if j == k - 1 else (j + 1) * size
        chunk = returns[lo:hi]
        if len(chunk) >= 2:
            out.append(_chunk_sharpe(chunk, periods_per_year, risk_free_annual))
    return out


def worst_subwindow_sharpe(
    returns: list[float],
    n_windows: int = DEFAULT_SUBWINDOWS,
    periods_per_year: int = 365,
    risk_free_annual: float = 0.0,
) -> float:
    """Minimax estimate: the WORST Sharpe over contiguous sub-periods.

    A lower (not central) estimate of skill. It is the right criterion for
    selection because the failure we are guarding against is not "the
    average was slightly overstated" but "the whole result came from one
    regime and will not repeat".
    """
    windows = subwindow_sharpes(returns, n_windows, periods_per_year, risk_free_annual)
    if not windows:
        return float("-inf")
    return min(windows)


def evaluate_candidates(candidates: list, train_slice: list[dict], engine_kwargs: dict) -> list[dict]:
    """Backtest every candidate on the training window.

    Returns one record per candidate that ran: sharpe, the return series
    (needed for the sub-window and SE computations) and turnover.
    Candidates that raise are skipped, matching the default selector.
    """
    records = []
    for cand in candidates:
        try:
            tr = run_strategy(train_slice, cand.weight_at, **engine_kwargs)
        except (ValueError, ZeroDivisionError):
            continue
        rets = tr["returns"]
        if len(rets) < 2:
            continue
        records.append(
            {
                "strategy": cand,
                "sharpe": tr["sharpe"],
                "returns": rets,
                "turnover": tr["turnover"],
            }
        )
    return records


def credibility_weight(gap: float, se: float) -> float:
    """How much to trust a selection that beat its prior by `gap`.

    Standard credibility/Bayesian form: gap / (gap + se). The pick gets
    that weight and the prior the rest. gap == se is the natural
    break-even — the selection is exactly as large as the noise it has to
    be distinguished from — and gives 0.5.
    """
    if gap <= 0:
        return 0.0
    if math.isinf(se):
        return 0.0
    if se <= 0:
        return 1.0
    return gap / (gap + se)


def make_robust_selection_fn(
    subwindows: int = DEFAULT_SUBWINDOWS,
    se_multiplier: float = DEFAULT_SE_MULTIPLIER,
    prior: WeightStrategy | None = None,
    shrink: bool = False,
):
    """Build a drop-in `selection_fn` for `walk_forward_at`.

    The rule, in order:
      1. backtest every candidate on the training window;
      2. compute the winner's Sharpe standard error (Lo 2002) and keep
         every candidate within `se_multiplier` SEs of the winner
         (Breiman's one-standard-error rule);
      3. among those, take the best WORST sub-window Sharpe (minimax);
      4. if `shrink`, blend the winner with `prior` at a credibility
         weight derived from how far it beat the prior relative to noise.

    Diagnostics are left on the returned callable (`last_diagnostics`) so
    the caller can recover the full trial-Sharpe record for deflation —
    without them a robust-selection run would be unable to compute DSR at
    all, which would make the rule un-auditable by construction.
    """

    def worker(candidates, train_slice, engine_kwargs):
        """Returns (chosen, trial_sharpes, diagnostics).

        Structured as a worker rather than a bare closure so `SelectionFn`
        can publish the diagnostics on every exit path, including this one.
        """
        ppy = engine_kwargs.get("periods_per_year", 365)
        rf = engine_kwargs.get("risk_free_annual", 0.0)

        records = evaluate_candidates(candidates, train_slice, engine_kwargs)
        trial_sharpes = [r["sharpe"] for r in records]
        if not records:
            return None, trial_sharpes, {"n_trials": 0, "rule": "robust"}

        best = max(records, key=lambda r: r["sharpe"])
        n_obs = len(best["returns"])
        se = sharpe_standard_error(
            best["sharpe"],
            n_obs,
            periods_per_year=ppy,
            skew=skewness(best["returns"]),
            kurt=kurtosis(best["returns"]),
        )
        threshold = best["sharpe"] - se_multiplier * se
        eligible = [r for r in records if r["sharpe"] >= threshold]
        if not eligible:
            eligible = [best]

        for r in eligible:
            r["worst"] = worst_subwindow_sharpe(r["returns"], subwindows, ppy, rf)
        pick = max(eligible, key=lambda r: r["worst"])

        alpha = 1.0
        prior_sharpe = None
        gap = None
        chosen = pick["strategy"]
        if shrink:
            prior_strategy = prior if prior is not None else risk_ensemble()
            try:
                ptr = run_strategy(train_slice, prior_strategy.weight_at, **engine_kwargs)
                prior_sharpe = ptr["sharpe"]
            except (ValueError, ZeroDivisionError):
                # None, not -inf. "We could not measure this" and "this is
                # infinitely bad" are different claims, and conflating them
                # would make an unmeasurable prior shrink the pick to nothing.
                # -inf also does not survive a JSON round-trip, and these
                # diagnostics are meant to be written to the trial record.
                prior_sharpe = None
            if prior_sharpe is not None and math.isfinite(prior_sharpe):
                gap = pick["sharpe"] - prior_sharpe
                alpha = credibility_weight(gap, se)
                if alpha < 1.0:
                    chosen = Blend([pick["strategy"], prior_strategy], [alpha, 1.0 - alpha])

        diagnostics = {
            "rule": "robust",
            "n_trials": len(candidates),
            "n_ran": len(records),
            "n_eligible": len(eligible),
            "best_train_sharpe": best["sharpe"],
            "sharpe_se": se,
            "threshold": threshold,
            "picked": repr(pick["strategy"]),
            "picked_train_sharpe": pick["sharpe"],
            "picked_worst_subwindow": pick["worst"],
            "shrink_alpha": alpha,
            # Published even when None. alpha == 0.0 is a legitimate outcome
            # (the pick failed to beat the prior) but it is indistinguishable
            # from "the prior could not be measured" unless the prior's own
            # Sharpe and the gap are recorded alongside it.
            "prior_train_sharpe": prior_sharpe,
            "shrink_gap": gap,
            "chosen": repr(chosen),
        }
        return chosen, trial_sharpes, diagnostics

    return SelectionFn("robust", worker)


def make_one_se_selection_fn(se_multiplier: float = DEFAULT_SE_MULTIPLIER, tie_break: str = "worst_subwindow"):
    """The one-standard-error rule with a pluggable tie-break.

    Kept separate from `make_robust_selection_fn` so the effect of the
    1-SE restriction can be measured on its own, without the minimax
    tie-break or shrinkage layered on top. `tie_break` is one of:
      * "worst_subwindow" — maximises the worst sub-period Sharpe;
      * "lowest_turnover" — cheapest to trade, most robust to cost-model
        error, and the tie-break Breiman's original rule would pick
        (parsimony: prefer the simplest model within one SE).
    """
    if tie_break not in ("worst_subwindow", "lowest_turnover"):
        raise ValueError(f"unknown tie_break {tie_break!r}")

    def worker(candidates, train_slice, engine_kwargs):
        """Returns (chosen, trial_sharpes, diagnostics). See the note on the
        sibling worker in `make_robust_selection_fn`."""
        ppy = engine_kwargs.get("periods_per_year", 365)
        rf = engine_kwargs.get("risk_free_annual", 0.0)

        records = evaluate_candidates(candidates, train_slice, engine_kwargs)
        trial_sharpes = [r["sharpe"] for r in records]
        if not records:
            return None, trial_sharpes, {"n_trials": 0, "rule": "one_se"}

        best = max(records, key=lambda r: r["sharpe"])
        se = sharpe_standard_error(
            best["sharpe"],
            len(best["returns"]),
            periods_per_year=ppy,
            skew=skewness(best["returns"]),
            kurt=kurtosis(best["returns"]),
        )
        threshold = best["sharpe"] - se_multiplier * se
        eligible = [r for r in records if r["sharpe"] >= threshold] or [best]

        if tie_break == "lowest_turnover":
            pick = min(eligible, key=lambda r: r["turnover"])
        else:
            for r in eligible:
                r["worst"] = worst_subwindow_sharpe(r["returns"], DEFAULT_SUBWINDOWS, ppy, rf)
            pick = max(eligible, key=lambda r: r["worst"])

        diagnostics = {
            "rule": f"one_se_{tie_break}",
            "n_trials": len(candidates),
            "n_eligible": len(eligible),
            "best_train_sharpe": best["sharpe"],
            "sharpe_se": se,
            "threshold": threshold,
            "picked": repr(pick["strategy"]),
            "picked_train_sharpe": pick["sharpe"],
        }
        return pick["strategy"], trial_sharpes, diagnostics

    return SelectionFn(f"one_se_{tie_break}", worker)
