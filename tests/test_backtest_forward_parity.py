"""BACKTEST ↔ FORWARD PARITY — the identity test.

One fixed historical dataset, executed two ways:

  Path A  run_strategy(...) over the whole history (the validated engine)
  Path B  freeze -> run_step(day 1) -> run_step(day 2) -> ...  (the runner
          the scheduled paper job actually uses)

Required identities, per asset and per day, within numerical tolerance:

  target weights      (post-clamp, pre-band decision)
  portfolio weights   (post-band position actually held)
  trades              (turnover events and their sizes)
  costs               (fractional cost charged on each turnover)
  daily returns       (sleeve returns incl. overnight/intraday split + cash)
  equity              (compounded portfolio value, engine vs forward log)

Passing means: the engine that was validated historically IS the engine
being forward-tested. Any drift in execution accounting, banding, cost
handling or cash accrual breaks this file loudly.

If this test fails, DO NOT loosen tolerances. Fix the implementation.
"""
import json
import random
from datetime import UTC, datetime

import pytest

import bot.strategy as S
from bot.algorithm import build_algorithm
from bot.execution import calculate_transition
from bot.portfolio_rules import combine_portfolio_rule
from bot.prospective import create_freeze, load_log, run_step
from bot.strategy import WeightStrategy

# ---------------------------------------------------------------------------
# Fixed dataset & strategy (deterministic; no randomness anywhere)
# ---------------------------------------------------------------------------

BASE_MS = int(datetime(2026, 3, 2, tzinfo=UTC).timestamp() * 1000)
SYMS = ("AAA", "BBB", "CCC")
N_DAYS = 170  # price points; bars = N_DAYS - 1 decisions after the seed bar


def _dataset():
    """Fixed OHLC series with real overnight gaps (open != prev close)."""
    rng = random.Random(20260823)
    data = {}
    for k, s in enumerate(SYMS):
        p = 100.0 + 17.0 * k
        candles = []
        for i in range(N_DAYS):
            gap = rng.gauss(0.0006, 0.008)           # overnight component
            intraday = rng.gauss(0.0004, 0.014)      # open->close component
            o = p * (1 + gap)
            c = o * (1 + intraday)
            candles.append({
                "open_time": BASE_MS + i * 86_400_000,
                "open": round(o, 10),
                "high": round(max(o, c) * 1.002, 10),
                "low": round(min(o, c) * 0.998, 10),
                "close": round(c, 10),
            })
            p = c
        data[s] = candles
    return data


class Sawtooth(WeightStrategy):
    """Deterministic fine-ramp target: moves +0.05 per day through 0..1 then
    snaps back, with a per-asset phase so portfolio rules engage. The 0.05
    step sits exactly ON the default rebalance band, so Path A/B must agree
    on band semantics too: under band=0.05 most moves are suppressed and
    positions re-fire two levels apart; under band=0 every step trades."""

    def __init__(self, hold=1, phase=0):
        self.hold = hold
        self.phase = phase

    def weight_at(self, candles, i):
        step = ((i - self.phase) // self.hold) % 42
        level = step if step <= 20 else 40 - step
        return level / 20.0

    def __repr__(self):
        return f"Sawtooth({self.hold},{self.phase})"


@pytest.fixture()
def sawtooth_registry(monkeypatch):
    monkeypatch.setattr(S, "_STRATEGY_TYPES", {**S._STRATEGY_TYPES, "Sawtooth": Sawtooth})


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

FRICTIONS = {
    "fee": 0.001,
    "spread_bps": 5.0,
    "slippage_bps": 5.0,
    "execution": "next_open",
    "risk_free_annual": 0.03,
}
COST_RATE = FRICTIONS["fee"] + (FRICTIONS["spread_bps"] + FRICTIONS["slippage_bps"]) / 10_000.0
RF_DAILY = FRICTIONS["risk_free_annual"] / 365


def _algo(band):
    return build_algorithm(rebalance_band=band)


def _now_for(day_index):
    """Noon UTC of candle `day_index` — that candle is 'today' (live print);
    everything before it is completed history for the decision."""
    from datetime import datetime as dt

    return dt.fromtimestamp((BASE_MS + day_index * 86_400_000 + 12 * 3600) / 1000, tz=UTC)


TOL = 1e-12


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", [0.0, 0.05], ids=["zero_band", "banded_5pct"])
def test_engine_and_forward_are_the_same_bot(tmp_path, monkeypatch, sawtooth_registry, band):
    data = _dataset()
    algo = _algo(band)

    strategies = {s: Sawtooth(hold=4, phase=k) for k, s in enumerate(SYMS)}

    # ---------------- PATH A: the historical engine -------------------------
    path_a = {}
    for s in SYMS:
        path_a[s] = run_strategy_all(data[s], strategies[s].weight_at, band)
    n_bars = len(path_a[SYMS[0]]["returns"])

    # ---------------- PATH B: one run_step per historical day ---------------
    manifest = create_freeze(
        assets=[{"symbol": s, "source": "test", "periods_per_year": 365,
                 "strategy": strategies[s]} for s in SYMS],
        frictions=dict(FRICTIONS),
        algorithm=algo,
        path=tmp_path / f"freeze_{int(band * 1000)}.json",
        now=datetime(2026, 8, 23, tzinfo=UTC),
        git_commit="parity",
    )
    log_path = tmp_path / f"log_{int(band * 1000)}.jsonl"

    # Warmup seed: bar 1's state as the engine left it (the prospective
    # runner cannot trade before two completed candles exist, but the
    # portfolio it inherits ALREADY holds positions).
    seed_assets = {}
    for s in SYMS:
        seed_assets[s] = {
            "weight": path_a[s]["weights"][0],
            "target": path_a[s]["targets"][0],
            "sleeve_ret": path_a[s]["returns"][0],
            "note": None,
        }
    bar1_ts = BASE_MS + 86_400_000
    seed_rule = sum(seed_assets[s]["sleeve_ret"] for s in SYMS) / len(SYMS)
    log_path.write_text(json.dumps({
        "ts": datetime.fromtimestamp(bar1_ts / 1000, tz=UTC).isoformat(),
        "date": datetime.fromtimestamp(bar1_ts / 1000, tz=UTC).date().isoformat(),
        "assets": seed_assets,
        "port_ret": seed_rule,
        "rule_ret": seed_rule,
        "overlay_weight": 1.0,
        "exposure": 1.0,
        "throttled": False,
        "outages": [], "missed_fills": [],
    }) + "\n")

    fwd_by_day = {}
    for p in range(2, n_bars + 1):  # candle p is 'today'
        snapshot = {s: data[s][: p + 1] for s in SYMS}
        res = run_step(manifest, lambda sym, src, d=snapshot: (d[sym], None),
                       now=_now_for(p), log_path=log_path)
        entry = res["entry"]
        day_index = int((datetime.fromisoformat(entry["ts"]).timestamp() * 1000 - BASE_MS) // 86_400_000)
        fwd_by_day[day_index] = entry

    # ---------------- required identities -----------------------------------
    # Bars pair 1:1: engine bar index b (0-based over returns) == calendar day
    # b+1 == forward log day b+1.
    assert sorted(fwd_by_day) == list(range(2, n_bars + 1))

    total_cost_A = 0.0
    total_cost_B = 0.0
    trades_A = 0
    trades_B = 0

    for s in SYMS:
        A = path_a[s]
        for k in range(1, n_bars):  # bar 0 compared via the seed row itself
            day_index = k + 1
            B = fwd_by_day[day_index]["assets"][s]

            # --- target weights identical (post-clamp decision) ----------
            assert B["target"] == pytest.approx(A["targets"][k], abs=TOL), (s, k, "target")
            # --- portfolio weights identical (post-band held weight) -----
            assert B["weight"] == pytest.approx(A["weights"][k], abs=TOL), (s, k, "weight")

            # --- trades identical (turnover size and event flag) ---------
            turnover_B = abs(B["weight"] - fwd_prev_weight(log_path, s, day_index))
            assert turnover_B == pytest.approx(A["turnovers"][k], abs=TOL), (s, k, "turnover")
            if A["turnovers"][k] > 0:
                trades_A += 1
            if turnover_B > 0:
                trades_B += 1

            # --- costs identical -----------------------------------------
            cost_B = COST_RATE * turnover_B
            assert cost_B == pytest.approx(A["bar_costs"][k], abs=TOL), (s, k, "cost")
            total_cost_A += A["bar_costs"][k]
            total_cost_B += cost_B

            # --- daily returns identical (full transition decomposition) --
            expected_tr = calculate_transition(
                A["weights"][k - 1], A["weights"][k],
                previous_close=data[s][k]["close"],
                execution_price=open_of_candle(data[s][k + 1], data[s][k]["close"]),
                closing_price=data[s][k + 1]["close"],
                costs=COST_RATE if A["turnovers"][k] > 0 else 0.0,
                cash_rate_period=RF_DAILY,
                cash_basis="previous",
            )
            assert B["overnight"] == pytest.approx(expected_tr["overnight"], abs=TOL), (s, k, "overnight")
            assert B["intraday"] == pytest.approx(expected_tr["intraday"], abs=TOL), (s, k, "intraday")
            assert B["sleeve_ret"] == pytest.approx(expected_tr["return"], abs=TOL), (s, k, "sleeve")
            assert B["sleeve_ret"] == pytest.approx(A["returns"][k], abs=TOL), (s, k, "engine sleeve")

    # trade LEDGERS identical in count and total cost
    assert trades_A == trades_B
    assert total_cost_A == pytest.approx(total_cost_B, abs=TOL)

    # ---------------- portfolio-level equity identity -----------------------
    dailies = {s: {BASE_MS + (k + 1) * 86_400_000: path_a[s]["returns"][k] for k in range(n_bars)} for s in SYMS}
    timeline = sorted(dailies[SYMS[0]])
    ref_rule = combine_portfolio_rule(
        dailies, timeline, len(SYMS),
        vol_window=algo["weighting"]["vol_window"],
        max_multiple_of_equal=algo["weighting"]["max_multiple_of_equal"],
        use_tilt=algo["xs_momentum"]["enabled"], tilt_lookback=algo["xs_momentum"]["lookback"],
        max_tilt=algo["xs_momentum"]["max_tilt"],
        use_crisis=algo["crisis_derisk"]["enabled"], corr_window=algo["crisis_derisk"]["corr_window"],
        corr_threshold=algo["crisis_derisk"]["corr_threshold"], derisk=algo["crisis_derisk"]["multiplier"],
        use_dd_throttle=algo["drawdown_throttle"]["enabled"],
        dd_trigger=algo["drawdown_throttle"]["dd_trigger"],
        dd_exit=algo["drawdown_throttle"]["dd_exit"],
        throttle=algo["drawdown_throttle"]["factor"],
    )
    from bot.__main__ import _vol_overlay

    ref_final_rule = 1.0
    for r in ref_rule:
        ref_final_rule *= 1 + r
    entries = load_log(log_path)
    fwd_final_rule = 1.0
    for e in entries:
        fwd_final_rule *= 1 + e["rule_ret"]
    # rule layer (pre-overlay) identical
    assert fwd_final_rule == pytest.approx(ref_final_rule, rel=1e-12)

    # overlay applied identically -> final equity identical
    overlaid = _vol_overlay(ref_rule, target=algo["overlay"]["target_vol"])
    ref_equity = 1.0
    for r in overlaid:
        ref_equity *= 1 + r
    fwd_equity = 1.0
    for e in entries:
        fwd_equity *= 1 + e["port_ret"]
    assert fwd_equity == pytest.approx(ref_equity, rel=1e-12), (fwd_equity, ref_equity)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run_strategy_all(candles, weight_fn, band):
    """Path A wrapper: full-history engine pass with per-bar audit lists."""
    from bot.engine import run_strategy

    res = run_strategy(
        candles, weight_fn,
        fee=FRICTIONS["fee"], spread_bps=FRICTIONS["spread_bps"],
        slippage_bps=FRICTIONS["slippage_bps"], execution="next_open",
        risk_free_annual=FRICTIONS["risk_free_annual"],
        rebalance_band=band,
    )
    turnovers = [abs(res["weights"][i] - (res["weights"][i - 1] if i else 0.0))
                 for i in range(len(res["weights"]))]
    res["turnovers"] = turnovers
    return res


def open_of_candle(candle, fallback):
    from bot.execution import open_of

    return open_of(candle, fallback)


def fwd_prev_weight(log_path, symbol, before_day_index):
    """Forward weight HELD coming into `before_day_index` (seed row counts)."""
    entries = load_log(log_path)
    prior = [e for e in entries
             if datetime.fromisoformat(e["ts"]).timestamp() * 1000 < BASE_MS + before_day_index * 86_400_000]
    return prior[-1]["assets"][symbol]["weight"]
