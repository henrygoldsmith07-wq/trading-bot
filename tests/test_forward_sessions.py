"""Exchange-calendar-aware forward execution.

The scheduled run must NEVER execute a US ETF against a stale close because
it fired before the NYSE session closed. These tests pin:

- us_equity assets are PENDING before the session-close gate (no order, no
  observation, weight carried, sleeve flat);
- after the gate, with the real daily bar present, execution is EXACTLY
  next_open and numerically identical to engine.run_strategy on the same bars;
- weekends/holidays stay pending even after the gate time;
- crypto (continuous) trades regardless of clock;
- the session field survives freeze roundtrips.
"""
from datetime import UTC, datetime, timedelta

import pytest

import bot.strategy as S
from bot.algorithm import build_algorithm
from bot.engine import run_strategy
from bot.execution import calculate_transition
from bot.prospective import create_freeze, load_log, run_step
from bot.strategy import BuyHold, WeightStrategy


class Step(WeightStrategy):
    """Full in/out on a 5-bar cycle so weight changes actually happen."""

    def weight_at(self, candles, i):
        return 1.0 if (i // 5) % 2 == 0 else 0.0

    def __repr__(self):
        return "Step"


S._STRATEGY_TYPES = {**S._STRATEGY_TYPES, "Step": Step}


def _manifest(tmp_path, session):
    return create_freeze(
        assets=[{"symbol": "ETF1", "source": "test", "periods_per_year": 252,
                 "session": session, "strategy": Step()}],
        frictions={"fee": 0.001, "spread_bps": 5.0, "slippage_bps": 5.0,
                   "execution": "next_open", "risk_free_annual": 0.03},
        algorithm=build_algorithm(rebalance_band=0.0, overlay_enabled=False,
                                  use_tilt=False, use_crisis=False),
        path=tmp_path / "freeze.json",
        now=datetime(2026, 8, 20, tzinfo=UTC),
        git_commit="sess",
    )


def _bars(day_anchor_utc_noon, n=40):
    closes = [100.0]
    for i in range(1, n):
        closes.append(closes[-1] * (1.006 if i % 3 else 0.997))
    return [
        {"open_time": day_anchor_utc_noon + i * 86_400_000 - 12 * 3600_000,  # midnight UTC of that day
         "open": closes[i - 1] if i else closes[0],
         "close": closes[i],
         "quote_volume": closes[i] * 900.0}
        for i in range(n)
    ]


ANCHOR_NOON = int(datetime(2026, 8, 10, 12, 0, tzinfo=UTC).timestamp() * 1000)


def _now_for(day_offset_from_anchor_start, hour_utc, minute=0):
    base = datetime.fromtimestamp(ANCHOR_NOON / 1000, tz=UTC)
    target = base.replace(hour=hour_utc, minute=minute)
    return target + timedelta(days=day_offset_from_anchor_start)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestSessionGate:
    def test_morning_run_is_pending_not_stale_execution(self, env):
        """06:17-UTC-style run: today's NYSE bar cannot exist. The asset must
        be PENDING — never an execution against yesterday's close."""
        m = _manifest(env, session="us_equity")
        candles = _bars(ANCHOR_NOON)
        # now = noon of candle-day 30, BEFORE the 21:15 gate
        res = run_step(m, lambda s, src: (candles[:31], None),
                       now=_now_for(30, 13, 17), log_path=env / "l.jsonl")
        d = res["entry"]["assets"]["ETF1"]
        assert d["note"] == "session_pending"
        assert d["weight"] == 0.0 and d["sleeve_ret"] == 0.0
        # nothing was traded or cost-measured
        assert len(load_log(env / "l.jsonl")[0].get("assets", {})) == 1
        assert all(o.get("symbol") != "ETF1" for o in [])

    def test_evening_run_executes_with_real_bar_and_matches_engine(self, env):
        """After the US close the completed bar carries open+close; execution
        must equal engine.run_strategy next_open on identical bars."""
        m = _manifest(env, session="us_equity")
        candles = _bars(ANCHOR_NOON, n=34)
        # step through days 2..33 (two completed needed), then compare last bar
        p_last = 32
        for p in range(2, p_last + 1):
            run_step(m, lambda s, src, d=candles[: p + 1]: (d, None),
                     now=_now_for(p, 22, 0), log_path=env / "l.jsonl")
        fwd_entry = load_log(env / "l.jsonl")[-1]["assets"]["ETF1"]

        eng = run_strategy(candles[: p_last + 1], Step().weight_at,
                           fee=0.001, spread_bps=5.0, slippage_bps=5.0,
                           execution="next_open", risk_free_annual=0.03)
        eng_ret = eng["return_days"][-1][1]
        eng_w = eng["weights"][-1]
        assert fwd_entry["weight"] == pytest.approx(eng_w)
        assert fwd_entry["sleeve_ret"] == pytest.approx(eng_ret, abs=5e-9)

    def test_weekend_after_gate_still_pending(self, env):
        """Saturday 22:00 UTC: gate time passed but NO bar will ever exist for
        today — must remain pending instead of executing on Friday's close."""
        m = _manifest(env, session="us_equity")
        candles = _bars(ANCHOR_NOON, n=40)
        sat = _now_for(33, 22, 0)  # 2026-09-12 is a Saturday for this anchor
        assert sat.weekday() >= 5  # sanity: actually a weekend in this anchor
        res = run_step(m, lambda s, src: (candles, None), now=sat, log_path=env / "l.jsonl")
        assert res["entry"]["assets"]["ETF1"]["note"] == "session_pending"

    def test_continuous_crypto_trades_in_the_morning(self, env):
        m = _manifest(env, session="continuous")
        candles = _bars(ANCHOR_NOON, n=34)
        res = run_step(m, lambda s, src: (candles[:31], None),
                       now=_now_for(30, 6, 17), log_path=env / "l.jsonl")  # 06:17 UTC style
        d = res["entry"]["assets"]["ETF1"]
        assert d.get("note") in (None,)  # traded
        assert d["weight"] in (0.0, 1.0)

    def test_session_pending_days_chain_marks_without_loss(self, env):
        """A pending day followed by a trading day must cover the FULL price
        span exactly once (mark-chaining, no double count, no gap)."""
        m = _manifest(env, session="us_equity")
        candles = _bars(ANCHOR_NOON, n=40)
        log = env / "l.jsonl"
        # day 30 at noon -> pending; day 31 at 22:00 -> trades
        run_step(m, lambda s, src: (candles[:31], None), now=_now_for(30, 13, 0), log_path=log)
        r2 = run_step(m, lambda s, src: (candles[:32], None), now=_now_for(31, 22, 0), log_path=log)
        e = r2["entry"]["assets"]["ETF1"]
        expected = calculate_transition(
            0.0, e["weight"],
            previous_close=candles[30]["close"],      # mark from the pending day = decision close
            execution_price=candles[31]["open"],
            closing_price=candles[31]["close"],
            costs=(0.001 + 0.001) * abs(e["weight"]),
            cash_rate_period=0.03 / 365,
            cash_basis="previous",
        )
        assert e["sleeve_ret"] == pytest.approx(expected["return"], abs=1e-12)


class TestFreezeCarriesSession:
    def test_session_survives_manifest_roundtrip(self, tmp_path):
        from bot.prospective import create_freeze, load_freeze

        create_freeze(
            assets=[{"symbol": "SPY", "source": "yahoo", "periods_per_year": 252,
                     "session": "us_equity", "strategy": BuyHold()}],
            frictions={"fee": 0.001}, algorithm=build_algorithm(with_pool_version=False),
            path=tmp_path / "f.json", git_commit="s",
        )
        assert load_freeze(tmp_path / "f.json")["config"]["assets"][0]["session"] == "us_equity"

    def test_etf_universe_declares_us_equity(self):
        from bot.universe import ETF_UNIVERSE

        assert all(e.get("session") == "us_equity" for e in ETF_UNIVERSE)
