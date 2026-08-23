"""Point-in-time universe: the survivorship fix.

Proves that eligibility is computed from each asset's OWN past (a 2023
listing cannot join a 2021 portfolio; an illiquid-then asset stays out
however famous it is now), that per-day denominators flow into every
combiner, and that forward snapshots compound into a real PIT dataset.
"""
import json
from datetime import UTC, datetime

import pytest

from bot.portfolio_rules import combine_portfolio_rule
from bot.universe_pit import (
    DAY_MS,
    eligible_on,
    load_snapshots,
    mean_daily_quote_volume,
    point_in_time_universe,
    record_snapshot,
)
from bot.walkforward import combine_portfolio, combine_portfolio_invvol


def _candles(first_day, n_days, quote_volume=20_000_000.0):
    return [
        {"open_time": DAY_MS * (first_day + i), "close": 100.0 + i, "quote_volume": quote_volume}
        for i in range(n_days)
    ]


class TestEligibility:
    def test_listing_age_required(self):
        # listed at day 100 with only 10 days of bars by day 105 -> too young
        candles = _candles(100, 10)
        assert not eligible_on(candles, DAY_MS * 105, min_history_days=90)

    def test_mature_liquid_asset_eligible(self):
        candles = _candles(0, 200)  # listed day 0, alive through day 199
        assert eligible_on(candles, DAY_MS * 150)

    def test_future_dates_not_eligible_before_existence(self):
        candles = _candles(500, 50)
        assert not eligible_on(candles, DAY_MS * 400)

    def test_dead_asset_exits(self):
        candles = _candles(0, 100)  # last bar covers day 99
        assert not eligible_on(candles, DAY_MS * 300)   # long dead
        assert eligible_on(candles, DAY_MS * 100)       # still fresh then

    def test_low_trailing_volume_blocks_even_famous_today(self):
        candles = _candles(0, 200, quote_volume=100_000.0)  # illiquid THEN
        assert not eligible_on(candles, DAY_MS * 150, min_mean_daily_quote_volume=5_000_000.0)

    def test_no_volume_field_does_not_block_etf_style_feeds(self):
        candles = [{k: v for k, v in c.items() if k != "quote_volume"} for c in _candles(0, 150)]
        assert eligible_on(candles, DAY_MS * 140)


class TestMeanDailyQuoteVolume:
    def test_strictly_past_window_only(self):
        candles = _candles(0, 40, quote_volume=1_000.0)
        cut = DAY_MS * 35
        prior = [c["quote_volume"] for c in candles if c["open_time"] < cut][-30:]
        assert mean_daily_quote_volume(candles, cut, window=30) == pytest.approx(sum(prior) / len(prior))


class TestPITUniverse:
    def test_staggered_life_cycles(self):
        """EARLY lives the whole span; DEAD dies early; LATE joins only after
        its own listing + minimum history — regardless of being famous now."""
        hist = {
            "EARLY": _candles(0, 500),
            "LATE": _candles(300, 200),   # lists at day 300
            "DEAD": _candles(0, 120),     # last bar day 119
        }
        timeline = [DAY_MS * d for d in (95, 150, 250, 395)]
        pit = point_in_time_universe(hist, timeline)
        assert pit[timeline[0]] == {"EARLY", "DEAD"}   # LATE unlisted yet
        assert pit[timeline[1]] == {"EARLY"}           # DEAD gone; LATE too young
        assert pit[timeline[2]] == {"EARLY"}
        assert pit[timeline[3]] == {"EARLY", "LATE"}   # LATE finally holdable

    def test_denominators_match_eligibility_counts(self):
        hist = {"A": _candles(0, 500), "B": _candles(300, 200)}
        timeline = [DAY_MS * d for d in (95, 395)]
        pit = point_in_time_universe(hist, timeline)
        assert [len(pit[t]) for t in timeline] == [1, 2]


class TestCombinerDenominators:
    def _streams(self):
        # A trades every day with +1%; B joins later with +2% days
        days_a = [DAY_MS * i for i in range(1, 60)]
        days_b = [DAY_MS * i for i in range(30, 60)]
        return (
            {"A": dict.fromkeys(days_a, 0.01), "B": dict.fromkeys(days_b, 0.02)},
            sorted(days_a),
        )

    def test_fixed_denominator_default_unchanged(self):
        streams, timeline = self._streams()
        out = combine_portfolio(streams, timeline, n_assets=2)
        assert all(r == pytest.approx(0.005 if t < DAY_MS * 30 else 0.015) for r, t in zip(out, timeline))

    def test_pit_denominator_scales_by_eligible_count(self):
        streams, timeline = self._streams()
        denom = {t: (1 if t < DAY_MS * 30 else 2) for t in timeline}
        out = combine_portfolio(streams, timeline, n_assets=2, denominator_by_day=denom)
        assert out[0] == pytest.approx(0.01)          # A alone / 1 eligible
        assert out[-1] == pytest.approx(0.015)        # both / 2 eligible

    def test_invvol_and_rule_accept_denominators(self):
        streams, timeline = self._streams()
        denom = {t: (1 if t < DAY_MS * 30 else 2) for t in timeline}
        iv = combine_portfolio_invvol(streams, timeline, 2, denominator_by_day=denom)
        rule = combine_portfolio_rule(streams, timeline, 2, use_tilt=False, use_crisis=False,
                                      denominator_by_day=denom)
        assert iv[0] == pytest.approx(0.01)
        assert rule[0] == pytest.approx(0.01)
        assert iv[-1] == pytest.approx(0.015)


class TestSnapshots:
    def test_record_and_load_roundtrip(self, tmp_path):
        log = tmp_path / "universe_log.jsonl"
        r1 = record_snapshot([("BTC", 9e9), ("XYZ", 4e8)], log_path=log,
                             now=datetime(2026, 8, 23, tzinfo=UTC))
        assert r1["status"] == "logged"
        r2 = record_snapshot([("BTC", 9e9), ("XYZ", 4e8)], log_path=log,
                             now=datetime(2026, 8, 23, tzinfo=UTC))  # same date: idempotent
        assert r2["status"] == "already_logged"
        snaps = load_snapshots(log)
        assert snaps["2026-08-23"] == ["BTC", "XYZ"]

    def test_second_day_appends(self, tmp_path):
        log = tmp_path / "u.jsonl"
        record_snapshot([("BTC", 1e10)], log_path=log, now=datetime(2026, 8, 23, tzinfo=UTC))
        record_snapshot([("BTC", 1e10), ("NEW", 3e8)], log_path=log, now=datetime(2026, 8, 24, tzinfo=UTC))
        snaps = load_snapshots(log)
        assert snaps["2026-08-24"] == ["BTC", "NEW"]

    def test_torn_line_tolerated(self, tmp_path):
        p = tmp_path / "u.jsonl"
        e = {"date": "2026-08-23", "generated_at": "t", "source": "s",
             "universe": [{"symbol": "BTC", "quote_volume_usd": 1.0}]}
        p.write_text(json.dumps(e) + "\n{'date': torn")
        assert load_snapshots(p) == {"2026-08-23": ["BTC"]}
