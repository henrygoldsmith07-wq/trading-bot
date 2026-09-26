import json
from datetime import UTC, date, datetime

import pytest

from bot.algorithm import build_algorithm
from bot.prospective import (
    _coerce_session_date,
    _last_weekday,
    _observed_fixed_holiday,
    _validate_freeze_config,
    _validate_freeze_manifest,
    append_log,
    checkpoints_due,
    create_freeze,
    forward_day_kind,
    forward_performance,
    load_freeze,
    load_log,
    monthly_returns,
    outage_stats,
    run_step,
    slippage_stats,
    trailing_overlay_weight,
    us_equity_market_closed,
)
from bot.strategy import BuyHold, TrendVol, strategy_from_spec, strategy_to_spec

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
RUN_NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


def _algo(**overrides):
    return build_algorithm(with_pool_version=False, **overrides)


def _mk_freeze(tmp_path, strategies=None):
    assets = [
        {"symbol": "AAA", "source": "test", "periods_per_year": 365, "strategy": strategies["AAA"]},
        {"symbol": "BBB", "source": "test", "periods_per_year": 365, "strategy": strategies["BBB"]},
    ]
    manifest = create_freeze(
        assets,
        frictions={"fee": 0.001, "spread_bps": 5, "slippage_bps": 5, "execution": "next_open", "risk_free_annual": 0.03},
        algorithm=_algo(),
        path=tmp_path / "freeze.json",
        now=NOW,
        git_commit="abc123",
    )
    return manifest, tmp_path / "freeze.json"


def _candles(closes, day_of_first=None):
    if day_of_first is None:
        today_index = int(RUN_NOW.timestamp() * 1000 // 86_400_000)
        day_of_first = today_index - (len(closes) - 1)
    return [
        {"open_time": (day_of_first + i) * 86_400_000, "close": c} for i, c in enumerate(closes)
    ]


def _fetcher(data, problems=None):
    problems = problems or {}

    def f(sym, source):
        if sym in problems:
            return [], problems[sym]
        return data[sym], None

    return f


def test_freeze_roundtrip_and_tamper_detection(tmp_path):
    manifest, path = _mk_freeze(tmp_path, {"AAA": TrendVol(50, 20, 0.25), "BBB": TrendVol(100, 20, 0.4)})
    loaded = load_freeze(path)
    assert loaded["config_sha256"] == manifest["config_sha256"]
    assert loaded["frozen_at_date"] == NOW.date().isoformat()
    assert loaded["git_commit_at_freeze"] == "abc123"
    blob = json.loads(path.read_text())
    blob["config"]["assets"][0]["strategy"]["params"]["lookback"] = 42  # tamper
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        load_freeze(path)


def test_strategy_spec_roundtrip():
    s = TrendVol(75, 20, 0.3)
    s2 = strategy_from_spec(strategy_to_spec(s))
    assert repr(s2) == repr(s)
    with pytest.raises(ValueError):
        strategy_from_spec({"type": "NotAThing", "params": {}})


def test_append_log_restores_missing_record_separator(tmp_path):
    log = tmp_path / "log.jsonl"
    first = {"date": "2026-08-23", "port_ret": 0.0, "assets": {"AAA": {}}}
    second = {"date": "2026-08-24", "port_ret": 0.0, "assets": {"AAA": {}}}
    log.write_text(json.dumps(first), encoding="utf-8")

    append_log(second, log)

    entries = load_log(log)
    assert [entry["date"] for entry in entries] == ["2026-08-23", "2026-08-24"]


def test_freeze_rejects_duplicate_assets(tmp_path):
    from bot.algorithm import build_algorithm
    from bot.strategy import BuyHold

    assets = [
        {"symbol": "AAA", "source": "test", "periods_per_year": 365, "strategy": BuyHold()},
        {"symbol": "AAA", "source": "test", "periods_per_year": 365, "strategy": BuyHold()},
    ]
    with pytest.raises(ValueError, match="duplicate frozen asset"):
        create_freeze(
            assets=assets,
            frictions={"fee": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0,
                       "execution": "next_open", "risk_free_annual": 0.0},
            algorithm=build_algorithm(with_pool_version=False),
            path=tmp_path / "dup.json",
            now=NOW,
            git_commit="dup",
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda a, f: a[0].update({"periods_per_year": 0}), "periods_per_year"),
        (lambda a, f: a[0].update({"session": "moon_market"}), "unsupported session"),
        (lambda a, f: f.update({"fee": -0.01}), "non-negative"),
        (lambda a, f: f.update({"spread_bps": float("nan")}), "finite"),
        (lambda a, f: f.update({"execution": "close"}), "next_open"),
        (lambda a, f: f.update({"mystery_cost": 1.0}), "unknown friction"),
    ],
)
def test_freeze_rejects_invalid_asset_or_friction_contract(tmp_path, mutator, match):
    from bot.algorithm import build_algorithm
    from bot.strategy import BuyHold

    assets = [{"symbol": "AAA", "source": "test", "periods_per_year": 365,
               "strategy": BuyHold()}]
    frictions = {"fee": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0,
                 "execution": "next_open", "risk_free_annual": 0.0}
    mutator(assets, frictions)
    with pytest.raises(ValueError, match=match):
        create_freeze(
            assets=assets,
            frictions=frictions,
            algorithm=build_algorithm(with_pool_version=False),
            path=tmp_path / "invalid.json",
            now=NOW,
            git_commit="invalid",
        )


def test_load_freeze_rejects_timestamp_date_disagreement(tmp_path):
    manifest, path = _mk_freeze(tmp_path, {"AAA": BuyHold(), "BBB": BuyHold()})
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["frozen_at_date"] = "2026-08-21"
    raw["config_sha256"] = raw["config_sha256"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        load_freeze(path, verify_code=False)


def _valid_freeze_config_for_validation():
    return {
        "assets": [{
            "symbol": "AAA",
            "source": "test",
            "periods_per_year": 365,
            "session": "continuous",
            "strategy": strategy_to_spec(BuyHold()),
        }],
        "frictions": {
            "fee": 0.0,
            "spread_bps": 0.0,
            "slippage_bps": 0.0,
            "execution": "next_open",
            "risk_free_annual": 0.0,
        },
        "algorithm": _algo(),
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (None, "mapping"),
        (lambda c: c.pop("algorithm"), "missing required"),
        (lambda c: c.update({"assets": []}), "non-empty list"),
        (lambda c: c.update({"assets": ["bad"]}), "must be a mapping"),
        (lambda c: c["assets"][0].pop("source"), "missing key"),
        (lambda c: c["assets"][0].update({"symbol": ""}), "symbol must be non-empty"),
        (lambda c: c["assets"][0].update({"source": ""}), "source must be non-empty"),
        (lambda c: c["assets"][0].update({"strategy": {"type": "Nope", "params": {}}}), "invalid strategy"),
        (lambda c: c.update({"frictions": []}), "frictions must be a mapping"),
        (lambda c: c["frictions"].pop("fee"), "missing required key"),
        (lambda c: c["frictions"].update({"spread_bps": -1.0}), "spread_bps must be non-negative"),
        (lambda c: c["frictions"].update({"slippage_bps": -1.0}), "slippage_bps must be non-negative"),
        (lambda c: c.update({"overlay": "bad"}), "legacy overlay view must be a mapping"),
        (lambda c: c.update({"overlay": {"target_vol": 0.0}}), "target_vol must be positive"),
    ],
)
def test_freeze_config_validation_fail_closed(mutate, match):
    config = _valid_freeze_config_for_validation()
    candidate = [] if mutate is None else config
    if mutate is not None:
        mutate(config)
    with pytest.raises(ValueError, match=match):
        _validate_freeze_config(candidate)


def _valid_manifest_for_validation():
    config = _valid_freeze_config_for_validation()
    from bot.identity import CODE_FINGERPRINT_ALGO
    from bot.prospective import _config_hash

    return {
        "frozen_at": "2026-09-01T12:00:00+00:00",
        "frozen_at_date": "2026-09-01",
        "code_fingerprint_algo": CODE_FINGERPRINT_ALGO,
        "config_sha256": _config_hash(config),
        "code_sha256": "abc",
        "config": config,
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (None, "mapping"),
        (lambda m: m.pop("frozen_at"), "invalid frozen_at"),
        (lambda m: m.update({"frozen_at": "2026-09-01T12:00:00"}), "timezone-aware"),
        (lambda m: m.update({"code_fingerprint_algo": "old"}), "unknown code fingerprint"),
        (lambda m: m.update({"config_sha256": ""}), "missing config hash"),
        (lambda m: m.update({"code_sha256": ""}), "missing code fingerprint"),
        (lambda m: m.update({"config": []}), "config must be a mapping"),
    ],
)
def test_freeze_manifest_validation_fail_closed(mutate, match):
    manifest = _valid_manifest_for_validation()
    candidate = [] if mutate is None else manifest
    if mutate is not None:
        mutate(manifest)
    with pytest.raises(ValueError, match=match):
        _validate_freeze_manifest(candidate)


def test_load_freeze_rejects_unreadable_manifest(tmp_path):
    path = tmp_path / "bad-freeze.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_freeze(path, verify_code=False)


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (["{broken"], "JSON is corrupt"),
        (["[]"], "not an object"),
        ([json.dumps({"port_ret": 0.0, "assets": {"A": {}}})], "has no date"),
        ([json.dumps({"date": "bad", "port_ret": 0.0, "assets": {"A": {}}})], "invalid date"),
        ([json.dumps({"date": "2026-09-01", "port_ret": True, "assets": {"A": {}}})], "invalid port_ret"),
        ([json.dumps({"date": "2026-09-01", "port_ret": -1.1, "assets": {"A": {}}})], "impossible port_ret"),
        ([json.dumps({"date": "2026-09-01", "port_ret": 0.0, "assets": {}})], "no asset detail"),
        ([
            json.dumps({"date": "2026-09-01", "port_ret": 0.0, "assets": {"A": {}}}),
            json.dumps({"date": "2026-09-01", "port_ret": 0.0, "assets": {"A": {}}}),
        ], "repeats date"),
        ([
            json.dumps({"date": "2026-09-02", "port_ret": 0.0, "assets": {"A": {}}}),
            json.dumps({"date": "2026-09-01", "port_ret": 0.0, "assets": {"A": {}}}),
        ], "not strictly increasing"),
    ],
)
def test_load_log_rejects_corrupt_evidence(tmp_path, lines, match):
    path = tmp_path / "bad-log.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_log(path)



def test_run_step_refuses_same_day_as_freeze(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    res = run_step(
        manifest,
        _fetcher({"AAA": _mk_rising(400), "BBB": _mk_rising(400)}),
        now=NOW,
        log_path=log,
    )
    assert res["status"] == "not_after_freeze"
    assert not log.exists()


def test_run_step_logs_and_is_idempotent(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    res1 = run_step(manifest, _fetcher({"AAA": rising, "BBB": rising}), now=RUN_NOW, log_path=log)
    assert res1["status"] == "logged"
    res2 = run_step(manifest, _fetcher({"AAA": rising, "BBB": rising}), now=RUN_NOW, log_path=log)
    assert res2["status"] == "already_logged"
    assert len(load_log(log)) == 1


def test_order_lifecycle_timestamps_are_chronological(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    rising = _mk_rising(400)
    res = run_step(
        manifest,
        _fetcher({"AAA": rising, "BBB": rising}),
        now=RUN_NOW,
        log_path=tmp_path / "log.jsonl",
    )
    orders = res["entry"]["orders"]
    assert orders, "rising series should produce at least one simulated order"
    for order in orders:
        signal = datetime.fromisoformat(order["signal_generated_ts"])
        intent = datetime.fromisoformat(order["intent_ts"])
        submitted = datetime.fromisoformat(order["submitted_ts"])
        filled = datetime.fromisoformat(order["fill_ts"])
        assert (intent - signal).total_seconds() == pytest.approx(1.0)
        assert signal < intent <= submitted <= filled


def test_explicit_session_date_survives_runner_crossing_midnight(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    closed_session = _mk_rising(400)
    next_day_partial = dict(closed_session[-1])
    next_day_partial["open_time"] += 86_400_000
    next_day_partial["close"] *= 1.5
    feed = closed_session + [next_day_partial]
    delayed_now = datetime(2026, 8, 24, 0, 30, tzinfo=UTC)

    result = run_step(
        manifest,
        _fetcher({"AAA": feed, "BBB": feed}),
        now=delayed_now,
        session_date=date(2026, 8, 23),
        log_path=tmp_path / "log.jsonl",
    )

    assert result["entry"]["date"] == "2026-08-23"
    assert result["entry"]["session_date"] == "2026-08-23"
    assert result["entry"]["assets"]["AAA"]["price"] == pytest.approx(closed_session[-1]["close"])
    assert datetime.fromisoformat(result["entry"]["ts"]).date().isoformat() == "2026-08-24"


def test_forward_performance_preserves_partial_intervals_but_withholds_sharpe():
    entries = [
        {"date": "2026-09-07", "port_ret": 0.01, "assets": {"A": {"sleeve_ret": 0.01}}},
        {
            "date": "2026-09-08",
            "port_ret": 0.02,
            "assets": {"A": {"sleeve_ret": 0.02}, "B": {"note": "feed outage", "sleeve_ret": 0.0}},
            "outages": [{"symbol": "B", "problem": "feed outage"}],
        },
        {"date": "2026-09-09", "port_ret": -0.005, "assets": {"A": {"sleeve_ret": -0.005}}},
    ]
    perf = forward_performance(entries, freeze_date="2026-09-06", risk_free_annual=0.03)
    assert perf["return"] == pytest.approx(1.01 * 1.02 * 0.995 - 1.0)
    assert len(perf["curve"]) == 3
    assert perf["quality"]["partial"] == 1
    assert perf["return_quality"] == "degraded"
    assert perf["sharpe"] is None
    assert "partial/dark" in perf["sharpe_reason"]


def test_forward_sharpe_uses_observed_cadence_not_hardcoded_365():
    dates = [
        "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
        "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
        "2026-09-22",
    ]
    entries = [
        {"date": d, "port_ret": 0.001 if i % 2 == 0 else -0.0004, "assets": {"A": {"sleeve_ret": 0.0}}}
        for i, d in enumerate(dates)
    ]
    perf = forward_performance(entries, freeze_date="2026-09-06", risk_free_annual=0.03)
    assert perf["periods_per_year"] == pytest.approx(365.2425 * 11 / 16)
    assert perf["periods_per_year"] < 365
    assert perf["sharpe"] is not None


def test_us_equity_regular_holiday_calendar_matches_known_nyse_dates():
    assert us_equity_market_closed(date(2026, 9, 7)) is True   # Labor Day
    assert us_equity_market_closed(date(2027, 6, 18)) is True  # observed Juneteenth
    assert us_equity_market_closed(date(2027, 12, 31)) is False  # Jan 1 2028 is Saturday; no prior-Friday closure
    assert us_equity_market_closed(date(2028, 4, 14)) is True  # Good Friday
    assert us_equity_market_closed(date(2026, 9, 8)) is False


def test_session_date_coercion_accepts_supported_types_and_rejects_bad_values():
    fallback = date(2026, 9, 26)
    assert _coerce_session_date(None, fallback) == fallback
    assert _coerce_session_date(datetime(2026, 9, 25, 23, 0, tzinfo=UTC), fallback) == date(2026, 9, 25)
    assert _coerce_session_date(date(2026, 9, 24), fallback) == date(2026, 9, 24)
    assert _coerce_session_date("2026-09-23", fallback) == date(2026, 9, 23)
    with pytest.raises(ValueError, match="invalid session_date"):
        _coerce_session_date("not-a-date", fallback)
    with pytest.raises(ValueError, match="session_date must be"):
        _coerce_session_date(123, fallback)  # type: ignore[arg-type]


def test_calendar_helpers_cover_year_end_and_observed_fixed_holidays():
    assert _last_weekday(2026, 12, 0) == date(2026, 12, 28)
    assert _observed_fixed_holiday(7, 4, 2026) == date(2026, 7, 3)  # Saturday -> Friday
    assert _observed_fixed_holiday(7, 4, 2027) == date(2027, 7, 5)  # Sunday -> Monday
    assert _observed_fixed_holiday(7, 4, 2028) == date(2028, 7, 4)  # weekday unchanged
    assert us_equity_market_closed(date(2027, 1, 1)) is True
    assert us_equity_market_closed(date(2023, 1, 2)) is True  # Sunday New Year observed Monday


def test_forward_day_quality_handles_new_and_legacy_closed_labels_fail_closed():
    assert forward_day_kind({"date": "2026-09-07", "assets": {"SPY": {"note": "session_closed"}}}) == "closed"
    assert forward_day_kind({"date": "2026-09-07", "assets": {"SPY": {"note": "session_pending"}}}) == "closed"
    assert forward_day_kind({"date": "bad", "assets": {"SPY": {"note": "session_pending"}}}) == "dark"
    assert forward_day_kind({
        "date": "2026-09-07",
        "assets": {"BTC": {}, "SPY": {"note": "session_closed"}},
        "outages": [{"symbol": "SPY"}],
    }) == "partial"


def test_forward_performance_empty_dark_single_and_date_anchor_paths():
    empty = forward_performance([])
    assert empty["return"] is None and empty["return_quality"] == "none"

    dark = forward_performance([
        {"date": "2026-09-07", "port_ret": 0.0, "assets": {"A": {"note": "outage"}}},
        {"date": "2026-09-08", "port_ret": 0.0, "assets": {"A": {"note": "outage"}}},
    ])
    assert dark["return"] is None and dark["return_quality"] == "unmeasured"

    one = forward_performance([
        {"date": "2026-09-08", "port_ret": 0.01, "assets": {"A": {}}},
    ])
    assert one["return"] == pytest.approx(0.01)
    assert one["periods_per_year"] is None
    assert one["sharpe"] is None
    assert "fewer than two" in one["sharpe_reason"]

    pair = forward_performance([
        {"date": "2026-09-08", "port_ret": 0.01, "assets": {"A": {}}},
        {"date": "2026-09-10", "port_ret": -0.002, "assets": {"A": {}}},
    ], freeze_date=date(2026, 9, 7))
    assert pair["periods_per_year"] == pytest.approx(365.2425 * 2 / 3)
    assert pair["sharpe"] is not None


def _mk_rising(n, start=100.0, growth=1.004):
    return _candles([start * growth ** i for i in range(n)])


def test_run_step_records_outages(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    res = run_step(
        manifest,
        _fetcher({"AAA": rising, "BBB": rising}, problems={"BBB": "fetch failed: timeout"}),
        now=RUN_NOW,
        log_path=log,
    )
    entry = res["entry"]
    assert entry["outages"] == [{"symbol": "BBB", "problem": "fetch failed: timeout"}]
    assert outage_stats(load_log(log))["outage_events"] == 1


def test_run_step_flags_missed_fill_after_data_gap(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, path = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    # Remove two completed bars while keeping a valid current-day print:
    # the previous completed bar is still recent, but the last completed
    # transition spans three calendar days (two missed fills).
    closes = [100.0 * 1.004 ** i for i in range(300)]
    full = _candles(closes)
    candles = full[:-4] + full[-2:]
    res = run_step(manifest, _fetcher({"AAA": candles, "BBB": _mk_rising(400)}), now=RUN_NOW, log_path=log)
    mf = res["entry"]["missed_fills"]
    assert any(m["symbol"] == "AAA" and m["delayed_days"] >= 2 for m in mf)


def test_run_step_uses_only_completed_candles_for_decision(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    log = tmp_path / "log.jsonl"
    rising = _mk_rising(400)
    # Replace TODAY'S live mark with a wild price. It may affect marking, but
    # the target must use only completed bars ending yesterday.
    candles = [dict(row) for row in rising]
    candles[-1]["close"] = 1e9
    res = run_step(manifest, _fetcher({"AAA": candles, "BBB": rising}), now=RUN_NOW, log_path=log)
    det = res["entry"]["assets"]["AAA"]
    assert det["price"] == 1e9
    assert det["target"] == pytest.approx(TrendVol(10, 5, 0.5).weight(rising[:-1]))


def test_run_step_blocks_malformed_feed_without_crashing(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    bad = _mk_rising(20)
    bad[5] = "not-a-candle"
    res = run_step(
        manifest,
        _fetcher({"AAA": bad, "BBB": _mk_rising(20)}),
        now=RUN_NOW,
        log_path=tmp_path / "log.jsonl",
    )
    detail = res["entry"]["assets"]["AAA"]
    assert detail["weight"] == 0.0
    assert detail["note"].startswith("fail_closed:")
    assert any(o["symbol"] == "AAA" for o in res["entry"]["outages"])


def test_run_step_blocks_stale_continuous_history_even_with_current_mark(tmp_path):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    full = _mk_rising(40)
    # Keep history only through five days ago, then provide today's live mark.
    stale = full[:-5] + [full[-1]]
    res = run_step(
        manifest,
        _fetcher({"AAA": stale, "BBB": full}),
        now=RUN_NOW,
        log_path=tmp_path / "log.jsonl",
    )
    detail = res["entry"]["assets"]["AAA"]
    assert detail["weight"] == 0.0
    assert "stale completed history" in detail["note"]
    assert any(a["symbol"] == "AAA" and a["level"] == "stale_data" for a in res["entry"]["alerts"])


def test_run_step_nonfinite_strategy_target_holds_position(tmp_path, monkeypatch):
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    monkeypatch.setattr(TrendVol, "weight", lambda self, candles: float("nan"))
    res = run_step(
        manifest,
        _fetcher({"AAA": _mk_rising(40), "BBB": _mk_rising(40)}),
        now=RUN_NOW,
        log_path=tmp_path / "log.jsonl",
    )
    assert all(d["weight"] == 0.0 for d in res["entry"]["assets"].values())
    assert len([a for a in res["entry"]["alerts"] if a["level"] == "strategy_error"]) == 2


def test_slippage_stats_from_log():
    entries = [
        {"assets": {"A": {"slippage_bps": 10.0}, "B": {"slippage_bps": None}}, "port_ret": 0.0},
        {"assets": {"A": {"slippage_bps": -6.0}, "B": {"slippage_bps": 8.0}}, "port_ret": 0.0},
    ]
    s = slippage_stats(entries)
    assert s["count"] == 3
    assert s["mean_abs_bps"] == pytest.approx(8.0)


def test_checkpoints_due_gating():
    due = checkpoints_due(date(2026, 1, 1), date(2026, 2, 15))
    assert [c["due"] for c in due] == [True, False, False, False]
    due2 = checkpoints_due(date(2025, 1, 1), date(2026, 3, 1))
    assert [c["due"] for c in due2] == [True, True, True, True]


def test_monthly_returns_publish_negatives():
    entries = [
        {"date": "2026-06-01", "port_ret": 0.01},
        {"date": "2026-06-02", "port_ret": -0.02},
        {"date": "2026-07-01", "port_ret": 0.03},
    ]
    months = monthly_returns(entries)
    assert months["2026-06"] == pytest.approx(1.01 * 0.98 - 1)
    assert months["2026-06"] < 0
    assert months["2026-07"] == pytest.approx(0.03)


def test_trailing_overlay_weight_bounds():
    calm = [0.001] * 25
    wild = [0.05, -0.05] * 25
    assert trailing_overlay_weight(calm, 0.25) == 1.0
    assert 0.0 < trailing_overlay_weight(wild, 0.25) < 1.0
    assert trailing_overlay_weight([], 0.25) == 1.0  # warmup: fully invested


def test_forward_runner_never_reselects(tmp_path):
    # the frozen strategy spec is the only input; a tampered/unknown spec must fail
    strat = {"AAA": TrendVol(10, 5, 0.5), "BBB": TrendVol(10, 5, 0.5)}
    manifest, _ = _mk_freeze(tmp_path, strat)
    manifest["config"]["assets"][0]["strategy"] = {"type": "Mystery", "params": {}}
    with pytest.raises(ValueError):
        run_step(manifest, _fetcher({"AAA": _mk_rising(400), "BBB": _mk_rising(400)}), now=RUN_NOW, log_path=tmp_path / "log.jsonl")
