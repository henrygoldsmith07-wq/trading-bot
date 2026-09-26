import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _load_api():
    spec = importlib.util.spec_from_file_location("api_summary", Path(__file__).resolve().parents[1] / "api" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api():
    return _load_api()


def _fake_candles(n=1600, start=100.0):
    out = []
    px = start
    for i in range(n):
        px *= 1.003 if i > 300 else 1.0
        out.append({"open_time": 1_500_000_000_000 + i * 86_400_000, "close": round(px, 8)})
    return out


def test_build_summary_shape(monkeypatch, api, tmp_path):
    # isolate from the committed canonical record so legacy shape holds
    monkeypatch.setattr(api, "CANONICAL_RUN", str(tmp_path / "missing" / "run.json"))
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    s = api.build_summary("BTCUSDT")
    assert s["symbol"] == "BTCUSDT"
    assert "RiskEnsemble" in s["strategy"]
    assert -1.0 <= s["oos"]["max_drawdown"] <= 0.0
    assert 0.0 <= s["now"]["weight_now"] <= 1.0
    assert isinstance(s["now"]["trend_up"], bool)
    assert 2 <= len(s["curve"]) <= 200
    assert s["curve"][-1]["v"] == pytest.approx(s["oos"]["final"], rel=1e-3)
    assert "paper" in s["disclaimer"].lower()
    ts = [p["t"] for p in s["curve"]]
    assert ts == sorted(ts)


def test_payload_never_uses_the_word_live(monkeypatch, api):
    """The UI taxonomy is research / out-of-sample / forward. A key or label
    called 'live' would leak a fourth, unearned evidence class."""
    monkeypatch.setattr(api, "CANONICAL_RUN", str(Path(__file__).resolve().parents[1] / "nope" / "run.json"))
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    s = api.build_summary("BTCUSDT")
    assert "live" not in s
    assert "live" not in json.dumps(s).lower()


def test_build_summary_prefers_canonical_record(monkeypatch, api):
    """When runs/canonical-v1 exists, headline NUMBERS come from the sealed
    record — same source as the README table — while curve/current stay fresh."""
    import os

    if not os.path.exists(api.CANONICAL_RUN):
        pytest.skip("canonical run not generated yet")
    # Stub the history fetch, as the neighbouring tests do. Every assertion
    # below reads from the sealed record, so this changes nothing about what
    # is being checked — but without it the test reaches out to a live
    # external API, which returns HTTP 451 to CI runner IPs and turns the
    # suite red for reasons that have nothing to do with the code.
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    s = api.build_summary("BTCUSDT")
    record = json.loads(open(api.CANONICAL_RUN, encoding="utf-8").read())
    iv = record["results"]["metrics"]["inv_vol_rm"]
    assert s["oos"]["sharpe"] == pytest.approx(iv["sharpe"], abs=1e-6)
    assert s["oos"]["cagr"] == pytest.approx(iv["cagr"], abs=1e-9)
    assert "canonical-v1" in s["canonical_run_id"]
    assert len(s.get("rules_table", [])) >= 1
    # curve remains the live-computed research curve (may differ from record)
    assert len(s["curve"]) >= 2


def test_asgi_app_serves_json(monkeypatch, api):
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/api/summary"}
    asyncio.run(api.app(scope, receive, send))

    start = [m for m in sent if m["type"] == "http.response.start"][0]
    body = [m for m in sent if m["type"] == "http.response.body"][0]
    assert start["status"] == 200
    headers = dict((k.decode(), v.decode()) for k, v in start["headers"])
    assert headers["content-type"] == "application/json"
    payload = json.loads(body["body"])
    # evidence-first shape: forward block always present, research nested below
    assert "forward" in payload and "research" in payload
    assert payload["research"].get("evidence_label", "").startswith("RESEARCH / HISTORICAL ONLY")
    assert "oos" in payload["research"]


def test_asgi_app_research_failure_degrades_not_500(monkeypatch, api):
    def boom(symbol):
        raise RuntimeError("network down")

    monkeypatch.setattr(api, "fetch_daily_history", boom)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(api.app({"type": "http", "method": "GET", "path": "/api/summary"}, receive, send))
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    assert start["status"] == 200  # forward evidence still served
    payload = json.loads([m for m in sent if m["type"] == "http.response.body"][0]["body"])
    assert "network down" in payload["research"]["error"]
    fwd = payload["forward"]
    # repo-root freeze.json absent in this test env -> unavailable; if a real
    # freeze exists, it must at least be intact
    assert fwd["available"] is False or fwd["code_verified"] is True


# ---------------------------------------------------------------------------
# forward (evidence-first) summary
# ---------------------------------------------------------------------------

FREEZE_TS = "2026-08-23T09:00:00+00:00"


def _write_freeze(path: Path, frozen_date="2026-08-23", commit="6d6606dabc"):

    from bot.identity import code_fingerprint
    from bot.prospective import _config_hash

    config = {"assets": [{"symbol": "BTCUSDT"}], "frictions": {"fee": 0.001}}
    cfg_sha = _config_hash(config)
    path.write_text(json.dumps({
        "frozen_at": FREEZE_TS,
        "frozen_at_date": frozen_date,
        "git_commit_at_freeze": commit,
        "config": config,
        "config_sha256": cfg_sha,
        "code_fingerprint_algo": "sha256-lf-v1",
        "code_sha256": code_fingerprint(),
    }))


def _write_log(path: Path, days=10, first="2026-08-24", ret=0.001,
               n_assets=3, outage_day=3, pending_day=None, dark_day=None):
    """Forward log with per-sleeve detail, so day classification is exercised.

    Every day carries `assets`; a `note` on a sleeve means that sleeve did not
    print (outage) or was held (session_pending). `dark_day` marks every sleeve
    as noted — the zero-information case.
    """
    from datetime import timedelta

    d0 = datetime.fromisoformat(first)
    with open(path, "w") as f:
        for i in range(days):
            assets = {}
            for a in range(n_assets):
                sym = f"S{a}"
                if dark_day is not None and i == dark_day:
                    assets[sym] = {"note": "outage", "sleeve_ret": 0.0}
                elif outage_day is not None and i == outage_day and a == 0:
                    assets[sym] = {"note": "outage", "sleeve_ret": 0.0}
                elif pending_day is not None and i == pending_day and a == 0:
                    assets[sym] = {"note": "session_pending", "sleeve_ret": 0.0}
                else:
                    assets[sym] = {"sleeve_ret": 0.001}
            outages = ([{"symbol": "S0"}] if outage_day is not None and i == outage_day else [])
            if dark_day is not None and i == dark_day:
                outages = [{"symbol": f"S{a}"} for a in range(n_assets)]
            f.write(json.dumps({
                "date": (d0 + timedelta(days=i)).date().isoformat(),
                "port_ret": ret * (-1 if i % 5 == 4 else 1),
                "assets": assets,
                "outages": outages,
                "missed_fills": [{"s": 1}] if i == 6 else [],
            }) + "\n")


def test_order_lifecycle_warning_detects_metadata_without_rewriting_returns(api):
    entries = [{
        "date": "2026-09-07",
        "orders": [{
            "symbol": "BTCUSDT",
            "side": "BUY",
            "signal_generated_ts": "2026-09-06T23:59:59+00:00",
            "intent_ts": "2029-06-02T00:00:00+00:00",
            "submitted_ts": "2026-09-07T23:47:10+00:00",
            "fill_ts": "2026-09-07T23:47:10+00:00",
            "fill_price": 100.0,
        }],
    }]
    warning = api._order_lifecycle_warnings(entries)
    assert warning["count"] == 1
    assert warning["reasons"]["non_monotonic_timestamps"] == 1


def test_clean_order_lifecycle_has_no_warning(api):
    entries = [{
        "date": "2026-09-07",
        "orders": [{
            "symbol": "BTCUSDT",
            "side": "BUY",
            "signal_generated_ts": "2026-09-06T23:59:59+00:00",
            "intent_ts": "2026-09-07T00:00:00+00:00",
            "submitted_ts": "2026-09-07T21:45:00+00:00",
            "fill_ts": "2026-09-07T21:45:00+00:00",
            "fill_price": 100.0,
        }],
    }]
    assert api._order_lifecycle_warnings(entries) == {"count": 0, "reasons": {}}


def test_order_lifecycle_allows_next_day_wall_clock_fill_for_prior_session(api):
    entries = [{
        "date": "2026-09-07",
        "orders": [{
            "symbol": "BTCUSDT",
            "side": "BUY",
            "signal_generated_ts": "2026-09-06T23:59:59+00:00",
            "intent_ts": "2026-09-07T00:00:00+00:00",
            "submitted_ts": "2026-09-08T00:30:00+00:00",
            "fill_ts": "2026-09-08T00:30:00+00:00",
            "fill_price": 100.0,
        }],
    }]
    assert api._order_lifecycle_warnings(entries) == {"count": 0, "reasons": {}}


class TestForwardSummary:
    def test_unavailable_without_freeze(self, api, tmp_path):
        res = api.build_forward_summary(freeze_path=str(tmp_path / "none.json"))
        assert res["available"] is False
        assert "freeze" in res["reason"]

    def test_started_false_before_first_day(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(tmp_path / "log.jsonl"))
        assert res["available"] and not res["started"]
        assert res["parameter_changes"] == 0
        assert res["days_untouched"] is None

    def test_full_metrics_from_log(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=12)
        today = datetime(2026, 9, 20, tzinfo=UTC)  # 28 days after freeze
        res = api.build_forward_summary(
            freeze_path=str(fp), log_path=str(lp),
            benchmark_fetch=lambda: [
                {"date": "2026-08-24", "close": 5000.0},
                {"date": "2026-09-04", "close": 5150.0},
            ],
            today=today,
        )
        assert res["available"] and res["started"]
        assert res["frozen_date"] == "2026-08-23"
        assert res["code_verified"] is True
        assert res["days_untouched"] == 28
        assert res["parameter_changes"] == 0
        assert res["n_days_recorded"] == 12
        assert res["data_outages"] == 1
        # one sleeve dark on day 3 -> 11 days fully observed, 1 partial
        assert res["days_full"] == 11
        assert res["days_partial"] == 1
        assert res["days_dark"] == 0
        assert res["data_outage_days"] == 1
        assert res["data_outage_events"] == 1
        assert res["missed_fills"] == 1
        assert res["benchmark_return"] == pytest.approx(0.03)
        assert -1.0 <= res["max_drawdown"] <= 0.0
        # The curve compounds the append-only tape exactly as written. A
        # partial row cannot simply be deleted because later sleeves chain from
        # prior marks; removing it would remove a real interval from the path.
        entries = [json.loads(ln) for ln in lp.read_text().splitlines() if ln.strip()]
        assert len(res["curve"]) == 12
        assert [c["t"] for c in res["curve"]] == [e["date"] for e in entries]
        eq = 1.0
        for i, e in enumerate(entries):
            eq *= 1.0 + e["port_ret"]
            assert res["curve"][i]["v"] == pytest.approx(round(eq, 5), abs=1e-6)
        assert res["forward_return"] == pytest.approx(eq - 1, abs=1e-4)
        assert res["forward_return_quality"] == "degraded"
        assert res["forward_sharpe"] is None
        assert "partial/dark" in res["forward_sharpe_reason"]

    def test_partial_only_tape_preserves_path_but_withholds_sharpe(self, api, tmp_path):
        """Partial marks stay in the cumulative path but cannot support Sharpe."""
        from datetime import timedelta

        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        # Every day is partly observed: one sleeve in real outage on each of
        # the five scheduled days, so no day has all sleeves printing. A real
        # fetch failure — NOT 'session_pending', which would mean the session
        # simply never opened and would be classed as a closed market.
        d0 = datetime.fromisoformat("2026-08-24")
        rows = [
            {
                "date": (d0 + timedelta(days=i)).date().isoformat(),
                "port_ret": 0.001,
                "assets": {"S0": {"note": "fetch failed: HTTP Error 451: ", "sleeve_ret": 0.0},
                           "S1": {"sleeve_ret": 0.001},
                           "S2": {"sleeve_ret": 0.001}},
                "outages": [{"symbol": "S0", "problem": "fetch failed: HTTP Error 451: "}],
                "missed_fills": [],
            }
            for i in range(5)
        ]
        lp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        res = api.build_forward_summary(
            freeze_path=str(fp), log_path=str(lp),
            benchmark_fetch=lambda: [], today=datetime(2026, 9, 20, tzinfo=UTC),
        )
        assert res["started"] is True
        assert res["n_days_recorded"] == 5   # scheduled days: still disclosed
        assert res["days_full"] == 0
        assert res["days_partial"] == 5
        assert res["days_closed"] == 0       # a failure, not a shut exchange
        assert res["forward_return"] == pytest.approx((1.001 ** 5) - 1, abs=1e-4)
        assert res["forward_sharpe"] is None
        assert res["forward_return_quality"] == "degraded"
        assert res["max_drawdown"] == pytest.approx(0.0)
        assert len(res["curve"]) == 5
        assert res["first_day"] == "2026-08-24"

    def test_closed_market_day_is_neither_evidence_nor_outage(self, api, tmp_path):
        """A shut exchange is not a broken feed.

        Three of thirteen sleeves are US ETFs, so on a weekend they are
        structurally `session_pending`. Booking those days as data outages
        would put ~29% of scheduled days above the 20% threshold and withhold
        the verdict grade forever — the experiment would read as broken
        because of a calendar. The day is also not evidence: the sleeves that
        did print produced a return for part of the portfolio only.
        """
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=5, first="2026-08-29", outage_day=None, pending_day=0)
        res = api.build_forward_summary(
            freeze_path=str(fp), log_path=str(lp),
            benchmark_fetch=lambda: [], today=datetime(2026, 9, 20, tzinfo=UTC),
        )
        assert res["days_full"] == 4
        assert res["days_closed"] == 1
        assert res["days_partial"] == 0
        assert res["days_dark"] == 0
        # counted as neither: not evidence, and not an outage
        assert res["data_outage_days"] == 0
        assert res["data_outage_events"] == 0
        assert res["n_days_recorded"] == 5      # still disclosed as scheduled
        assert len(res["curve"]) == 5

    def test_real_failure_alongside_pending_session_is_still_an_outage(self, api):
        """One sleeve waiting on a session does not excuse a genuine failure
        on another. `closed` requires that NOTHING broke."""
        entry = {
            "date": "2026-08-24",
            "port_ret": 0.0,
            "assets": {
                "SPY": {"note": "session_pending", "sleeve_ret": 0.0, "session": "us_equity"},
                "GLD": {"note": "fetch failed: HTTP Error 451: ", "sleeve_ret": 0.0},
                "BTCUSDT": {"sleeve_ret": 0.001},
            },
            "outages": [{"symbol": "GLD", "problem": "fetch failed: HTTP Error 451: "}],
            "missed_fills": [],
        }
        assert api.is_market_closed(entry) is False
        # ...whereas an explicitly classified closed session is not an outage
        clean = json.loads(json.dumps(entry))
        clean["assets"]["GLD"] = {"note": "session_pending", "sleeve_ret": 0.0}
        clean["outages"] = []
        clean["dayStatus"] = "market_closed"
        assert api.is_market_closed(clean) is True

    def test_manifest_tamper_surfaces_not_verified(self, api, tmp_path):
        """Reader-side verification covers the manifest seal (config hash).
        A stale code sha is NOT flagged here — the tag pins what traded, and
        execution-time refusal guards the writer. Runtime mismatch is
        reported transparently instead."""
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        blob = json.loads(fp.read_text())
        blob["code_sha256"] = "f" * 64
        blob["config"]["frictions"]["fee"] = 9.99  # real tamper: config change
        from bot.prospective import _config_hash

        blob["config_sha256"] = _config_hash(blob["config"])  # keep config seal valid? no—recompute shows tamper path separately
        # first: config-tamper variant keeps OLD config hash -> mismatch
        blob["config_sha256"] = "e" * 64
        fp.write_text(json.dumps(blob))
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=3)
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["code_verified"] is False
        assert "config sha mismatch" in res["code_reason"]

    def test_stale_code_sha_reported_as_runtime_mismatch(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        blob = json.loads(fp.read_text())
        blob["code_sha256"] = "f" * 64  # reader-side cannot validate this value
        fp.write_text(json.dumps(blob))
        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(tmp_path / "log.jsonl"))
        assert res["code_verified"] is True
        assert res["runtime_matches_freeze"] is False

    def test_corrupt_forward_json_fails_closed(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=2)
        with lp.open("a", encoding="utf-8") as f:
            f.write('{"date":BROKEN}\n')

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["available"] is True
        assert res["started"] is False
        assert res["evidence_verified"] is False
        assert "integrity failure" in res["reason"]
        assert "line 3" in res["evidence_reason"]

    def test_duplicate_forward_date_fails_closed(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=2)
        first = lp.read_text(encoding="utf-8").splitlines()[0]
        with lp.open("a", encoding="utf-8") as f:
            f.write(first + "\n")

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["evidence_verified"] is False
        assert "repeats date" in res["evidence_reason"]

    def test_pre_freeze_forward_entry_fails_closed(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp, frozen_date="2026-08-23")
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=1, first="2026-08-22")

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["evidence_verified"] is False
        assert "non-forward evidence" in res["evidence_reason"]

    def test_same_day_as_freeze_is_not_forward_evidence(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp, frozen_date="2026-08-23")
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=1, first="2026-08-23")

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["evidence_verified"] is False
        assert "<= freeze date" in res["evidence_reason"]

    def test_impossible_forward_return_fails_closed(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=1, ret=-1.5)

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp))
        assert res["evidence_verified"] is False
        assert "impossible port_ret" in res["evidence_reason"]

    def test_benchmark_failure_degrades_to_none(self, api, tmp_path):
        fp = tmp_path / "freeze.json"
        _write_freeze(fp)
        lp = tmp_path / "log.jsonl"
        _write_log(lp, days=3)

        def boom():
            raise RuntimeError("fred down")

        res = api.build_forward_summary(freeze_path=str(fp), log_path=str(lp),
                                        benchmark_fetch=boom)
        assert res["benchmark_return"] is None


# ---------------------------------------------------------------------------
# verdict payload — what the dashboard hero renders
# ---------------------------------------------------------------------------

class TestVerdictPayload:
    def test_degrades_to_none_without_canonical_record(self, monkeypatch, api, tmp_path):
        """A missing sealed record must silence the verdict, not fake one."""
        monkeypatch.setattr(api, "ROOT", str(tmp_path))
        assert api.build_verdict_payload({"available": False}) is None

    def test_never_raises(self, api):
        """Every input file is optional on a cold start; none may 500 the page."""
        for forward in (None, {}, {"available": False}, {"available": True, "started": True},
                        {"available": True, "started": True, "days_full": 5,
                         "code_verified": True, "parameter_changes": 0, "data_outages": 0}):
            out = api.build_verdict_payload(forward)
            assert out is None or isinstance(out, dict)

    def test_shape(self, api):
        import os

        if not os.path.exists(os.path.join(api.ROOT, "runs", "canonical-v1", "run.json")):
            pytest.skip("canonical run not generated yet")
        v = api.build_verdict_payload(None)
        assert v is not None, "a sealed canonical record exists, so a verdict must be gradeable"
        vd = v["verdict"]
        for key in ("historical_evidence", "walk_forward_robustness", "selection_bias_risk",
                    "cost_robustness", "prospective_forward_evidence", "overall"):
            assert key in vd
        # the hero renders this string verbatim: "<grade> — <n> trading days"
        assert "trading days" in vd["prospective_forward_evidence"]
        assert vd["overall"] in ("INVALIDATED", "promising, not validated", "not established",
                                 "validated (provisional)", "partially supported")

    def test_broken_seal_forces_invalidated(self, api):
        import os

        if not os.path.exists(os.path.join(api.ROOT, "runs", "canonical-v1", "run.json")):
            pytest.skip("canonical run not generated yet")
        v = api.build_verdict_payload({
            "available": True, "started": True, "code_verified": False,
            "parameter_changes": 0, "days_full": 400, "data_outages": 0,
        })
        assert v is not None
        assert v["verdict"]["overall"] == "INVALIDATED"
        assert v["details"]["forward"]["grade"] == "COMPROMISED"

    def test_forward_days_drive_the_forward_grade(self, api):
        """The hero number and the verdict must move together."""
        import os

        if not os.path.exists(os.path.join(api.ROOT, "runs", "canonical-v1", "run.json")):
            pytest.skip("canonical run not generated yet")
        # days_full, not len(entries): the grade must follow the days that were
        # ACTUALLY observed, so a dark sleeve cannot advance the evidence.
        thin = api.build_verdict_payload({
            "available": True, "started": True, "code_verified": True,
            "parameter_changes": 0, "days_full": 12, "data_outage_days": 0,
        })
        thick = api.build_verdict_payload({
            "available": True, "started": True, "code_verified": True,
            "parameter_changes": 0, "days_full": 400, "data_outage_days": 0,
        })
        assert thin is not None and thick is not None
        assert thin["details"]["forward"]["grade"] == "Insufficient"
        assert thick["details"]["forward"]["grade"] == "Strong"
        assert "12 trading days" in thin["verdict"]["prospective_forward_evidence"]
        assert "400 trading days" in thick["verdict"]["prospective_forward_evidence"]

    def test_dark_days_do_not_advance_the_grade(self, api):
        """A feed outage must not manufacture forward evidence.

        Thirty scheduled days with every sleeve dark are worth zero observed
        days. Grading them as thirty would let the hero number improve while
        nothing at all is being measured.
        """
        import os

        if not os.path.exists(os.path.join(api.ROOT, "runs", "canonical-v1", "run.json")):
            pytest.skip("canonical run not generated yet")
        dark = api.build_verdict_payload({
            "available": True, "started": True, "code_verified": True,
            "parameter_changes": 0, "n_days_recorded": 30,
            "days_full": 0, "data_outage_days": 30,
        })
        assert dark is not None
        assert dark["details"]["forward"]["inputs"]["days_recorded"] == 0
        assert dark["details"]["forward"]["grade"] == "Insufficient"
        assert "0 trading days" in dark["verdict"]["prospective_forward_evidence"]

    def test_outage_days_are_days_not_events(self, api):
        """`data_outages` counts asset-day events; grading must see DAYS.

        Ten blocked sleeves on each of three days is three outage days, not
        thirty — the latter is 10x the denominator and made the ratio
        meaningless.
        """
        view = api._graded_forward_view({
            "n_days_recorded": 30, "days_full": 27,
            "data_outages": 30, "data_outage_days": 3,
        })
        assert view["n_days_recorded"] == 27
        assert view["data_outages"] == 3

    def test_missing_day_split_grades_as_zero(self, api):
        """Unknown observability is graded as none, never as the old count."""
        view = api._graded_forward_view({"n_days_recorded": 400})
        assert view["n_days_recorded"] == 0
        assert view["data_outages"] == 0
        assert api._graded_forward_view(None) is None


class TestClassifyForwardDays:
    """A day is worth one unit of evidence only when every sleeve printed."""

    def _day(self, notes: dict[str, str | None], day: str | None = None) -> dict:
        out = {"assets": {s: {"note": n} for s, n in notes.items()}}
        if day is not None:
            out["date"] = day
        return out

    def test_all_sleeves_live_is_a_full_day(self, api):
        out = api.classify_forward_days([self._day({"A": None, "B": None})])
        assert (out["full"], out["partial"], out["dark"]) == (1, 0, 0)

    def test_one_dark_sleeve_makes_the_day_partial(self, api):
        out = api.classify_forward_days([self._day({"A": None, "B": "outage"})])
        assert (out["full"], out["partial"], out["dark"]) == (0, 1, 0)

    def test_weekday_session_pending_is_partial_not_silently_called_closed(self, api):
        """On a weekday, pending-only is an operational/session-timing issue
        unless closure is explicit; it must not masquerade as a holiday."""
        out = api.classify_forward_days([self._day({"A": None, "B": "session_pending"})])
        assert (out["full"], out["partial"], out["dark"], out["closed"]) == (0, 1, 0, 0)

    def test_weekend_session_pending_is_closed_not_outage(self, api):
        out = api.classify_forward_days([
            self._day({"A": None, "B": "session_pending"}, day="2026-08-29")
        ])
        assert (out["full"], out["partial"], out["dark"], out["closed"]) == (0, 0, 0, 1)

    def test_known_nyse_holiday_pending_is_closed_not_outage(self, api):
        out = api.classify_forward_days([
            self._day({"BTC": None, "SPY": "session_pending"}, day="2026-09-07")
        ])
        assert (out["full"], out["partial"], out["dark"], out["closed"]) == (0, 0, 0, 1)

    def test_pending_plus_real_failure_is_still_an_outage(self, api):
        """A shut session does not excuse a genuine failure elsewhere. Only a
        day on which NOTHING broke may be called a closed market."""
        out = api.classify_forward_days([self._day({"A": None, "B": "outage", "C": "session_pending"})])
        assert (out["full"], out["partial"], out["dark"], out["closed"]) == (0, 1, 0, 0)

    def test_every_sleeve_dark_is_worth_nothing(self, api):
        out = api.classify_forward_days([self._day({"A": "outage", "B": "outage"})])
        assert (out["full"], out["partial"], out["dark"]) == (0, 0, 1)

    def test_missing_assets_block_is_dark_not_full(self, api):
        """Absent detail must never be read as 'everything printed'."""
        out = api.classify_forward_days([{"port_ret": 0.0}, {}])
        assert (out["full"], out["partial"], out["dark"]) == (0, 0, 2)

    def test_mixed_run(self, api):
        days = [
            self._day({"A": None, "B": None}),          # full
            self._day({"A": None, "B": "outage"}),      # partial
            self._day({"A": "outage", "B": "outage"}),  # dark
            self._day({"A": None, "B": None}),          # full
            self._day({"A": None, "B": "session_pending"}),  # weekday timing issue
        ]
        out = api.classify_forward_days(days)
        assert (out["full"], out["partial"], out["dark"], out["closed"]) == (2, 2, 1, 0)

    def test_empty_log(self, api):
        assert api.classify_forward_days([]) == {"full": 0, "partial": 0, "dark": 0, "closed": 0}


def test_asgi_payload_carries_verdict(monkeypatch, api):
    monkeypatch.setattr(api, "fetch_daily_history", lambda symbol: _fake_candles())
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(api.app({"type": "http", "method": "GET", "path": "/api/summary"}, receive, send))
    payload = json.loads([m for m in sent if m["type"] == "http.response.body"][0]["body"])
    # verdict may legitimately be null (no sealed record), but the key must exist
    assert "verdict" in payload
    assert payload["verdict"] is None or "overall" in payload["verdict"]["verdict"]
