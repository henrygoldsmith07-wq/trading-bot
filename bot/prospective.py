"""Prospective validation: freeze, then walk forward without touching anything.

Discipline enforced by design:
- `create_freeze` snapshots the exact per-asset strategy picks, universe,
  frictions, and overlay settings into a hash-sealed manifest with a UTC
  timestamp (committing it to git makes the freeze tamper-evident). It also
  seals the IMPLEMENTATION: a code fingerprint over the bot's source is
  recorded alongside the config hash.
- `load_freeze` verifies the config hash and — by default — the code
  fingerprint. `run_step` refuses to trade when the running source differs
  from the frozen implementation: a freeze pins behaviour, not just numbers,
  so the scheduled runner must execute the frozen commit (the CI workflow
  checks out `git_commit_at_freeze`), never an edited main.
- Every step appends to a JSONL log recording prices, weights, realized
  slippage, data outages, and missed fills.
- `report` compares bot vs S&P 500 vs BTC at 1/3/6/12-month checkpoints and
  publishes the monthly return table — negative months included.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path

from .cost_calibration import (
    append_observation as append_observation_fn,
)
from .cost_calibration import (
    build_observation as build_observation_fn,
)
from .execution import calculate_transition, open_of
from .identity import CODE_FINGERPRINT_ALGO, code_fingerprint, verify_freeze_code
from .portfolio_rules import day_allocation
from .strategy import strategy_from_spec, strategy_to_spec


def _volatility_context(completed: list[dict]) -> tuple[float | None, float | None, float | None]:
    """(realized vol 20d annualized, 30d ADV in USD, latest day base volume)
    — context fields for cost observations; None when data is thin.

    UNITS: Binance `quote_volume` is ALREADY the quote-currency (≈USD)
    turnover for that bar. It must be averaged directly — multiplying by
    close would produce price × USD nonsense."""
    import math

    closes = [c["close"] for c in completed]
    if len(closes) >= 21:
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - 20, len(closes))]
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        rv = math.sqrt(max(var, 0.0) * 365) if var > 0 else None
    else:
        rv = None
    adv = None
    quotes = [c.get("quote_volume") for c in completed[-30:]]
    usable = [float(qv) for qv in quotes if qv is not None]
    if usable:
        adv = sum(usable) / len(usable)
    day_vol_base = completed[-1].get("volume")
    return (float(rv) if rv is not None else None,
            float(adv) if adv is not None else None,
            float(day_vol_base) if day_vol_base is not None else None)

FREEZE_FILE = "freeze.json"
LOG_FILE = "forward_log.jsonl"
CHECKPOINTS = [("1 month", 30), ("3 months", 91), ("6 months", 182), ("12 months", 365)]


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _validate_freeze_config(config: dict) -> None:
    """Validate the complete immutable experiment contract, not just its hash."""
    from .algorithm import validate_algorithm

    if not isinstance(config, dict):
        raise ValueError("freeze config must be a mapping")
    required = {"assets", "frictions", "algorithm"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"freeze config missing required key(s): {sorted(missing)}")

    assets = config["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("freeze config assets must be a non-empty list")
    seen: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError(f"asset {index} must be a mapping")
        required_asset = {"symbol", "source", "periods_per_year", "strategy"}
        missing_asset = required_asset - set(asset)
        if missing_asset:
            raise ValueError(f"asset {index} missing key(s): {sorted(missing_asset)}")
        symbol = asset["symbol"]
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"asset {index} symbol must be non-empty")
        if symbol in seen:
            raise ValueError(f"duplicate frozen asset symbol: {symbol}")
        seen.add(symbol)
        source = asset["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"asset {symbol} source must be non-empty")
        ppy = asset["periods_per_year"]
        if isinstance(ppy, bool) or not isinstance(ppy, int) or ppy <= 0:
            raise ValueError(f"asset {symbol} periods_per_year must be a positive integer")
        session = asset.get("session", "continuous")
        if session not in ("continuous", "us_equity"):
            raise ValueError(f"asset {symbol} has unsupported session {session!r}")
        try:
            strategy_from_spec(asset["strategy"])
        except Exception as exc:
            raise ValueError(f"asset {symbol} has invalid strategy spec") from exc

    frictions = config["frictions"]
    if not isinstance(frictions, dict):
        raise ValueError("freeze config frictions must be a mapping")
    allowed_frictions = {
        "fee", "spread_bps", "slippage_bps", "execution", "risk_free_annual"
    }
    unknown = set(frictions) - allowed_frictions
    if unknown:
        raise ValueError(f"unknown friction key(s): {sorted(unknown)}")
    for required_key in ("fee", "execution"):
        if required_key not in frictions:
            raise ValueError(f"frictions missing required key: {required_key}")
    for key in ("fee", "spread_bps", "slippage_bps", "risk_free_annual"):
        value = frictions.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"frictions.{key} must be finite")
    if float(frictions["fee"]) < 0.0:
        raise ValueError("frictions.fee must be non-negative")
    if float(frictions.get("spread_bps", 0.0)) < 0.0:
        raise ValueError("frictions.spread_bps must be non-negative")
    if float(frictions.get("slippage_bps", 0.0)) < 0.0:
        raise ValueError("frictions.slippage_bps must be non-negative")
    if frictions["execution"] != "next_open":
        raise ValueError(
            "prospective freezes require frictions.execution='next_open'; "
            "same-close execution is a historical optimistic baseline only"
        )

    validate_algorithm(config["algorithm"])
    legacy_overlay = config.get("overlay")
    if legacy_overlay is not None:
        if not isinstance(legacy_overlay, dict):
            raise ValueError("legacy overlay view must be a mapping")
        if "target_vol" in legacy_overlay:
            target = legacy_overlay["target_vol"]
            if (
                isinstance(target, bool)
                or not isinstance(target, (int, float))
                or not math.isfinite(float(target))
                or float(target) <= 0.0
            ):
                raise ValueError("legacy overlay target_vol must be positive and finite")


def _validate_freeze_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("freeze manifest must be a mapping")
    try:
        frozen_at = datetime.fromisoformat(manifest["frozen_at"])
        frozen_date = date.fromisoformat(manifest["frozen_at_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("freeze manifest has invalid frozen_at/frozen_at_date") from exc
    if frozen_at.tzinfo is None:
        raise ValueError("freeze manifest frozen_at must be timezone-aware")
    if frozen_at.astimezone(UTC).date() != frozen_date:
        raise ValueError("freeze manifest frozen_at_date disagrees with frozen_at")
    if manifest.get("code_fingerprint_algo") != CODE_FINGERPRINT_ALGO:
        raise ValueError("freeze manifest has unknown code fingerprint algorithm")
    if not isinstance(manifest.get("config_sha256"), str) or not manifest["config_sha256"]:
        raise ValueError("freeze manifest missing config hash")
    if not isinstance(manifest.get("code_sha256"), str) or not manifest["code_sha256"]:
        raise ValueError("freeze manifest missing code fingerprint")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("freeze manifest config must be a mapping")
    _validate_freeze_config(config)


def create_freeze(
    assets: list[dict],
    frictions: dict,
    algorithm: dict,
    path: str | Path = FREEZE_FILE,
    now: datetime | None = None,
    git_commit: str | None = None,
    image_digest: str | None = None,
    git_tag: str | None = None,
    research_context: dict | None = None,
) -> dict:
    """Write the freeze manifest. `assets`: [{symbol, source, periods_per_year,
    strategy (object)}].

    `algorithm` is the COMPLETE portfolio construction (bot/algorithm.py):
    selection mode, weighting, tilt, crisis, band, throttle, overlay. It is
    nested under config so the existing tamper seal covers it — a manifest
    without it cannot prove what would run and is rejected.

    `research_context` pins the state of the research ledger at freeze time
    (entry count + sha256), so multiple-testing corrections are computed
    against an immutable record of how much was actually searched.
    """
    from .algorithm import validate_algorithm

    validate_algorithm(algorithm)
    config = {
        "assets": [
            {
                "symbol": a["symbol"],
                "source": a["source"],
                "periods_per_year": a["periods_per_year"],
                "session": a.get("session", "continuous"),
                "strategy": strategy_to_spec(a["strategy"]),
            }
            for a in assets
        ],
        "frictions": frictions,
        "overlay": {"target_vol": algorithm["overlay"]["target_vol"]},  # legacy view
        "algorithm": algorithm,
    }
    if research_context is not None:
        config["research_context"] = research_context
    _validate_freeze_config(config)
    freeze_now = now or datetime.now(UTC)
    if freeze_now.tzinfo is None:
        freeze_now = freeze_now.replace(tzinfo=UTC)
    freeze_now = freeze_now.astimezone(UTC)
    manifest = {
        "frozen_at": freeze_now.isoformat(),
        "frozen_at_date": freeze_now.date().isoformat(),
        "git_commit_at_freeze": git_commit,
        "git_tag": git_tag,
        "image_digest": image_digest,
        "retune_policy": "FORWARD PERIOD IS NEVER USED FOR SELECTION OR TUNING",
        "code_policy": "RUNNER MUST EXECUTE THE FROZEN COMMIT; REFUSE ON CODE MISMATCH",
        "config": config,
        "config_sha256": _config_hash(config),
        "code_fingerprint_algo": CODE_FINGERPRINT_ALGO,
        "code_sha256": code_fingerprint(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    encoded = json.dumps(manifest, indent=2, allow_nan=False)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return manifest


def load_freeze(path: str | Path = FREEZE_FILE, verify_code: bool = True) -> dict:
    """Load and verify the complete immutable experiment manifest."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("freeze manifest is unreadable") from exc
    _validate_freeze_manifest(manifest)
    actual = _config_hash(manifest["config"])
    if actual != manifest.get("config_sha256"):
        raise ValueError("freeze manifest hash mismatch — config was modified after freezing")
    if verify_code:
        verify_freeze_code(manifest)
    return manifest


def load_log(path: str | Path = LOG_FILE) -> list[dict]:
    """Load the prospective tape with strict chronology and numeric checks."""
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict] = []
    previous_day: str | None = None
    seen: set[str] = set()
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"forward log JSON is corrupt at line {line_no}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"forward log line {line_no} is not an object")
        day = entry.get("date")
        if not isinstance(day, str):
            raise ValueError(f"forward log line {line_no} has no date")
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError(f"forward log line {line_no} has invalid date") from exc
        if day in seen:
            raise ValueError(f"forward log repeats date {day}")
        if previous_day is not None and day <= previous_day:
            raise ValueError("forward log dates are not strictly increasing")
        port_ret = entry.get("port_ret")
        if not isinstance(port_ret, (int, float)) or isinstance(port_ret, bool):
            raise ValueError(f"forward log line {line_no} has invalid port_ret")
        port_ret_f = float(port_ret)
        if not math.isfinite(port_ret_f) or port_ret_f < -1.0:
            raise ValueError(f"forward log line {line_no} has impossible port_ret")
        if not isinstance(entry.get("assets"), dict) or not entry["assets"]:
            raise ValueError(f"forward log line {line_no} has no asset detail")
        entry["port_ret"] = port_ret_f
        entries.append(entry)
        seen.add(day)
        previous_day = day
    return entries


def _coerce_session_date(value: date | str | None, fallback: date) -> date:
    """Resolve the market-session date this step is intended to account."""
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid session_date {value!r}") from exc
    raise ValueError("session_date must be a date, ISO date string, or None")


def forward_day_kind(entry: dict) -> str:
    """Classify observation quality without deleting the recorded return."""
    assets = entry.get("assets") or {}
    if not assets:
        return "dark"
    live = sum(1 for detail in assets.values() if detail.get("note") is None)
    if live == len(assets):
        return "full"

    noted = [detail for detail in assets.values() if detail.get("note") is not None]
    only_pending = bool(noted) and all(detail.get("note") == "session_pending" for detail in noted)
    explicit_closed = entry.get("dayStatus") == "market_closed"
    try:
        weekend = date.fromisoformat(entry.get("date", "")).weekday() >= 5
    except ValueError:
        weekend = False
    if only_pending and not entry.get("outages") and (explicit_closed or weekend):
        return "closed"
    return "partial" if live else "dark"


def classify_forward_days(entries: list[dict]) -> dict[str, int]:
    counts = {"full": 0, "partial": 0, "dark": 0, "closed": 0}
    for entry in entries:
        counts[forward_day_kind(entry)] += 1
    return counts


def forward_performance(
    entries: list[dict],
    *,
    freeze_date: date | str | None = None,
    risk_free_annual: float = 0.0,
) -> dict:
    """Canonical performance view of the append-only forward tape.

    Every recorded ``port_ret`` is compounded.  Removing a partial record
    would remove a real mark-chained interval from the path.  Quality counts
    instead decide whether inferential statistics such as Sharpe are safe to
    publish.
    """
    from .metrics import max_drawdown, sharpe

    quality = classify_forward_days(entries)
    if not entries:
        return {
            "return": None,
            "sharpe": None,
            "sharpe_reason": "no forward observations",
            "max_drawdown": None,
            "curve": [],
            "periods_per_year": None,
            "quality": quality,
            "return_quality": "none",
        }

    if quality["dark"] == len(entries):
        return {
            "return": None,
            "sharpe": None,
            "sharpe_reason": "all forward observations are dark/unmeasured",
            "max_drawdown": None,
            "curve": [],
            "periods_per_year": None,
            "quality": quality,
            "return_quality": "unmeasured",
        }

    dates = [date.fromisoformat(entry["date"]) for entry in entries]
    rets = [float(entry["port_ret"]) for entry in entries]
    equity = 1.0
    equity_path = [1.0]
    curve = []
    for entry, ret in zip(entries, rets, strict=True):
        equity *= 1.0 + ret
        equity_path.append(equity)
        curve.append({"t": entry["date"], "v": round(equity, 5)})

    anchor: date | None = None
    if isinstance(freeze_date, str):
        anchor = date.fromisoformat(freeze_date)
    elif isinstance(freeze_date, date):
        anchor = freeze_date
    if anchor is not None and anchor < dates[-1]:
        span_days = (dates[-1] - anchor).days
        n_periods = len(entries)
    elif len(dates) >= 2:
        span_days = (dates[-1] - dates[0]).days
        n_periods = len(entries) - 1
    else:
        span_days = 0
        n_periods = 0
    periods_per_year = 365.2425 * n_periods / span_days if span_days > 0 and n_periods > 0 else None

    degraded = quality["partial"] > 0 or quality["dark"] > 0
    if len(rets) < 2:
        fwd_sharpe = None
        sharpe_reason = "fewer than two forward observations"
    elif degraded:
        fwd_sharpe = None
        sharpe_reason = "withheld because the tape contains partial/dark data intervals"
    elif periods_per_year is None or periods_per_year <= 0.0:
        fwd_sharpe = None
        sharpe_reason = "cannot infer a stable observation frequency"
    else:
        fwd_sharpe = sharpe(rets, max(1, int(round(periods_per_year))), risk_free_annual)
        sharpe_reason = None

    return {
        "return": equity - 1.0,
        "sharpe": fwd_sharpe,
        "sharpe_reason": sharpe_reason,
        "max_drawdown": max_drawdown(equity_path),
        "curve": curve,
        "periods_per_year": periods_per_year,
        "quality": quality,
        "return_quality": "degraded" if degraded else "complete",
    }


def append_log(entry: dict, path: str | Path = LOG_FILE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, allow_nan=False)

    # Validate the existing tape before extending it. A complete final record
    # without a newline is valid evidence, but it still needs a separator or
    # the next JSON object would be glued onto it.
    if p.exists():
        load_log(p)
        raw = p.read_text(encoding="utf-8")
        if raw and not raw.endswith(("\n", "\r")):
            with open(p, "a", encoding="utf-8") as separator:
                separator.write("\n")
                separator.flush()
                os.fsync(separator.fileno())

    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def trailing_overlay_weight(port_rets: list[float], target: float, window: int = 20) -> float:
    """Risk-overlay weight for today from the trailing window (no lookahead)."""
    import math

    if len(port_rets) < window:
        return 1.0
    hist = port_rets[-window:]
    m = sum(hist) / window
    var = sum((x - m) ** 2 for x in hist) / (window - 1)
    rv = math.sqrt(max(var, 0.0) * 365)
    if rv <= 0:
        return 1.0
    return min(1.0, target / rv)


def replay_throttle_state(port_rets: list[float], th: dict) -> tuple[float, float, bool]:
    """Replay the drawdown-throttle state machine over past portfolio returns.

    Returns (equity, peak, throttled) as of today's decision — strictly past
    data, identical to how combine_portfolio_rule evolves its own state.
    """
    equity = 1.0
    peak = 1.0
    throttled = False
    if not th.get("enabled"):
        return equity, peak, False
    for r in port_rets:
        dd = equity / peak - 1.0
        if throttled:
            if dd > th["dd_exit"]:
                throttled = False
        elif dd <= th["dd_trigger"]:
            throttled = True
        equity *= 1.0 + r
        peak = max(peak, equity)
    return equity, peak, throttled


def _feed_sanity_problem(candles) -> str | None:
    """Validate the ordered price history before any signal can consume it."""
    if not isinstance(candles, list):
        return "feed payload is not a list"
    previous_ts: float | None = None
    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            return f"candle {index} is not an object"
        try:
            ts = float(candle["open_time"])
            close = float(candle["close"])
        except (KeyError, TypeError, ValueError):
            return f"candle {index} missing numeric timestamp/close"
        if not math.isfinite(ts) or ts < 0.0:
            return f"candle {index} has invalid timestamp"
        if not math.isfinite(close) or close <= 0.0:
            return f"candle {index} has invalid close"
        if previous_ts is not None and ts <= previous_ts:
            return "feed timestamps are not strictly increasing"
        previous_ts = ts
    return None


def _bar_sanity_problem(candle: dict) -> str | None:
    """FAIL-CLOSED validation for the execution/mark bar."""
    if not isinstance(candle, dict):
        return "bar is not an object"

    def positive_finite(value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) and numeric > 0.0 else None

    close = positive_finite(candle.get("close"))
    if close is None:
        return f"invalid close ({candle.get('close')})"

    raw_open = candle.get("open")
    raw_high = candle.get("high")
    raw_low = candle.get("low")
    # `open` is independently optional in the engine contract: next-open
    # execution uses it when available and otherwise falls back to the prior
    # valid mark. High/low are auxiliary sanity fields and therefore must not
    # make a perfectly valid {open_time, open, close} bar fail closed.
    open_px = positive_finite(raw_open) if raw_open is not None else None
    if raw_open is not None and open_px is None:
        return "non-finite/zero open field"
    if (raw_high is None) != (raw_low is None):
        return "partial high/low fields"
    if raw_high is not None and raw_low is not None:
        high = positive_finite(raw_high)
        low = positive_finite(raw_low)
        if None in (high, low):
            return "non-finite/zero high/low field"
        assert high is not None and low is not None
        if high < low:
            return f"high<low ({high}<{low})"
        tol = 1.001
        if not (low / tol <= close <= high * tol):
            return f"close {close} outside [low,high]=[{low},{high}]"
    if open_px is not None:
        move = abs(close / open_px - 1.0)
        if move > 0.5:
            return f"|open->close| {move:.0%} exceeds 50% sanity bound"
    return None


def _session_pending(
    session: str,
    now: datetime,
    candles: list[dict],
    session_date: date | None = None,
) -> bool:
    """True when this asset's current daily bar cannot exist yet.

    - 'continuous' (crypto, UTC days): always live — the bar opens at 00:00.
    - 'us_equity' (NYSE): pending before 21:15 UTC (16:00 ET close + buffer,
      valid for both EST and EDT) and on any day whose bar has not appeared
      (weekends/holidays) — the dangerous fallback is executing against a
      STALE close while calling it today's fill.
    """
    if session == "continuous":
        return False
    if session != "us_equity":
        return False
    target_date = session_date or now.date()
    after_close = target_date < now.date() or now.hour > 21 or (now.hour == 21 and now.minute >= 15)
    if not candles:
        return True
    last_date = datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=UTC).date()
    if last_date == target_date and not after_close:
        # a same-day print before the gate is an intraday partial; treat as
        # pending so we never split a session that hasn't closed
        return True
    if last_date != target_date:
        return True
    return False


def _run_id_for(now: datetime) -> str:
    """Run identity for cost observations: the step's UTC date + hour bucket."""
    return f"fwd-{now.astimezone(UTC).strftime('%Y%m%dT%H')}"


def run_step(
    manifest: dict,
    fetcher,
    now: datetime | None = None,
    log_path: str | Path = LOG_FILE,
    allow_code_mismatch: bool = False,
    kwargs_quote_fetcher=None,
    cost_observation_path: str | Path | None = None,
    session_date: date | str | None = None,
) -> dict:
    if cost_observation_path is None:
        from .cost_calibration import OBSERVATIONS_LOG

        cost_observation_path = OBSERVATIONS_LOG
    """One forward day of paper trading from the frozen config only.

    `fetcher(symbol, source)` -> (candles, problem) where problem is None or a
    string describing an outage. Returns the log entry (or the existing entry
    if today was already logged — steps are idempotent per date).

    Refuses to execute unless the running source matches the manifest's
    frozen implementation. `allow_code_mismatch=True` is an explicit
    research-replay escape hatch — the scheduled runner never passes it.
    """
    if not allow_code_mismatch:
        verify_freeze_code(manifest)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)
    effective_date = _coerce_session_date(session_date, now.date())
    today = effective_date.isoformat()
    try:
        freeze_day = date.fromisoformat(manifest["frozen_at_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("freeze manifest has no valid frozen_at_date") from exc
    if effective_date <= freeze_day:
        return {
            "status": "not_after_freeze",
            "freeze_date": freeze_day.isoformat(),
            "date": today,
        }
    log = load_log(log_path)
    if any(e["date"] <= freeze_day.isoformat() for e in log):
        raise ValueError("forward log contains evidence on/before the freeze date")
    if log and log[-1]["date"] > today:
        raise ValueError("forward log contains a future-dated entry")
    for e in log:
        if e["date"] == today:
            return {"status": "already_logged", "entry": e}

    config = manifest["config"]
    frictions = config["frictions"]
    cost_rate = frictions["fee"] + (frictions.get("spread_bps", 0) + frictions.get("slippage_bps", 0)) / 10_000.0
    prev_entries = log
    prev_weights = {}
    if prev_entries:
        for sym, detail in prev_entries[-1].get("assets", {}).items():
            prev_weights[sym] = detail.get("weight", 0.0)

    n_assets = len(config["assets"])
    algo = config["algorithm"]  # required: the frozen portfolio construction
    band = float(algo["rebalance_band"])
    quote_fetcher = kwargs_quote_fetcher  # optional live bid/ask capture
    outages = []
    alerts = []  # non-fatal warnings (stale prints) that still make the audit
    missed_fills = []
    orders = []  # full lifecycle records for every turnover event
    sleeve_rets = []
    asset_details = {}
    cost_observations = []
    for a in config["assets"]:
        sym = a["symbol"]
        session = a.get("session", "continuous")
        strategy = strategy_from_spec(a["strategy"])  # raises on tampered spec: no fallback
        prev_w = prev_weights.get(sym, 0.0)  # EFFECTIVE (post-band) weight held
        try:
            candles, problem = fetcher(sym, a["source"])
        except Exception as exc:
            candles, problem = [], f"fetch exception: {type(exc).__name__}"
        if problem:
            outages.append({"symbol": sym, "problem": problem})
            sleeve_rets.append(0.0)
            asset_details[sym] = {"weight": prev_w, "target": prev_w, "price": None, "sleeve_ret": 0.0, "slippage_bps": None, "note": problem}
            continue
        feed_problem = _feed_sanity_problem(candles)
        if feed_problem:
            outages.append({"symbol": sym, "problem": f"bad feed: {feed_problem}"})
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "price": None,
                "sleeve_ret": 0.0, "slippage_bps": None,
                "note": f"fail_closed:{feed_problem}",
            }
            continue

        # ---- exchange-calendar gate ------------------------------------------
        # Crypto runs a continuous UTC session: today's bar exists from 00:00.
        # US ETFs trade an NYSE session; before it closes (~21:15 UTC covers
        # EST/EDT), TODAY'S bar cannot exist yet — executing on the last
        # available close would silently replace next_open with a stale fill.
        # In that state the asset is PENDING: hold the previous weight, mark
        # nothing, and let tomorrow's mark-chained transition cover the span.
        if _session_pending(session, now, candles, effective_date):
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "sleeve_ret": 0.0,
                "slippage_bps": None, "note": "session_pending",
                "session": session,
            }
            continue

        # ---- feed hygiene: prints from the FUTURE are ignored --------------
        # A malformed/clock-skewed feed can hand us bars dated AFTER the
        # decision instant; consuming one would let tomorrow's price leak
        # into today's fill. Compare INSTANTS (not calendar dates) so an
        # intraday-timestamped bar later today-but-after-now also fails.
        # Fail closed instead: keep only prints at or before `now`.
        now_ms = int(now.timestamp() * 1000)
        usable = [
            c for c in candles
            if c["open_time"] <= now_ms
            and datetime.fromtimestamp(c["open_time"] / 1000, tz=UTC).date() <= effective_date
        ]
        future_dropped = len(usable) < len(candles)
        if future_dropped:
            alerts.append({"symbol": sym, "level": "future_dated_rows_dropped",
                           "dropped": len(candles) - len(usable)})
        candles = usable
        if not candles:
            outages.append({"symbol": sym, "problem": "no usable current/past candles"})
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "price": None,
                "sleeve_ret": 0.0, "slippage_bps": None,
                "note": "fail_closed:no usable candles",
            }
            continue
        latest_date = datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=UTC).date()
        if session == "continuous" and latest_date != effective_date:
            problem = "missing current continuous-session bar"
            outages.append({"symbol": sym, "problem": problem})
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "price": None,
                "sleeve_ret": 0.0, "slippage_bps": None, "note": problem,
            }
            continue

        # use only completed candles for the decision
        completed = [
            c for c in candles
            if datetime.fromtimestamp(c["open_time"] / 1000, tz=UTC).date() < effective_date
        ]
        if len(completed) < 2:
            outages.append({"symbol": sym, "problem": "insufficient completed candles"})
            sleeve_rets.append(0.0)
            asset_details[sym] = {"weight": prev_w, "target": prev_w, "price": None, "sleeve_ret": 0.0, "slippage_bps": None, "note": "insufficient history"}
            continue
        # data-staleness alert: a print much older than one bar means the feed
        # is silently frozen; trade the stale weight but say so loudly
        completed_date = datetime.fromtimestamp(completed[-1]["open_time"] / 1000, tz=UTC).date()
        age_days = float((effective_date - completed_date).days)
        if session == "continuous" and age_days > 3.0:
            alerts.append({"symbol": sym, "level": "stale_data", "age_days": round(age_days, 1)})
            problem = f"stale completed history ({age_days:.1f}d)"
            outages.append({"symbol": sym, "problem": problem})
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "price": None,
                "sleeve_ret": 0.0, "slippage_bps": None, "note": problem,
            }
            continue

        # ---- FAIL CLOSED on corrupted prints --------------------------------
        # An impossible final bar (non-positive close, high<low, |1d move|
        # beyond a wide sanity bound, non-finite fields) must never become an
        # execution price. The asset sits out this session; the incident is
        # logged as an outage so the audit trail shows the gap.
        session_rows = [
            c for c in candles
            if datetime.fromtimestamp(c["open_time"] / 1000, tz=UTC).date() == effective_date
        ]
        if not session_rows:
            problem = "no bar for requested session date"
            outages.append({"symbol": sym, "problem": problem})
            sleeve_rets.append(0.0)
            asset_details[sym] = {
                "weight": prev_w, "target": prev_w, "price": None,
                "sleeve_ret": 0.0, "slippage_bps": None, "note": problem,
            }
            continue
        last_bar = session_rows[-1]
        sanity_problem = _bar_sanity_problem(last_bar)
        if sanity_problem:
            outages.append({"symbol": sym, "problem": f"bad candle: {sanity_problem}"})
            sleeve_rets.append(0.0)
            asset_details[sym] = {"weight": prev_w, "target": prev_w, "price": None,
                                  "sleeve_ret": 0.0, "slippage_bps": None,
                                  "note": f"fail_closed:{sanity_problem}"}
            continue

        try:
            raw_target = strategy.weight(completed)
            if (
                isinstance(raw_target, bool)
                or not isinstance(raw_target, (int, float))
                or not math.isfinite(float(raw_target))
            ):
                raise ValueError(f"non-finite/non-numeric target {raw_target!r}")
            w_target = max(0.0, min(1.0, float(raw_target)))
        except Exception as exc:  # one bad symbol must not create an order
            alerts.append({"symbol": sym, "level": "strategy_error",
                           "detail": f"{type(exc).__name__}: {exc}"})
            w_target = prev_w  # hold previous weight; no new signal this bar
        # rebalance band — same semantics as engine.run_strategy: act only when
        # the target moves further from the HELD weight than the band
        w_eff = w_target if abs(w_target - prev_w) > band else prev_w
        decision_close = completed[-1]["close"]
        today_candle = session_rows[-1]
        exec_price = open_of(today_candle, fallback=decision_close)  # frozen next-open convention
        closing_price = today_candle["close"]  # latest snapshot (live print / closed equity bar)
        # Accounting anchor: the LAST MARKED price for this asset (previous
        # entry's snapshot), falling back to the decision close on the first
        # day or after an outage. Chaining marks means every price interval
        # is accounted exactly once — no silent tail gaps between snapshots.
        prev_detail = prev_entries[-1].get("assets", {}).get(sym, {}) if prev_entries else {}
        accounting_anchor = prev_detail.get("mark_price") or decision_close
        tr = calculate_transition(
            prev_w,
            w_eff,
            previous_close=accounting_anchor,
            execution_price=exec_price,
            closing_price=closing_price,
            costs=cost_rate,
            cash_rate_period=config["frictions"].get("risk_free_annual", 0.0) / 365.0,
            cash_basis="previous",  # engine next_open convention: cash on old allocation
        )
        sleeve_ret = tr["return"]
        sleeve_rets.append(sleeve_ret)
        slip_bps = None
        if abs(w_eff - prev_w) > 0.01:
            slip_bps = (exec_price / decision_close - 1.0) * 10_000.0  # decision-to-execution gap
        # a data gap that delayed a weight change = missed fill(s)
        last_gap_days = (completed[-1]["open_time"] - completed[-2]["open_time"]) / 86_400_000
        if last_gap_days > 1.5 and abs(w_target - prev_w) > 0.05:
            missed_fills.append({"symbol": sym, "delayed_days": int(last_gap_days - 1)})
        asset_details[sym] = {
            "weight": w_eff,
            "target": w_target,
            "price": closing_price,
            "mark_price": closing_price,
            "exec_price": exec_price,
            "decision_close": decision_close,
            "accounting_anchor": accounting_anchor,
            "overnight": tr["overnight"],
            "intraday": tr["intraday"],
            "sleeve_ret": sleeve_ret,
            "slippage_bps": slip_bps,
        }
        # ---- event chain: signal → intent → fill → position -----------------
        # Every turnover appends the full order lifecycle so the tape can be
        # replayed as: signal generated → intended order → simulated
        # submission → fill → resulting position. Holds are not orders.
        if abs(w_eff - prev_w) > 1e-9:
            signal_ts = datetime.fromtimestamp(
                completed[-1]["open_time"] / 1000 + 86_399.0, tz=UTC
            ).isoformat()
            orders.append({
                "symbol": sym,
                "side": "BUY" if w_eff > prev_w else "SELL",
                "signal_generated_ts": signal_ts,
                "intent_ts": (datetime.fromtimestamp(
                    completed[-1]["open_time"] / 1000 + 86_400.0, tz=UTC
                )).isoformat(),  # next daily open after the signal close
                "submitted_ts": now.isoformat(),
                "fill_ts": now.isoformat(),
                "fill_price": exec_price,
                "qty_weight_delta": round(w_eff - prev_w, 6),
                "cost_fraction": round(cost_rate * abs(w_eff - prev_w), 8),
                "position_after": round(w_eff, 6),
            })
        # ---- cost observation: measure what V1 predicts vs the tape -------
        if tr["turnover"] > 1e-9 and cost_observation_path is not None:
            bid = ask = None
            if quote_fetcher is not None:
                try:
                    q = quote_fetcher(sym)
                except Exception:
                    q = None
                if q:
                    bid, ask = q.get("bid"), q.get("ask")
            rv_annual, adv30_usd, day_vol_base = _volatility_context(completed)
            predicted_cost_bps = cost_rate * 10_000.0
            side = "BUY" if w_eff > prev_w else ("SELL" if w_eff < prev_w else "FLAT")
            signal_ts = datetime.fromtimestamp(
                completed[-1]["open_time"] / 1000 + 86_399.0, tz=UTC
            ).isoformat()
            obs = build_observation_fn(
                symbol=sym,
                side=side,
                target_weight=w_eff,
                previous_weight=prev_w,
                decision_close=decision_close,
                exec_price=exec_price,
                mark_price=closing_price,
                bid=bid,
                ask=ask,
                predicted_cost_bps=predicted_cost_bps,
                realized_vol_annual=rv_annual,
                adv30_usd=adv30_usd,
                day_volume_base=day_vol_base,
                signal_ts=signal_ts,
                ts=now.isoformat(),
            )
            # forward-evidence purity: full provenance stamped at write time
            from .evidence import EVIDENCE_FORWARD_PAPER, new_observation_meta

            obs.update(new_observation_meta(
                freeze_manifest=manifest,
                run_id=_run_id_for(now),
                simulated_execution_at=now.isoformat(),
            ))
            obs["evidenceClass"] = EVIDENCE_FORWARD_PAPER
            obs["simulatedExecutionAt"] = now.isoformat()
            cost_observations.append(obs)
            append_observation_fn(obs, path=cost_observation_path)

    # ---- portfolio construction: THE FROZEN ALGORITHM ----------------------
    # Sleeve history for every asset comes from the log (strictly past days);
    # today's weights/exposure come from the same day_allocation function the
    # backtest combiner uses — identical math by construction.
    sleeve_hist: dict[str, list[float]] = {a["symbol"]: [] for a in config["assets"]}
    for e in prev_entries:
        for sym, detail in e.get("assets", {}).items():
            if sym in sleeve_hist and isinstance(detail, dict):
                sleeve_hist[sym].append(float(detail.get("sleeve_ret", 0.0)))
    present = [sym for sym in asset_details if asset_details[sym].get("note") is None]
    xs, wt, cd, th = algo["xs_momentum"], algo["weighting"], algo["crisis_derisk"], algo["drawdown_throttle"]
    # Overlay and throttle read the RULE series (pre-overlay), matching how
    # the backtest pipeline stacks _vol_overlay on top of combine_portfolio_rule.
    rule_history = [e["rule_ret"] for e in prev_entries]
    equity, peak, throttled = replay_throttle_state(rule_history, th)
    dd = equity / peak - 1.0
    weights, exposure, throttled_new = day_allocation(
        sleeve_hist,
        present,
        n_assets,
        vol_window=wt["vol_window"],
        max_multiple_of_equal=wt["max_multiple_of_equal"],
        use_tilt=xs["enabled"],
        tilt_lookback=xs["lookback"],
        max_tilt=xs["max_tilt"],
        use_crisis=cd["enabled"],
        corr_window=cd["corr_window"],
        corr_threshold=cd["corr_threshold"],
        derisk=cd["multiplier"],
        dd=dd,
        throttled=throttled,
        use_dd_throttle=th["enabled"],
        dd_trigger=th["dd_trigger"],
        dd_exit=th["dd_exit"],
        throttle=th["factor"],
    )
    port_gross = sum(weights.get(sym, 0.0) * asset_details[sym]["sleeve_ret"] for sym in present)
    rule_ret = port_gross * exposure

    ov = algo["overlay"]
    overlay_w = trailing_overlay_weight(rule_history, ov["target_vol"], window=ov["window"]) if ov["enabled"] else 1.0
    prev_overlay = prev_entries[-1].get("overlay_weight", 1.0) if prev_entries else 1.0
    overlay_fee = ov["fee_on_turnover"]
    port_ret = overlay_w * rule_ret - overlay_fee * abs(overlay_w - prev_overlay)
    # dayStatus: every scheduled day is recorded, interesting or not
    n_pending = sum(1 for d in asset_details.values() if d.get("note") == "session_pending")
    n_outage_assets = sum(
        1 for d in asset_details.values()
        if isinstance(d.get("note"), str) and d["note"] not in ("session_pending",)
    )
    if len(outages) == n_assets and n_assets > 0:
        day_status = "data_outage"
    elif not orders and n_pending == len(asset_details) and n_pending > 0:
        day_status = "session_pending"
    elif not orders and n_outage_assets > 0:
        day_status = "partial_outage"
    elif not orders:
        day_status = "no_signal"
    else:
        day_status = "traded"
    entry = {
        "ts": now.isoformat(),
        "date": today,
        "session_date": today,
        "assets": asset_details,
        "port_ret": port_ret,
        "rule_ret": rule_ret,
        "overlay_weight": overlay_w,
        "exposure": exposure,
        "throttled": throttled_new,
        "dayStatus": day_status,
        "orders": orders,
        "outages": outages,
        "missed_fills": missed_fills,
    }
    if alerts:
        entry["alerts"] = alerts
    append_log(entry, log_path)
    return {"status": "logged", "entry": entry}


def checkpoints_due(freeze_date: date, as_of: date) -> list[dict]:
    elapsed = (as_of - freeze_date).days
    return [
        {"label": label, "days": days, "elapsed": elapsed, "due": elapsed >= days}
        for label, days in CHECKPOINTS
    ]


def monthly_returns(entries: list[dict]) -> dict[str, float]:
    """Compound forward-log returns per calendar month (negatives kept)."""
    months: dict[str, list[float]] = {}
    for e in entries:
        months.setdefault(e["date"][:7], []).append(e["port_ret"])
    out = {}
    for m, rets in sorted(months.items()):
        eq = 1.0
        for r in rets:
            eq *= 1.0 + r
        out[m] = eq - 1.0
    return out


def slippage_stats(entries: list[dict]) -> dict:
    obs = [d["slippage_bps"] for e in entries for d in e.get("assets", {}).values() if isinstance(d, dict) and d.get("slippage_bps") is not None]
    if not obs:
        return {"count": 0, "mean_abs_bps": None}
    return {"count": len(obs), "mean_abs_bps": sum(abs(x) for x in obs) / len(obs)}


def outage_stats(entries: list[dict]) -> dict:
    return {
        "outage_days": sum(1 for e in entries if e.get("outages")),
        "outage_events": sum(len(e.get("outages", [])) for e in entries),
        "missed_fills": sum(len(e.get("missed_fills", [])) for e in entries),
    }


def alert_stats(entries: list[dict]) -> dict:
    """Data-staleness (and other) alerts across the forward log."""
    alerts = [a for e in entries for a in e.get("alerts", [])]
    return {
        "total": len(alerts),
        "by_level": {lvl: sum(1 for a in alerts if a.get("level") == lvl) for lvl in sorted({a.get("level", "?") for a in alerts})},
        "symbols": sorted({a["symbol"] for a in alerts}),
    }
