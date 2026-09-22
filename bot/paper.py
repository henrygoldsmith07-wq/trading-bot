"""Paper trading engine: simulated fills, no real orders are ever placed.

Reliability features (the broker is the part that must never lose track):

- Persistent MULTI-ASSET portfolio state, written atomically
  (temp file + os.replace) with a sha256 checksum so a crash mid-write can
  never leave a half-updated balance file behind.
- Append-only ORDER LEDGER (JSONL): every fill records its idempotency key,
  deltas, post-trade balances, and a decision explanation.
- CRASH RECOVERY: if the state file is corrupt or fails its checksum, the
  portfolio is rebuilt by replaying the ledger's deltas from start cash.
- DUPLICATE-ORDER PREVENTION: every decision carries an idempotency key
  (date|symbol|side|target); re-running a cycle cannot double-fill.
- DATA-STALENESS ALERTS: symbols whose latest candle is too old get their
  trading blocked for the cycle and raise alerts into the audit trail.
- DAILY AUDIT REPORTS: markdown files under reports_dir with positions,
  fills, alerts, and the explanation behind every decision (holds included).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

STATE_SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _checksum(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive_finite(value) -> bool:
    return _finite(value) and float(value) > 0.0


class LedgerCorruptionError(ValueError):
    """Raised when durable paper-trading evidence cannot be replayed safely."""


class OrderLedger:
    """Append-only JSONL record of every simulated order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, order: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(order, sort_keys=True, allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        out = []
        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                # Only an unterminated final record is a valid crash artefact.
                # Corruption in the middle of an append-only ledger must never
                # be silently skipped because replay would invent balances.
                if line_no == len(lines) and not raw.endswith(("\n", "\r")):
                    continue
                raise LedgerCorruptionError(f"corrupt ledger JSON at line {line_no}") from exc
            if not isinstance(entry, dict):
                raise LedgerCorruptionError(f"ledger line {line_no} is not an object")
            out.append(entry)
        return out

    def idem_keys(self) -> set[str]:
        return {e["idem_key"] for e in self.entries() if e.get("idem_key")}


def _save_state_atomic(state: dict, path: Path) -> None:
    body = {k: v for k, v in state.items() if k != "checksum"}
    wrapped = {"schema_version": STATE_SCHEMA_VERSION, **body}
    wrapped["checksum"] = _checksum({k: v for k, v in wrapped.items()})
    encoded = json.dumps(wrapped, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        wrapped = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(wrapped, dict):
        return None
    checksum = wrapped.pop("checksum", None)
    try:
        checksum_ok = checksum == _checksum(wrapped)
    except (TypeError, ValueError):
        return None
    if not checksum_ok or wrapped.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    positions = wrapped.get("positions")
    if not _finite(wrapped.get("cash")) or float(wrapped["cash"]) < -1e-6 or not isinstance(positions, dict):
        return None
    for sym, qty in positions.items():
        if not isinstance(sym, str) or not _finite(qty) or float(qty) < -1e-12:
            return None
    return wrapped


class PaperPortfolio:
    """Persistent multi-asset paper portfolio with ledger-based recovery."""

    def __init__(
        self,
        start_cash: float = 10_000.0,
        fee: float = 0.001,
        state_file: str = "paper_state.json",
        ledger_file: str = "paper_ledger.jsonl",
    ):
        if not _finite(start_cash) or float(start_cash) < 0.0:
            raise ValueError("start_cash must be finite and non-negative")
        if not _finite(fee) or float(fee) < 0.0:
            raise ValueError("fee must be finite and non-negative")
        self.start_cash = float(start_cash)
        self.fee = float(fee)
        self.state_file = Path(state_file)
        self.ledger = OrderLedger(ledger_file)
        self._idem_keys: set[str] = set()

        # The ledger is fsynced BEFORE the state snapshot is replaced, so it is
        # the source of truth after a crash. A valid-but-stale state file can
        # exist if the process dies between those two operations.
        state = _load_state(self.state_file)
        recovered = self._recover_from_ledger(self.start_cash)
        if recovered is not None:
            self.cash, self.positions = recovered
            state_matches = (
                state is not None
                and math.isclose(float(state["cash"]), self.cash, rel_tol=0.0, abs_tol=1e-6)
                and set(state["positions"]) == set(self.positions)
                and all(
                    math.isclose(float(state["positions"][sym]), qty, rel_tol=0.0, abs_tol=1e-10)
                    for sym, qty in self.positions.items()
                )
            )
            if not state_matches:
                self.persist()
        elif state is not None:
            self.cash = float(state["cash"])
            self.positions = {s: float(q) for s, q in state["positions"].items()}
            self._idem_keys = self.ledger.idem_keys()
        else:
            self.cash = self.start_cash
            self.positions = {}
            self._idem_keys = set()

    def _recover_from_ledger(self, start_cash: float) -> tuple[float, dict[str, float]] | None:
        """Rebuild balances by validating and replaying every durable fill."""
        entries = [e for e in self.ledger.entries() if e.get("kind") == "fill"]
        if not entries:
            return None

        positions: dict[str, float] = {}
        seen_keys: set[str] = set()
        cash: float | None = None
        required = {
            "idem_key",
            "symbol",
            "side",
            "qty",
            "price",
            "notional",
            "fee",
            "cash_before",
            "cash_after",
            "position_after",
        }

        for index, e in enumerate(entries, start=1):
            missing = required - set(e)
            if missing:
                raise LedgerCorruptionError(
                    f"ledger fill {index} missing fields: {','.join(sorted(missing))}"
                )
            idem_key = str(e["idem_key"])
            if not idem_key or idem_key in seen_keys:
                raise LedgerCorruptionError(f"ledger fill {index} has duplicate/empty idempotency key")
            seen_keys.add(idem_key)

            sym = e["symbol"]
            side = e["side"]
            if not isinstance(sym, str) or not sym or side not in ("BUY", "SELL"):
                raise LedgerCorruptionError(f"ledger fill {index} has invalid symbol/side")

            numeric_names = (
                "qty",
                "price",
                "notional",
                "fee",
                "cash_before",
                "cash_after",
                "position_after",
            )
            if any(not _finite(e[name]) for name in numeric_names):
                raise LedgerCorruptionError(f"ledger fill {index} contains a non-finite number")

            qty = float(e["qty"])
            price = float(e["price"])
            notional = float(e["notional"])
            fee_paid = float(e["fee"])
            cash_before = float(e["cash_before"])
            cash_after = float(e["cash_after"])
            position_after = float(e["position_after"])

            if price <= 0.0 or notional < 0.0 or fee_paid < 0.0 or cash_before < -1e-6 or cash_after < -1e-6:
                raise LedgerCorruptionError(f"ledger fill {index} contains an impossible balance/price")
            if qty > 1e-12 and side != "BUY":
                raise LedgerCorruptionError(f"ledger fill {index} side disagrees with quantity")
            if qty < -1e-12 and side != "SELL":
                raise LedgerCorruptionError(f"ledger fill {index} side disagrees with quantity")
            if cash is None:
                cash = cash_before
            elif not math.isclose(cash_before, cash, rel_tol=0.0, abs_tol=2e-5):
                raise LedgerCorruptionError(f"ledger cash continuity breaks at fill {index}")

            expected_notional = abs(qty) * price
            if not math.isclose(notional, expected_notional, rel_tol=1e-9, abs_tol=2e-5):
                raise LedgerCorruptionError(f"ledger notional is inconsistent at fill {index}")
            expected_cash = (
                cash_before - notional - fee_paid
                if side == "BUY"
                else cash_before + notional - fee_paid
            )
            if not math.isclose(cash_after, expected_cash, rel_tol=0.0, abs_tol=2e-5):
                raise LedgerCorruptionError(f"ledger cash arithmetic is inconsistent at fill {index}")

            expected_position = positions.get(sym, 0.0) + qty
            if position_after < -1e-10 or not math.isclose(
                position_after, expected_position, rel_tol=0.0, abs_tol=2e-10
            ):
                raise LedgerCorruptionError(f"ledger position continuity breaks at fill {index}")

            cash = cash_after
            if abs(position_after) < 1e-12:
                positions.pop(sym, None)
            else:
                positions[sym] = position_after

        self._idem_keys = seen_keys
        return (cash if cash is not None else float(start_cash)), positions

    def persist(self) -> None:
        _save_state_atomic(
            {
                "cash": self.cash,
                "positions": self.positions,
                "updated_at": _utc_now().isoformat(),
            },
            self.state_file,
        )

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, qty in self.positions.items():
            px = prices.get(sym)
            if px is None:  # unknown mark: carry last known cost basis conservatively
                continue
            total += qty * px
        return total

    def target_position_for(self, symbol: str, target_weight: float, prices: dict[str, float]) -> float:
        if not _finite(target_weight) or not 0.0 <= float(target_weight) <= 1.0:
            raise ValueError(f"target_weight must be finite and within [0, 1] (got {target_weight})")
        missing_marks = sorted(
            sym
            for sym, qty in self.positions.items()
            if abs(qty) > 1e-12 and not _positive_finite(prices.get(sym))
        )
        if missing_marks:
            raise ValueError(f"missing usable price for held positions: {','.join(missing_marks)}")
        px = prices.get(symbol)
        if not _positive_finite(px):
            raise ValueError(f"no usable price for {symbol}")
        assert px is not None
        px_value = float(px)
        eq = self.equity(prices)
        if not _finite(eq) or eq < 0.0:
            raise ValueError("portfolio equity is not finite/non-negative")
        return float(target_weight) * eq / px_value

    def rebalance(
        self,
        symbol: str,
        target_weight: float,
        price: float,
        idem_key: str,
        reason: str = "",
        ts: datetime | None = None,
        min_notional: float = 1.0,
        market_prices: dict[str, float] | None = None,
    ) -> dict | None:
        """Move `symbol` toward `target_weight` of equity. Returns the fill
        record, or None when skipped (duplicate key / no price / dust).

        Idempotency: a previously-seen `idem_key` is rejected outright, so a
        retried or re-run cycle can never double-execute.
        """
        if idem_key in self._idem_keys:
            return {"skipped": "duplicate_order", "idem_key": idem_key}
        if not _positive_finite(price):
            return {"skipped": f"no usable price for {symbol}"}
        if not _finite(min_notional) or float(min_notional) < 0.0:
            return {"skipped": "invalid_min_notional"}
        price = float(price)
        min_notional = float(min_notional)
        # Position sizing must use TOTAL portfolio equity. In a multi-asset
        # portfolio, valuing only the symbol being traded silently drops every
        # other holding from equity and under-sizes later rebalances.
        prices = dict(market_prices or {})
        prices[symbol] = price
        try:
            target_qty = self.target_position_for(symbol, target_weight, prices)
        except ValueError as e:
            return {"skipped": str(e)}
        current_qty = self.positions.get(symbol, 0.0)
        delta_qty = target_qty - current_qty
        notional = abs(delta_qty) * price
        if notional < min_notional:
            return {"skipped": "below_min_notional", "notional": notional}

        side = "BUY" if delta_qty > 0 else "SELL"
        cash_before = self.cash
        notional = abs(delta_qty) * price
        if side == "BUY":
            # affordability cap: additional qty whose notional+fee fits in cash;
            # a clamp must NEVER flip a buy into an implicit sell
            max_delta_qty = cash_before / ((1.0 + self.fee) * price)
            if delta_qty > max_delta_qty:
                delta_qty = max_delta_qty
                notional = delta_qty * price
            if abs(delta_qty) < 1e-12 or notional < min_notional:
                return {"skipped": "insufficient_cash", "notional": notional}
            fee_paid = notional * self.fee
            self.cash = cash_before - notional - fee_paid
        else:
            fee_paid = notional * self.fee
            self.cash += notional - fee_paid
        self.positions[symbol] = current_qty + delta_qty
        if abs(self.positions[symbol]) < 1e-12:
            del self.positions[symbol]

        fill_ts = ts or _utc_now()
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.replace(tzinfo=UTC)
        fill = {
            "kind": "fill",
            "ts": fill_ts.isoformat(),
            "date": fill_ts.astimezone(UTC).date().isoformat(),
            "idem_key": idem_key,
            "symbol": symbol,
            "side": side,
            "qty": round(delta_qty, 12),
            "price": price,
            "notional": round(notional, 6),
            "fee": round(fee_paid, 8),
            "cash_before": round(cash_before, 8),
            "cash_after": round(self.cash, 8),
            "position_after": round(self.positions.get(symbol, 0.0), 12),
            "target_weight": target_weight,
            "reason": reason,
        }
        self.ledger.append(fill)
        self._idem_keys.add(idem_key)
        self.persist()
        return fill


# ---------------------------------------------------------------------------
# Staleness detection, audit reports, and the live loop
# ---------------------------------------------------------------------------

def staleness_alerts(
    candles_by_symbol: dict[str, list[dict]],
    now_ms: int | None = None,
    max_age_days: float = 3.0,
) -> list[dict]:
    """Alerts for missing, malformed, future-dated, or stale market data."""
    now = time.time() * 1000 if now_ms is None else now_ms
    alerts: list[dict] = []
    for sym, candles in candles_by_symbol.items():
        if not candles:
            alerts.append({"symbol": sym, "age_days": float("inf"), "level": "stale_data"})
            continue
        latest = candles[-1]
        if not isinstance(latest, dict) or not _finite(latest.get("open_time")):
            alerts.append({"symbol": sym, "level": "invalid_timestamp"})
            continue
        age_days = (now - float(latest["open_time"])) / 86_400_000
        if age_days < -0.05:
            alerts.append({"symbol": sym, "age_days": round(age_days, 1), "level": "future_data"})
        elif age_days > max_age_days:
            alerts.append({"symbol": sym, "age_days": round(age_days, 1), "level": "stale_data"})
    return alerts


def _scale_targets_to_gross_cap(
    targets: dict[str, tuple[float, str]],
    blocked_symbols: set[str],
    max_gross_exposure: float,
) -> tuple[dict[str, tuple[float, str]], dict | None]:
    """Proportionally cap tradeable long-only targets to deterministic gross exposure."""
    if not _finite(max_gross_exposure) or not 0.0 < float(max_gross_exposure) <= 1.0:
        raise ValueError("max_gross_exposure must be finite and within (0, 1]")
    cap = float(max_gross_exposure)
    raw_gross = sum(
        float(weight)
        for sym, (weight, _why) in targets.items()
        if sym not in blocked_symbols and _finite(weight) and 0.0 <= float(weight) <= 1.0
    )
    if raw_gross <= cap + 1e-12:
        return targets, None

    scale = cap / raw_gross
    scaled: dict[str, tuple[float, str]] = {}
    for sym, (weight, why) in targets.items():
        if sym not in blocked_symbols and _finite(weight) and 0.0 <= float(weight) <= 1.0:
            adjusted = float(weight) * scale
            scaled[sym] = (
                adjusted,
                f"{why} gross-cap raw_target={float(weight):.3f} scaled_target={adjusted:.3f}",
            )
        else:
            scaled[sym] = (weight, why)
    return scaled, {
        "level": "target_weights_scaled",
        "detail": f"raw_gross={raw_gross:.6f} cap={cap:.6f} scale={scale:.6f}",
    }


def decide_orders(
    targets: dict[str, tuple[float, str]],  # symbol -> (target_weight, explanation)
    cash: float,
    current_positions: dict[str, float],
    prices: dict[str, float],
    stale_symbols: set[str],
    weight_epsilon: float = 1e-4,
    as_of: datetime | None = None,
) -> list[dict]:
    """Plan this cycle's decisions WITHOUT executing: every symbol gets an
    explanation (holds included); stale symbols are blocked up front.
    Weights follow the engine convention: fraction of TOTAL equity
    (cash + positions), so a flat symbol's weight is 0."""
    decisions: list[dict] = []
    decision_time = as_of or _utc_now()
    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=UTC)
    date = decision_time.astimezone(UTC).date().isoformat()
    missing_marks = sorted(
        s for s, q in current_positions.items() if abs(q) > 1e-12 and not _positive_finite(prices.get(s))
    )
    total_equity = (
        float(cash) + sum(float(q) * float(prices[s]) for s, q in current_positions.items())
        if not missing_marks and _finite(cash)
        else float("nan")
    )
    for sym, (w, why) in targets.items():
        if sym in stale_symbols:
            decisions.append({"symbol": sym, "action": "blocked_stale", "explanation": why})
            continue
        if missing_marks:
            decisions.append(
                {
                    "symbol": sym,
                    "action": "blocked_missing_mark",
                    "explanation": f"{why} missing_marks={','.join(missing_marks)}",
                }
            )
            continue
        if not _finite(w) or not 0.0 <= float(w) <= 1.0:
            decisions.append(
                {
                    "symbol": sym,
                    "action": "blocked_invalid_target",
                    "explanation": f"{why} invalid_target_weight={w}",
                }
            )
            continue
        if not _positive_finite(prices.get(sym)):
            decisions.append({"symbol": sym, "action": "blocked_invalid_price", "explanation": why})
            continue
        if not _finite(total_equity) or total_equity <= 0.0:
            decisions.append({"symbol": sym, "action": "blocked_invalid_equity", "explanation": why})
            continue
        px = float(prices[sym])
        eq = total_equity
        w = float(w)
        cur_w = current_positions.get(sym, 0.0) * px / eq
        action = "hold" if abs(w - cur_w) <= weight_epsilon else ("BUY" if w > cur_w else "SELL")
        decisions.append(
            {
                "symbol": sym,
                "action": action,
                "target_weight": w,
                "current_weight": round(cur_w, 4),
                "explanation": f"{why} current_weight={cur_w:.3f}",
                "idem_key": build_idem_key(sym, "REBAL", w, date),
            }
        )
    return decisions


def daily_audit_report(
    portfolio: PaperPortfolio,
    prices: dict[str, float],
    decisions: list[dict],
    fills: list[dict],
    alerts: list[dict],
    as_of: datetime | None = None,
) -> str:
    """Markdown audit report for one cycle: positions, decisions (+why),
    fills, alerts."""
    now = as_of or _utc_now()
    lines = [f"# Paper audit report — {now.date().isoformat()} {now.strftime('%H:%M')} UTC"]
    lines.append(f"\nEquity: ${portfolio.equity(prices):,.2f}  Cash: ${portfolio.cash:,.2f}")
    lines.append("\n## Positions")
    if portfolio.positions:
        lines.append("| symbol | qty | price | value | weight |")
        lines.append("|---|---|---|---|---|")
        eq = max(portfolio.equity(prices), 1e-9)
        for sym, qty in sorted(portfolio.positions.items()):
            px = prices.get(sym, 0.0)
            val = qty * px
            lines.append(f"| {sym} | {qty:.6f} | {px:.2f} | ${val:,.2f} | {val / eq:.1%} |")
    else:
        lines.append("(flat)")
    lines.append("\n## Decisions & explanations")
    for d in decisions:
        lines.append(f"- `{d['symbol']}` {d['action']} — {d['explanation']}")
    lines.append("\n## Fills this cycle")
    if fills:
        for f in fills:
            if isinstance(f, dict) and f.get("kind") == "fill":
                lines.append(
                    f"- {f['ts'][:19]} {f['side']} {f['qty']:.6f} {f['symbol']} @ {f['price']:.2f} "
                    f"(fee ${f['fee']:.4f}) — {f['reason']}"
                )
    else:
        lines.append("(none)")
    lines.append("\n## Alerts")
    if alerts:
        for a in alerts:
            subject = a.get("symbol", "system")
            if "age_days" in a:
                lines.append(f"- {subject}: {a['level']} ({a['age_days']}d old)")
            else:
                detail = f" — {a['detail']}" if a.get("detail") else ""
                lines.append(f"- {subject}: {a['level']}{detail}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_audit_report(reports_dir: str | Path, content: str, as_of: datetime | None = None) -> Path:
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = (as_of or _utc_now()).date().isoformat()
    path = d / f"audit_{stamp}.md"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def build_idem_key(symbol: str, action: str, target_weight: float, date_iso: str | None = None) -> str:
    d = date_iso or _utc_now().date().isoformat()
    return f"{d}|{symbol}|{action}|{round(target_weight, 4)}"


def run_cycle(
    symbols: list[str],
    weight_fn,
    fetch_candles_fn,
    portfolio: PaperPortfolio,
    reports_dir: str | Path = "reports",
    max_age_days: float = 3.0,
    max_gross_exposure: float = 1.0,
    now: datetime | None = None,
    ai_note_fn=None,
) -> dict:
    """One full paper-trading cycle over `symbols`: fetch, explain, execute
    (idempotently), alert, and write the daily audit report.

    `ai_note_fn(report_text) -> str | None`, when provided, appends a clearly
    labeled advisory AI-commentary section to the report. It can never affect
    weights or fills — it runs after execution is complete.
    """
    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if not _finite(max_age_days) or float(max_age_days) < 0.0:
        raise ValueError("max_age_days must be finite and non-negative")
    if not symbols:
        raise ValueError("at least one symbol is required")
    normalized_symbols = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    if not normalized_symbols:
        raise ValueError("at least one non-empty symbol is required")
    if len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("symbols must be unique")

    candles_by_symbol = {}
    targets: dict[str, tuple[float, str]] = {}
    alerts: list[dict] = []
    for sym in normalized_symbols:
        try:
            candles = fetch_candles_fn(sym)
        except Exception as exc:  # data-source failure is isolated to this sleeve
            candles = []
            alerts.append({"symbol": sym, "level": "data_fetch_error", "detail": type(exc).__name__})
        if not isinstance(candles, list):
            alerts.append({"symbol": sym, "level": "invalid_candle_payload", "detail": type(candles).__name__})
            candles = []
        candles_by_symbol[sym] = candles
        if candles:
            if not all(isinstance(candle, dict) for candle in candles):
                targets[sym] = (0.0, "malformed candle payload — trading blocked")
                alerts.append({"symbol": sym, "level": "invalid_candle_payload"})
                continue
            closes = [candle.get("close") for candle in candles]
            latest = closes[-1]
            if not _positive_finite(latest):
                targets[sym] = (0.0, "invalid latest close — trading blocked")
                alerts.append({"symbol": sym, "level": "invalid_price"})
                continue
            try:
                w = weight_fn(sym, candles)
            except Exception as exc:  # strategy bugs must not create an order
                targets[sym] = (0.0, "strategy evaluation failed — trading blocked")
                alerts.append({"symbol": sym, "level": "strategy_error", "detail": type(exc).__name__})
                continue
            prev_close = closes[-2] if len(closes) >= 2 else latest
            targets[sym] = (
                w,
                f"close={float(latest):.2f} prev_close={float(prev_close):.2f} "
                f"candles={len(candles)} target_weight={float(w):.3f}" if _finite(w) else
                f"close={float(latest):.2f} prev_close={float(prev_close):.2f} "
                f"candles={len(candles)} target_weight={w}",
            )
        else:
            targets[sym] = (0.0, "no candle data — forced flat")

    alerts.extend(
        staleness_alerts(
            candles_by_symbol,
            now_ms=int(now.timestamp() * 1000),
            max_age_days=float(max_age_days),
        )
    )
    blocked_symbols = {a["symbol"] for a in alerts if a.get("symbol")}
    targets, gross_alert = _scale_targets_to_gross_cap(targets, blocked_symbols, max_gross_exposure)
    if gross_alert is not None:
        alerts.append(gross_alert)
    prices = {
        s: float(cs[-1]["close"])
        for s, cs in candles_by_symbol.items()
        if cs and _positive_finite(cs[-1].get("close"))
    }

    decisions = decide_orders(
        targets,
        portfolio.cash,
        portfolio.positions,
        prices,
        blocked_symbols,
        as_of=now,
    )
    fills = []
    # Reallocations must release cash before consuming it. Otherwise a BUY
    # encountered before its funding SELL can clamp to zero and poison the
    # day's idempotency key despite no position change.
    execution_order = sorted(
        (d for d in decisions if d["action"] in ("SELL", "BUY")),
        key=lambda d: 0 if d["action"] == "SELL" else 1,
    )
    for d in execution_order:
        res = portfolio.rebalance(
            d["symbol"],
            d["target_weight"],
            prices[d["symbol"]],
            idem_key=d["idem_key"],
            reason=d["explanation"],
            ts=now,
            market_prices=prices,
        )
        if res is not None:
            fills.append(res)

    trade_fills = [f for f in fills if f.get("kind") == "fill"]
    report = daily_audit_report(portfolio, prices, decisions, trade_fills, alerts, as_of=now)
    if ai_note_fn is not None:
        try:
            note = ai_note_fn(report)
        except Exception as exc:
            alerts.append({"level": "ai_commentary_error", "detail": type(exc).__name__})
            report = daily_audit_report(portfolio, prices, decisions, trade_fills, alerts, as_of=now)
            report += "\n## AI commentary (advisory only — does not affect decisions)\n\n(unavailable)\n"
        else:
            if note:
                report += "\n## AI commentary (advisory only — does not affect decisions)\n\n" + note.strip() + "\n"
    report_path = write_audit_report(reports_dir, report, as_of=now)
    return {"decisions": decisions, "fills": fills, "alerts": alerts, "report_path": str(report_path), "report": report}


def run(
    symbols: str = "BTCUSDT",
    interval: str = "1d",
    poll_seconds: int = 3600,
    fast: int = 20,
    slow: int = 50,
    strategy_name: str = "trendvol",
    once: bool = False,
    reports_dir: str | Path = "reports",
    ai_note_fn=None,
) -> None:
    """Live paper-trading loop. Ctrl+C stops; state persists across restarts."""
    from .data import fetch_candles
    from .strategy import SmaCrossover, TrendVol

    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    strat: TrendVol | SmaCrossover
    if strategy_name == "trendvol":
        strat = TrendVol(50, 20, 0.25)
        label = "TrendVol(50,0.25)"
    else:
        strat = SmaCrossover(fast, slow)
        label = f"SMA {fast}/{slow}"

    def weight_fn(_sym, candles):
        return strat.weight(candles)

    portfolio = PaperPortfolio()
    print(f"Paper trading {','.join(syms)} ({interval}) | {label} | Ctrl+C to stop")
    while True:
        # enough bars for the deepest indicator warmup (TrendVol lookback+vol window)
        result = run_cycle(
            syms,
            weight_fn,
            lambda s: fetch_candles(s, interval, limit=max(fast, slow) + 250),
            portfolio,
            reports_dir,
            ai_note_fn=ai_note_fn,
        )
        for d in result["decisions"]:
            print(f"[{_utc_now():%Y-%m-%d %H:%M:%S}] {d['symbol']} {d['action']} -> {d['explanation']}")
        print(f"  audit report: {result['report_path']}")
        if once:
            break
        try:
            time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print(f"\nStopped. Cash {portfolio.cash:.2f}; positions {portfolio.positions or '{}'}")
            break
