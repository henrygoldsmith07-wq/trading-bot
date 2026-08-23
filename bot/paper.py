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
import os
import time
from datetime import UTC, datetime
from pathlib import Path

STATE_SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _checksum(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class OrderLedger:
    """Append-only JSONL record of every simulated order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, order: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(order, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn final line after a crash: ignore, state replays the rest
        return out

    def idem_keys(self) -> set[str]:
        return {e["idem_key"] for e in self.entries() if e.get("idem_key")}


def _save_state_atomic(state: dict, path: Path) -> None:
    body = {k: v for k, v in state.items() if k != "checksum"}
    wrapped = {"schema_version": STATE_SCHEMA_VERSION, **body}
    wrapped["checksum"] = _checksum({k: v for k, v in wrapped.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(wrapped, indent=2))
    os.replace(tmp, path)


def _load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        wrapped = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    checksum = wrapped.pop("checksum", None)
    if checksum != _checksum(wrapped) or wrapped.get("schema_version") != STATE_SCHEMA_VERSION:
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
        self.start_cash = start_cash
        self.fee = fee
        self.state_file = Path(state_file)
        self.ledger = OrderLedger(ledger_file)
        self._idem_keys: set[str] = set()

        state = _load_state(self.state_file)
        if state is not None:
            self.cash = float(state["cash"])
            self.positions = {s: float(q) for s, q in state["positions"].items()}
            self._idem_keys = self.ledger.idem_keys()
        else:
            recovered = self._recover_from_ledger(start_cash)
            if recovered is None:
                self.cash = start_cash
                self.positions = {}
                self._idem_keys = set()
            else:
                self.cash, self.positions = recovered

    def _recover_from_ledger(self, start_cash: float) -> tuple[float, dict[str, float]] | None:
        """Rebuild balances by replaying ledger deltas (crash recovery)."""
        entries = [e for e in self.ledger.entries() if e.get("kind") == "fill"]
        if not entries:
            return None
        cash = float(entries[0].get("cash_before", start_cash))
        positions: dict[str, float] = {}
        for e in entries:
            cash = e["cash_after"]
            sym = e["symbol"]
            positions[sym] = e["position_after"]
            if positions[sym] == 0.0:
                del positions[sym]
        self._idem_keys = {e["idem_key"] for e in entries if e.get("idem_key")}
        return cash, positions

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
        eq = self.equity(prices)
        px = prices.get(symbol)
        if px is None or px <= 0:
            raise ValueError(f"no usable price for {symbol}")
        return max(0.0, target_weight) * eq / px

    def rebalance(
        self,
        symbol: str,
        target_weight: float,
        price: float,
        idem_key: str,
        reason: str = "",
        ts: datetime | None = None,
        min_notional: float = 1.0,
    ) -> dict | None:
        """Move `symbol` toward `target_weight` of equity. Returns the fill
        record, or None when skipped (duplicate key / no price / dust).

        Idempotency: a previously-seen `idem_key` is rejected outright, so a
        retried or re-run cycle can never double-execute.
        """
        if idem_key in self._idem_keys:
            return {"skipped": "duplicate_order", "idem_key": idem_key}
        prices = {symbol: price}
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
            fee_paid = notional * self.fee
            self.cash = cash_before - notional - fee_paid
        else:
            fee_paid = notional * self.fee
            self.cash += notional - fee_paid
        self.positions[symbol] = current_qty + delta_qty
        if abs(self.positions[symbol]) < 1e-12:
            del self.positions[symbol]

        fill = {
            "kind": "fill",
            "ts": (ts or _utc_now()).isoformat(),
            "date": (ts or _utc_now()).astimezone(UTC).date().isoformat(),
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
    """Alerts for symbols whose most recent candle is older than the cutoff."""
    now = time.time() * 1000 if now_ms is None else now_ms
    alerts: list[dict] = []
    for sym, candles in candles_by_symbol.items():
        age_days = (now - candles[-1]["open_time"]) / 86_400_000 if candles else float("inf")
        if age_days > max_age_days:
            alerts.append({"symbol": sym, "age_days": round(age_days, 1), "level": "stale_data"})
    return alerts


def decide_orders(
    targets: dict[str, tuple[float, str]],  # symbol -> (target_weight, explanation)
    cash: float,
    current_positions: dict[str, float],
    prices: dict[str, float],
    stale_symbols: set[str],
    weight_epsilon: float = 1e-4,
) -> list[dict]:
    """Plan this cycle's decisions WITHOUT executing: every symbol gets an
    explanation (holds included); stale symbols are blocked up front.
    Weights follow the engine convention: fraction of TOTAL equity
    (cash + positions), so a flat symbol's weight is 0."""
    decisions: list[dict] = []
    date = _utc_now().date().isoformat()
    total_equity = cash + sum(q * prices.get(s, 0.0) for s, q in current_positions.items())
    eq = max(total_equity, 1e-9)
    for sym, (w, why) in targets.items():
        if sym in stale_symbols:
            decisions.append({"symbol": sym, "action": "blocked_stale", "explanation": why})
            continue
        px = prices.get(sym, 0.0)
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
    lines.extend(f"- {a['symbol']}: {a['level']} ({a['age_days']}d old)" if "age_days" in a else "- none" for a in alerts)
    if not alerts:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_audit_report(reports_dir: str | Path, content: str, as_of: datetime | None = None) -> Path:
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = (as_of or _utc_now()).date().isoformat()
    path = d / f"audit_{stamp}.md"
    path.write_text(content, encoding="utf-8")
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
    now: datetime | None = None,
) -> dict:
    """One full paper-trading cycle over `symbols`: fetch, explain, execute
    (idempotently), alert, and write the daily audit report."""
    now = now or _utc_now()
    candles_by_symbol = {}
    targets: dict[str, tuple[float, str]] = {}
    for sym in symbols:
        candles = fetch_candles_fn(sym)
        candles_by_symbol[sym] = candles
        if candles:
            closes = [c["close"] for c in candles]
            w = weight_fn(sym, candles)
            prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
            targets[sym] = (
                w,
                f"close={closes[-1]:.2f} prev_close={prev_close:.2f} "
                f"candles={len(candles)} target_weight={w:.3f}",
            )
        else:
            targets[sym] = (0.0, "no candle data — forced flat")

    alerts = staleness_alerts(candles_by_symbol, now_ms=int(now.timestamp() * 1000), max_age_days=max_age_days)
    stale = {a["symbol"] for a in alerts}
    prices = {s: cs[-1]["close"] for s, cs in candles_by_symbol.items() if cs}

    decisions = []
    fills = []
    for d in decide_orders(targets, portfolio.cash, portfolio.positions, prices, stale) if prices else []:
        decisions.append(d)
        if d["action"] not in ("hold", "blocked_stale"):
            res = portfolio.rebalance(
                d["symbol"],
                d["target_weight"],
                prices[d["symbol"]],
                idem_key=d["idem_key"],
                reason=d["explanation"],
                ts=now,
            )
            if res is not None:
                fills.append(res)

    report = daily_audit_report(portfolio, prices, decisions, [f for f in fills if f.get("kind") == "fill"], alerts, as_of=now)
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
