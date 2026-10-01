"""Derive the README headline tables from the canonical run record.

Usage:
  python scripts/derive_canonical_readme.py            # rewrite README block
  python scripts/derive_canonical_readme.py --check    # fail if out of sync

The canonical record lives at runs/canonical-v2/run.json and is the single
source of truth for headline numbers. Everything between the CANONICAL
markers in README.md is regenerated from it — hand edits inside the block
will be overwritten (and --check fails in CI when they exist).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow importing bot/ from repo root regardless of how this script is invoked.
sys.path.insert(0, str(ROOT))

from bot.canonical_identity import (  # noqa: E402
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
    resolve_primary_metrics,
)

RECORD = ROOT / "runs" / "canonical-v2" / "run.json"
README = ROOT / "README.md"
BEGIN = "<!-- CANONICAL:BEGIN — generated from runs/canonical-v2/run.json; do not edit by hand -->"
END = "<!-- CANONICAL:END -->"


def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def render(record: dict) -> str:
    r = record["results"]
    m = r["metrics"]
    primary = resolve_primary_metrics(m)
    iv = m["inv_vol_rm"]
    eq_rm = m["equal_rm"]
    eq_raw = m["equal_raw"]
    spx = m["spx"]
    bh = m["btc_bh"]
    win = r.get("window") or m.get("window")
    n_assets = r.get("n_assets_selected")
    n_folds = r.get("n_folds")

    L = []
    L.append(
        f"**Canonical rule under evaluation: `{PRIMARY_RULE_ID}`** — {PRIMARY_RULE_DESCRIPTION}. "
        f"This is the strategy the freeze executes and the forward log grades."
    )
    L.append("")
    L.append(
        f"| Metric | PRIMARY `{PRIMARY_RULE_ID}` | S&P 500 |",
    )
    L.append("|---|---|---|")
    L.append(f"| OOS CAGR | {_pct(primary['cagr'])} | {_pct(spx['cagr'])} |")
    L.append(f"| Sharpe | {primary['sharpe']:.2f} | {spx['sharpe']:.2f} |")
    L.append(f"| Max drawdown | {_fmt_pct(primary['max_drawdown'])} | {_fmt_pct(spx['max_drawdown'])} |")
    L.append("")
    L.append(
        "> Comparators below (equal-weight, raw equal-weight, inverse-vol) are context only. "
        "They are not the frozen strategy and are never the claim."
    )
    L.append("")
    L.append(f"Out-of-sample window: {win['start']} → {win['end']} "
             f"({n_folds} yearly folds, {n_assets} assets, point-in-time denominators).")
    L.append("")
    # THE PRIMARY RULE IS THE HEADLINE COLUMN.
    # Everything to its right is a comparator: reported for context, never as
    # the claim. Previously this table led with "Bot equal", a different
    # portfolio from the frozen experiment.
    L.append("```")
    header = (f"{'':18}{'PRIMARY':>14}{'cmp equal':>13}{'cmp raw eq':>12}"
              f"{'cmp inv-vol':>12}{'S&P 500':>12}{'BTC b&h':>11}")
    L.append(header)
    L.append("-" * len(header))


    cols = [primary, eq_rm, eq_raw, iv, spx, bh]
    widths = [14, 13, 12, 12, 12, 11]

    def row(label, key, kind="pct"):
        cells = []
        for blk, w in zip(cols, widths):
            v = blk[key]
            cells.append(f"{_fmt_pct(v):>{w}}" if kind == "pct" else f"{v:>{w}.2f}")
        L.append(f"{label:18}" + "".join(cells))

    row("CAGR", "cagr")
    row("Volatility", "vol")
    L.append(f"{'Sharpe (excess)':18}" + "".join(f"{b['sharpe']:>{w}.2f}" for b, w in zip(cols, widths)))
    row("Max drawdown", "max_drawdown")
    L.append(f"{'Sortino':18}" + "".join(f"{b['sortino']:>{w}.2f}" for b, w in zip(cols, widths)))
    L.append(f"{'Calmar':18}" + "".join(f"{b['calmar']:>{w}.2f}" for b, w in zip(cols, widths)))
    row("ES 95% (1d)", "es95")
    L.append(f"{'Growth of $1':18}" + "".join(f"{b['final']:>{w}.2f}" for b, w in zip(cols, widths)))
    L.append("```")
    L.append("")
    L.append(r["verdict"])
    L.append("")
    L.append("**Fixed portfolio rules** (a-priori overlays; all risk-managed to 25% vol):")
    L.append("")
    L.append("| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR | Trials |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in m["rules"]:
        # A DSR is printed only when a real trial count backs it. Older records
        # carry a trial-count-1 value, which is flagged rather than shown as if
        # it were corrected evidence.
        dsr_raw = s.get("dsr")
        available = s.get("dsr_available", dsr_raw is not None)
        n_trials = s.get("dsr_n_trials")
        if dsr_raw is not None and available:
            dsr_cell = f"{dsr_raw:.3f}"
            trials_cell = str(n_trials) if n_trials else "1 (uncorrected)"
        else:
            dsr_cell = "n/a"
            trials_cell = "—"
        L.append(
            f"| {s['name']} | {_pct(s['cagr'])} | {s['sharpe']:.2f} | {_fmt_pct(s['max_drawdown'])} "
            f"| {_pct(s['es95'])} | {s['calmar']:.2f} | {s['psr']:.3f} | {dsr_cell} | {trials_cell} |"
        )
    L.append("")
    L.append(
        f"*Provenance: reproduced from `{record['run_id']}/run.json` — commit "
        f"`{r['environment']['git_commit']}`, code sha `{r['environment']['code_fingerprint']['sha256'][:12]}…`, "
        f"strategy defs `{r['environment']['strategy_definitions_hash']['combined'][:12]}…`, "
        f"portfolio rules `{r['environment']['portfolio_rules_hash']['combined'][:12]}…`, "
        f"universe `{r['environment']['universe_hash']['combined'][:12]}…`. "
        f"Verify with `python -m bot reproduce {record['run_id']}` (frozen cache required).*"
    )
    return "\n".join(L)


def main() -> int:
    check = "--check" in sys.argv
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    rendered = render(record)
    md = README.read_text(encoding="utf-8")
    if BEGIN not in md or END not in md:
        print("markers missing from README.md — insert them around the headline block first")
        return 2
    pre = md.split(BEGIN, 1)[0]
    post = md.split(END, 1)[1]
    new = pre + BEGIN + "\n" + rendered + "\n\n" + END + post
    if new == md:
        print("README already in sync with canonical run")
        return 0
    if check:
        print("FAIL: README headline block is out of sync with runs/canonical-v2/run.json")
        return 1
    README.write_text(new, encoding="utf-8")
    print("README headline block regenerated from canonical run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
