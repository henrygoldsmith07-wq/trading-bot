"""README block derivation from the evidence document.

The block between the CANONICAL markers is rendered from
`bot.reporting.build_evidence` — the single evidence document. The `record`
argument is deliberately ignored: no caller can feed the formatter a record
the evidence layer would refuse (a stale record, a comparator block, or a
record from a different run).

The fixture world encodes the repository's founding bug on purpose: the
equal-weight comparator returns 25.7% while the PRIMARY rule under evaluation
returns 10.0% and trails the 14.9% index. If any of that ever leaks back into
the summary line or the verdict sentence, these tests fail.
"""
from __future__ import annotations

import pytest

import scripts.derive_canonical_readme as dcr
from bot.canonical_identity import (
    PRIMARY_RULE_DESCRIPTION,
    PRIMARY_RULE_ID,
    PRIMARY_RULE_STAT_NAME,
)
from bot.evidence_model import build_canonical_verdict, render_verdict_sentence

# The primary rule TRAILS the benchmark while a comparator clobbers it. This
# is the exact shape of the historical equal-weight bug.
PRIMARY = {"cagr": 0.100, "sharpe": 0.75, "max_drawdown": -0.097,
           "vol": 0.110, "sortino": 1.37, "calmar": 1.19, "es95": -0.012, "final": 1.94}
SPX = {"cagr": 0.149, "vol": 0.167, "sharpe": 0.74, "max_drawdown": -0.254,
       "sortino": 1.06, "calmar": 0.59, "es95": -0.024, "final": 2.30}
EQUAL_COMPARATOR = {"cagr": 0.257, "vol": 0.213, "sharpe": 1.04, "max_drawdown": -0.232,
                    "sortino": 1.58, "calmar": 1.11, "es95": -0.025, "final": 3.95}
INV_VOL_COMPARATOR = {"cagr": 0.108, "vol": 0.112, "sharpe": 0.70, "max_drawdown": -0.122,
                      "sortino": 1.25, "calmar": 0.88, "es95": -0.012, "final": 1.85}


@pytest.fixture()
def record() -> dict:
    """A canonical run record whose comparators beat the index and whose
    primary rule does not. Rows carry a legacy uncorrected `dsr` with no trial
    accounting — exactly the shape the evidence layer must withhold."""
    return {
        "run_id": "canonical-v2",
        "created_at": "2026-09-29T06:57:46+00:00",
        "results": {
            # A STALE hand-written sentence claiming a win. It must never
            # reach the rendered block.
            "verdict": "risk-managed portfolio OOS CAGR BEATS S&P 500 (25.7% vs 14.9%); Sharpe beats (1.04 vs 0.74)",
            "n_folds": 6,
            "n_assets_selected": 13,
            "window": {"start": "2020-08-16", "end": "2026-08-14"},
            "environment": {
                "git_commit": "77627d3",
                "code_fingerprint": {"sha256": "a" * 64},
                "strategy_definitions_hash": {"combined": "b" * 64},
                "portfolio_rules_hash": {"combined": "c" * 64},
                "universe_hash": {"combined": "d" * 64},
            },
            "per_asset": [{"symbol": "BTCUSDT", "cagr": 0.10, "sharpe": 0.7, "max_drawdown": -0.1}],
            "picks_counter": {"TrendVol(lb=50)": 40},
            "metrics": {
                "window": {"start": "2020-08-16", "end": "2026-08-14"},
                PRIMARY_RULE_ID: PRIMARY,
                "inv_vol_rm": INV_VOL_COMPARATOR,
                "equal_rm": EQUAL_COMPARATOR,
                "equal_raw": {"cagr": 0.324, "vol": 0.246, "sharpe": 1.14,
                              "max_drawdown": -0.265, "sortino": 1.76, "calmar": 1.22,
                              "es95": -0.029, "final": 5.39},
                "spx": SPX,
                "btc_bh": {"cagr": 0.321, "vol": 0.573, "sharpe": 0.72,
                           "max_drawdown": -0.766, "sortino": 1.07, "calmar": 0.42,
                           "es95": -0.069, "final": 5.32},
                "rules": [
                    {"name": "inv-vol (selected underlying)", "cagr": 0.108, "sharpe": 0.70,
                     "max_drawdown": -0.122, "es95": -0.012, "calmar": 0.88,
                     "psr": 0.999, "dsr": 0.999},
                    {"name": PRIMARY_RULE_STAT_NAME, "cagr": 0.100, "sharpe": 0.75,
                     "max_drawdown": -0.097, "es95": -0.012, "calmar": 1.19,
                     "psr": 0.98, "dsr": 0.93},
                ],
            },
        },
        "tolerance": {"rel": 1e-12, "abs": 1e-12},
    }


@pytest.fixture()
def artifacts(record) -> dict:
    """The evidence world the renderer consumes: one record, no freeze, no
    forward tape. Everything renders from THIS, never from the caller's dict."""
    return {
        "root": ".",
        "freeze": None,
        "record": record,
        "record_path": "runs/canonical-v2/run.json",
        "forward_rows": [],
        "ledger_entries": [],
        "universe_rows": [],
        "cost_rows": [],
    }


def _expected_sentence(record: dict) -> str:
    m = record["results"]["metrics"]
    return render_verdict_sentence(build_canonical_verdict(
        primary_metrics=m[PRIMARY_RULE_ID],
        benchmark_metrics=m["spx"],
        generated_from_run=record["run_id"],
    ))


class TestRender:
    def test_contains_tables_and_provenance(self, artifacts):
        text = dcr.render(artifacts=artifacts)
        assert "Out-of-sample window: 2020-08-16" in text
        assert "cmp inv-vol" in text and "BTC b&h" in text
        assert "| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR |" in text
        assert PRIMARY_RULE_STAT_NAME in text
        assert "reproduce canonical-v2" in text
        assert "a" * 12 in text  # code sha prefix

    def test_summary_table_headlines_the_primary_rule(self, artifacts, record):
        text = dcr.render(artifacts=artifacts)
        m = record["results"]["metrics"]
        primary_pct = f"{m[PRIMARY_RULE_ID]['cagr'] * 100:.1f}%"
        comparator_pct = f"{m['equal_rm']['cagr'] * 100:.1f}%"
        summary_line = next(ln for ln in text.splitlines() if ln.startswith("| OOS CAGR"))
        assert primary_pct in summary_line
        assert comparator_pct not in summary_line, "a comparator must never share the summary line"

    def test_numbers_match_the_primary_metrics(self, artifacts, record):
        text = dcr.render(artifacts=artifacts)
        primary = record["results"]["metrics"][PRIMARY_RULE_ID]
        assert f"{primary['cagr'] * 100:.1f}%" in text
        assert f"{primary['sharpe']:.2f}" in text

    def test_verdict_is_the_canonical_sentence_not_a_handwritten_claim(self, artifacts, record):
        text = dcr.render(artifacts=artifacts)
        assert _expected_sentence(record) in text
        # the record's own stale verdict string claims a comparator win; it
        # must not survive rendering in any form
        assert "BEATS S&P 500" not in text
        assert "25.7% vs 14.9%" not in text

    def test_primary_identity_is_stated(self, artifacts):
        text = dcr.render(artifacts=artifacts)
        assert PRIMARY_RULE_ID in text
        assert PRIMARY_RULE_DESCRIPTION in text

    def test_uncorrected_dsr_never_renders_as_a_number(self, artifacts):
        """The fixture rows carry legacy dsr values (0.93, 0.999) computed at
        trial-count-1. They correct for nothing and must be withheld.

        Asserted on the TABLE COLUMNS, not substrings: PSR legitimately
        happens to equal 0.999 in the fixture, so a substring check would
        confuse the PSR column for a leaked DSR.
        """
        text = dcr.render(artifacts=artifacts)
        rule_rows = [
            ln for ln in text.splitlines()
            if ln.startswith("| ") and ln.count("|") == 10 and not ln.startswith("| Rule")
            and not ln.startswith("|---")
        ]
        assert rule_rows, "the fixed-rules table must render"
        for row in rule_rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert cells[7] == "n/a", f"DSR cell must be withheld, got {cells[7]!r} in: {row}"
            assert cells[8] == "—", f"trial count must be absent for unaccounted rows, got {cells[8]!r}"
        assert "DSR n/a" in text
        assert "DSR withheld" in text

    def test_render_ignores_the_record_argument(self, artifacts):
        """The record parameter is compatibility-only: supplying a different
        (or hostile) record cannot change a single rendered byte."""
        with_record = dcr.render({"run_id": "hostile", "results": {}}, artifacts=artifacts)
        without = dcr.render(artifacts=artifacts)
        assert with_record == without

    def test_two_renders_are_byte_identical(self, artifacts):
        assert dcr.render(artifacts=artifacts) == dcr.render(artifacts=artifacts)


class TestCheckAndRewrite:
    def test_rewrite_then_check_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        md = (
            "# Headline\n\n"
            + dcr.BEGIN + "\nold stale content\n\n" + dcr.END
            + "\n\nfooter"
        )
        dcr.README.write_text(md, encoding="utf-8")
        assert dcr.main() == 0  # rewrite
        assert dcr.main() == 0  # now in sync
        final = dcr.README.read_text()
        assert "old stale content" not in final
        assert "cmp inv-vol" in final

    def test_check_fails_when_out_of_sync(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        dcr.README.write_text(dcr.BEGIN + "\nstale numbers\n" + dcr.END, encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["derive", "--check"])
        assert dcr.main() == 1

    def test_missing_markers_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        dcr.README.write_text("# no markers here\n", encoding="utf-8")
        assert dcr.main() == 2
