"""README derivation from the canonical run record."""
import json

import pytest

import scripts.derive_canonical_readme as dcr


@pytest.fixture()
def record():
    return {
        "run_id": "canonical-v1",
        "results": {
            "verdict": "risk-managed portfolio OOS CAGR BEATS S&P 500 (10.8% vs 14.9%); Sharpe beats (0.70 vs 0.74)",
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
            "metrics": {
                "inv_vol_rm": {"cagr": 0.108, "vol": 0.112, "sharpe": 0.70,
                               "max_drawdown": -0.122, "sortino": 1.25, "calmar": 0.88,
                               "es95": -0.012, "final": 1.85},
                "equal_rm": {"cagr": 0.257, "vol": 0.213, "sharpe": 1.04,
                             "max_drawdown": -0.232, "sortino": 1.58, "calmar": 1.11,
                             "es95": -0.025, "final": 3.95},
                "equal_raw": {"cagr": 0.324, "vol": 0.246, "sharpe": 1.14,
                              "max_drawdown": -0.265, "sortino": 1.76, "calmar": 1.22,
                              "es95": -0.029, "final": 5.39},
                "spx": {"cagr": 0.149, "vol": 0.167, "sharpe": 0.74,
                        "max_drawdown": -0.254, "sortino": 1.06, "calmar": 0.59,
                        "es95": -0.024, "final": 2.30},
                "btc_bh": {"cagr": 0.321, "vol": 0.573, "sharpe": 0.72,
                           "max_drawdown": -0.766, "sortino": 1.07, "calmar": 0.42,
                           "es95": -0.069, "final": 5.32},
                "rules": [
                    {"name": "inv-vol (selected underlying)", "cagr": 0.108, "sharpe": 0.70,
                     "max_drawdown": -0.122, "es95": -0.012, "calmar": 0.88, "final": 1.85,
                     "psr": 0.97, "dsr": 0.90},
                    {"name": "+ tilt + crisis, banded 5% rebalance", "cagr": 0.100, "sharpe": 0.75,
                     "max_drawdown": -0.097, "es95": -0.011, "calmar": 1.02, "final": 1.78,
                     "psr": 0.98, "dsr": 0.93},
                ],
            },
        },
        "tolerance": {"rel": 1e-12, "abs": 1e-12},
    }


class TestRender:
    def test_contains_tables_and_provenance(self, record):
        text = dcr.render(record)
        assert "Out-of-sample window: 2020-08-16" in text
        assert "Bot inv-vol" in text and "BTC b&h" in text
        assert "| Rule | CAGR | Sharpe | maxDD | ES95 | Calmar | PSR | DSR |" in text
        assert "+ tilt + crisis, banded 5% rebalance" in text
        assert "reproduce canonical-v1" in text
        assert "a" * 12 in text  # code sha prefix

    def test_numbers_match_record(self, record):
        text = dcr.render(record)
        iv = record["results"]["metrics"]["inv_vol_rm"]
        assert _pct_fmt(iv["cagr"]) in text
        assert f"{iv['sharpe']:>14.2f}" in text

    def test_verdict_included(self, record):
        assert "BEATS S&P 500" in dcr.render(record)


def _pct_fmt(x):
    return f"{x * 100:.1f}%"


class TestCheckAndRewrite:
    def test_rewrite_then_check_passes(self, tmp_path, monkeypatch, record):
        monkeypatch.setattr(dcr, "RECORD", tmp_path / "rec.json")
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        dcr.RECORD.write_text(json.dumps(record), encoding="utf-8")
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
        assert "Bot inv-vol" in final

    def test_check_fails_when_out_of_sync(self, tmp_path, monkeypatch, record):
        monkeypatch.setattr(dcr, "RECORD", tmp_path / "rec.json")
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        dcr.RECORD.write_text(json.dumps(record), encoding="utf-8")
        md = dcr.BEGIN + "\nstale numbers\n" + dcr.END
        dcr.README.write_text(md, encoding="utf-8")
        # main() with --check returns 1 on drift
        sys_argv = ["derive", "--check"]
        monkeypatch.setattr("sys.argv", sys_argv)
        assert dcr.main() == 1

    def test_missing_markers_reported(self, tmp_path, monkeypatch, record):
        monkeypatch.setattr(dcr, "RECORD", tmp_path / "rec.json")
        monkeypatch.setattr(dcr, "README", tmp_path / "README.md")
        dcr.RECORD.write_text(json.dumps(record), encoding="utf-8")
        dcr.README.write_text("# no markers here\n", encoding="utf-8")
        assert dcr.main() == 2
