"""Code-identity sealing: the freeze must pin the implementation, not just
the configuration. These tests prove the fingerprint is deterministic across
platform line-ending differences, sensitive to any source change, and that
both load_freeze and run_step REFUSE to operate on mismatched code.
"""
import json
from datetime import UTC, datetime

import pytest

from bot.identity import CODE_FINGERPRINT_ALGO, code_fingerprint, verify_freeze_code


class TestFingerprint:
    def test_deterministic_across_calls(self):
        assert code_fingerprint() == code_fingerprint()

    def test_line_endings_do_not_change_digest(self, tmp_path):
        """A Windows CRLF working tree and a Linux LF checkout of the same
        commit MUST hash identically (autocrlf would otherwise make the seal
        platform-dependent)."""
        import shutil

        src = tmp_path / "src"
        dst = tmp_path / "dst"
        for d in (src, dst):
            (d / "bot").mkdir(parents=True)
            (d / "pyproject.toml").write_text('[project]\nname="x"\n', encoding="utf-8")
            shutil.copy("bot/strategy.py", d / "bot" / "strategy.py", follow_symlinks=True)
        # identical logical content, different EOL conventions
        text = (src / "bot" / "strategy.py").read_bytes().replace(b"\r\n", b"\n")
        (src / "bot" / "strategy.py").write_bytes(text)
        (dst / "bot" / "strategy.py").write_bytes(text.replace(b"\n", b"\r\n"))
        assert code_fingerprint(src) == code_fingerprint(dst)

    def test_sensitive_to_any_source_change(self, tmp_path):
        import shutil

        a = tmp_path / "a"
        b = tmp_path / "b"
        for d in (a, b):
            (d / "bot").mkdir(parents=True)
            shutil.copy("bot/strategy.py", d / "bot" / "strategy.py")
            (d / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        original = (a / "bot" / "strategy.py").read_bytes()
        # one byte of real implementation change: a comment flip inside a docstring
        mutated = original.replace(b"Long-only.", b"Short-only.") or original + b"\n# x"
        (b / "bot" / "strategy.py").write_bytes(mutated)
        assert code_fingerprint(a) != code_fingerprint(b)

    def test_algo_id_is_stable_and_reported(self):
        assert CODE_FINGERPRINT_ALGO == "sha256-lf-v1"


def _manifest_with(code_sha):
    return {
        "frozen_at": "2026-08-23T00:00:00+00:00",
        "git_commit_at_freeze": "abc1234",
        "code_fingerprint_algo": CODE_FINGERPRINT_ALGO,
        "code_sha256": code_sha,
    }


class TestVerifyFreezeCode:
    def test_ok_when_matching_running_tree(self):
        verify_freeze_code(_manifest_with(code_fingerprint()))  # no raise

    def test_refuses_on_mismatch(self):
        with pytest.raises(ValueError, match="CODE MISMATCH"):
            verify_freeze_code(_manifest_with("0" * 64))

    def test_refuses_pre_sealing_manifest(self):
        m = {"frozen_at": "old", "code_sha256": None}
        with pytest.raises(ValueError, match="no code fingerprint"):
            verify_freeze_code(m)

    def test_refuses_unknown_algorithm(self):
        m = {"code_fingerprint_algo": "md5-v0", "code_sha256": "x"}
        with pytest.raises(ValueError, match="unknown code-fingerprint algorithm"):
            verify_freeze_code(m)


class TestFreezeIntegration:
    def _make_freeze(self, tmp_path, monkeypatch):
        from bot.algorithm import build_algorithm
        from bot.prospective import create_freeze

        monkeypatch.chdir(tmp_path)
        manifest = create_freeze(
            assets=[
                {
                    "symbol": "BTCUSDT",
                    "source": "binance",
                    "periods_per_year": 365,
                    "strategy": __import__("bot.strategy", fromlist=["TrendVol"]).TrendVol(50, 20, 0.3),
                }
            ],
            frictions={"fee": 0.001},
            algorithm=build_algorithm(with_pool_version=False),
            path=tmp_path / "freeze.json",
            now=datetime(2026, 8, 23, tzinfo=UTC),
            git_commit="deadbeef",
        )
        return manifest

    def test_created_manifest_seals_real_code(self, tmp_path, monkeypatch):
        m = self._make_freeze(tmp_path, monkeypatch)
        assert m["code_sha256"] == code_fingerprint()

    def test_load_freeze_verifies_code_by_default(self, tmp_path, monkeypatch):
        from bot import prospective as P

        self._make_freeze(tmp_path, monkeypatch)
        # tamper ONLY the code field (config untouched -> config check passes)
        raw = json.loads((tmp_path / "freeze.json").read_text())
        raw["code_sha256"] = "f" * 64
        (tmp_path / "freeze.json").write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="CODE MISMATCH"):
            P.load_freeze(tmp_path / "freeze.json")

    def test_load_freeze_can_skip_code_check_explicitly(self, tmp_path, monkeypatch):
        from bot import prospective as P

        self._make_freeze(tmp_path, monkeypatch)
        raw = json.loads((tmp_path / "freeze.json").read_text())
        raw["code_sha256"] = "f" * 64
        (tmp_path / "freeze.json").write_text(json.dumps(raw))
        m = P.load_freeze(tmp_path / "freeze.json", verify_code=False)
        assert m["config"]["assets"][0]["symbol"] == "BTCUSDT"

    def test_run_step_refuses_mismatched_code_before_any_trading(self, tmp_path, monkeypatch):
        from bot import prospective as P

        manifest = self._make_freeze(tmp_path, monkeypatch)
        tampered = dict(manifest)
        tampered["code_sha256"] = "e" * 64

        def boom_fetcher(sym, source):  # must never be reached
            raise AssertionError("run_step traded on code that does not match the freeze")

        with pytest.raises(ValueError, match="CODE MISMATCH"):
            P.run_step(tampered, boom_fetcher)

    def test_run_step_runs_when_code_matches(self, tmp_path, monkeypatch):
        from bot import prospective as P

        manifest = self._make_freeze(tmp_path, monkeypatch)
        candles = [{"open_time": 86_400_000 * i, "close": 100.0 + i} for i in range(30)]
        result = P.run_step(manifest, lambda s, src: (candles, None), log_path=tmp_path / "log.jsonl")
        assert result["status"] in ("logged", "already_logged")
