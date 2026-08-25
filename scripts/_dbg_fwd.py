"""Debug forward canary mismatch (throwaway)."""
import pathlib
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from tests.test_no_lookahead import (  # noqa: E402
    TestForwardRunner,
    _candles,
)

t = TestForwardRunner()
tmp = pathlib.Path(tempfile.mkdtemp())
monkeypatch = None


class _MP:
    def chdir(self, p):
        import os
        os.chdir(p)


m = t._manifest(tmp)
data = _candles(200)

from bot.prospective import run_step  # noqa: E402

for label, extra in (("clean", 0), ("future", 6)):
    logp = tmp / f"l_{label}.jsonl"
    feed = [dict(c) for c in data[:91]]
    if extra:
        anchor = feed[-1]["open_time"]
        from tests.test_no_lookahead import _poison_bar  # noqa: E402

        for k in range(1, extra + 1):
            feed.append(_poison_bar(feed[-1]["close"] * 3.0,
                                    anchor + k * 86_400_000))
    now = t._now_for(90)
    r = run_step(m, lambda s, src, f=feed: (f, None), now=now, log_path=logp)
    d = r["entry"]["assets"]["AAA"]
    print(label, "| w:", d["weight"], "| target:", d.get("target"), "| ret:", round(d["sleeve_ret"], 6))
