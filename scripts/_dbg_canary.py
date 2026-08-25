"""Instrumented reproduction of the future-row canary (throwaway)."""
import pathlib
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import json  # noqa: E402

import test_no_lookahead as T  # noqa: E402


def load_entry(log):
    lines = [line for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
    return json.loads(lines[-1])
from bot import prospective as P  # noqa: E402

t = T.TestForwardRunner()
tmp = pathlib.Path(tempfile.mkdtemp())
m = t._manifest(tmp)
data = T._candles(200)

orig_sanity = P._bar_sanity_problem


def spy(candle):
    r = orig_sanity(candle)
    print("sanity:", candle["open_time"], candle.get("close"), "->", r)
    return r


P._bar_sanity_problem = spy

today_last = [dict(c) for c in data[:91]]
with_f = today_last + [
    T._poison_bar(today_last[-1]["close"] * 3.0,
                  today_last[-1]["open_time"] + k * 86_400_000)
    for k in range(1, 6)
]

for label, feed in (("clean", today_last), ("poison", with_f)):
    log = tmp / f"l_{label}.jsonl"
    if log.exists():
        log.unlink()
    r = P.run_step(m, lambda s, src, f=feed: (f, None),
                   now=t._now_for(90), log_path=log)
    e = load_entry(log)
    print(label, "exec:", e["assets"]["AAA"]["exec_price"],
          "ret:", round(e["assets"]["AAA"]["sleeve_ret"], 6))
