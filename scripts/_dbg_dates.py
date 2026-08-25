"""Numeric date comparison for the canary (throwaway)."""
import sys
from datetime import UTC, datetime

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import test_no_lookahead as T  # noqa: E402

t = T.TestForwardRunner()
now = t._now_for(90)
print("now:", now.isoformat())

poison_open = 1707862400000
print("poison dt:", datetime.fromtimestamp(poison_open / 1000, tz=UTC).isoformat())

real_last_open = T.BASE_MS if hasattr(T, "BASE_MS") else 1_700_000_000_000 + 90 * 86_400_000
print("real last open:", datetime.fromtimestamp(real_last_open / 1000, tz=UTC).isoformat())
