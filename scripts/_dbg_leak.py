"""Debug the leaking canary paths (throwaway)."""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from bot.strategy import TrendVol  # noqa: E402
from tests.test_no_lookahead import _candles, _corrupt_after  # noqa: E402

data = _candles(300)
strat = TrendVol(50, 20, 0.3)
i = 120
clean_w = strat.weight_at(data, i)
poisoned = _corrupt_after(data, i)
poison_w = strat.weight_at(poisoned, i)
print("clean:", clean_w, "poison:", poison_w)

# inspect what the strategy sees
s_clean = strat._series(data)
s_pois = strat._series(poisoned)
print("close[i-1] equal:", s_clean.close[i - 1] == s_pois.close[i - 1])
print("mean(i,50) equal:", s_clean.mean(i, 50) == s_pois.mean(i, 50))
print("ann_vol(i,20):", s_clean.ann_vol(i, 20), s_pois.ann_vol(i, 20))
