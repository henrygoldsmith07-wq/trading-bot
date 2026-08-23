"""Strategy-family structure: correlation clustering, near-duplicate
detection, and the effective (independent-bet) trial count.

The DSR punishes you per *trial*, but 85 grid candidates are not 85
independent bets — TrendVol(50) and TrendVol(75) are nearly the same trade.
Clustering OOS return streams by correlation recovers how many genuinely
distinct strategies were searched, giving a defensible (smaller) trial
count for multiple-testing corrections and surfacing near-duplicates that
inflate the pool without adding information.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("streams must be pre-aligned and length >= 2")
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def correlation_matrix(streams: dict[str, list[float]]) -> dict[tuple[str, str], float]:
    names = sorted(streams)
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[(a, b)] = pearson(streams[a], streams[b])
    return out


def near_duplicate_pairs(
    streams: dict[str, list[float]],
    corr_threshold: float = 0.995,
) -> list[dict]:
    """Pairs whose OOS return streams move together almost tick for tick —
    they add pool size (and thus apparent trials) but no new information."""
    pairs: list[dict[str, Any]] = []
    for (a, b), rho in correlation_matrix(streams).items():
        if abs(rho) >= corr_threshold:
            pairs.append({"a": a, "b": b, "correlation": rho})
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    return pairs


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def strategy_clusters(
    streams: dict[str, list[float]],
    corr_threshold: float = 0.90,
) -> dict[str, list[str]]:
    """Single-linkage clusters of strategies whose return correlations exceed
    `corr_threshold`. Returns {representative: [member names]} keyed by the
    alphabetically-first member of each cluster (deterministic)."""
    uf = _UnionFind(list(streams))
    for (a, b), rho in correlation_matrix(streams).items():
        if abs(rho) >= corr_threshold:
            uf.union(a, b)
    groups: dict[str, list[str]] = {}
    for name in streams:
        groups.setdefault(uf.find(name), []).append(name)
    out = {}
    for members in groups.values():
        rep = min(members)
        out[rep] = sorted(members)
    return dict(sorted(out.items()))


def effective_trial_count(
    streams: dict[str, list[float]],
    corr_threshold: float = 0.90,
) -> dict:
    """Effective number of independent trials searched.

    Three views reported together:
    - n_strategies: raw pool size (what naive DSR uses).
    - n_clusters: single-linkage families at `corr_threshold`.
    - n_effective: N / (1 + (N-1)*rho_avg) — the design-effect correction
      used for correlated estimators; falls back between the two extremes
      (rho=0 -> N, rho=1 -> 1).

    The conservative choice for deflation is min(n_effective, n_strategies);
    report it alongside the raw count rather than silently substituting it.
    """
    n = len(streams)
    if n < 2:
        return {"n_strategies": n, "n_clusters": n, "n_effective": n, "avg_pairwise_corr": 0.0}
    rhos = list(correlation_matrix(streams).values())
    rho_avg = mean(rhos)
    clusters = strategy_clusters(streams, corr_threshold)
    n_eff = n / (1.0 + (n - 1) * max(rho_avg, 0.0))
    return {
        "n_strategies": n,
        "n_clusters": len(clusters),
        "n_effective": n_eff,
        "avg_pairwise_corr": rho_avg,
    }
