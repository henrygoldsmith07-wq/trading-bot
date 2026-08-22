import json

from bot import cache


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fetch(sym):
        calls.append(sym)
        return [{"open_time": 1, "close": 100.0}]

    a, from_cache = cache.load_or_fetch("BTCUSDT", fetch)
    assert a == [{"open_time": 1, "close": 100.0}]
    assert not from_cache

    b, from_cache2 = cache.load_or_fetch("BTCUSDT", fetch)
    assert b == a
    assert from_cache2
    assert calls == ["BTCUSDT"]  # fetched once only


def test_ttl_expiry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache.load_or_fetch("ETHUSDT", lambda s: [{"open_time": 1, "close": 5.0}])
    # age the cache file beyond the TTL
    path = cache._cache_path("ETHUSDT")
    blob = json.loads(path.read_text())
    blob["fetched_at"] -= 100 * 3600
    path.write_text(json.dumps(blob))
    _, from_cache = cache.load_or_fetch("ETHUSDT", lambda s: [{"open_time": 1, "close": 6.0}])
    assert not from_cache


def test_clear_removes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache.load_or_fetch("BTCUSDT", lambda s: [{"open_time": 1, "close": 1.0}])
    cache.load_or_fetch("ETHUSDT", lambda s: [{"open_time": 1, "close": 2.0}])
    cache.clear()
    assert not list(tmp_path.joinpath(".cache").glob("*.json"))


def test_corrupt_cache_refetches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = cache._cache_path("BTCUSDT")
    path.parent.mkdir(exist_ok=True)
    path.write_text("not json{")
    candles, from_cache = cache.load_or_fetch("BTCUSDT", lambda s: [{"open_time": 9, "close": 9.0}])
    assert candles == [{"open_time": 9, "close": 9.0}]
    assert not from_cache
