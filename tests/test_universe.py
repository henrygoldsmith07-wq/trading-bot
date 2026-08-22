import json

from bot.universe import parse_symbols


def _ticker(symbols_vols):
    return json.dumps([{"symbol": s, "quoteVolume": v} for s, v in symbols_vols])


def test_orders_by_volume_and_filters_stables():
    raw = _ticker(
        [
            ("BTCUSDT", "5e9"),
            ("ETHUSDT", "3e9"),
            ("USDCUSDT", "9e9"),  # stablecoin pair — excluded
            ("BTCUPUSDT", "8e9"),  # leveraged token — excluded
            ("BNBUSDT", "1e9"),
            ("SOLUSDT", "2e9"),
            ("FOOBAR", "7e9"),  # wrong quote — excluded
        ]
    )
    assert parse_symbols(raw, n=5) == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def test_limits_to_n():
    raw = _ticker([(f"A{i}USDT", str(10 - i)) for i in range(8)])
    assert len(parse_symbols(raw, n=3)) == 3


def test_bad_volume_rows_skipped():
    raw = json.dumps(
        [
            {"symbol": "BTCUSDT", "quoteVolume": "1e9"},
            {"symbol": "XRPUSDT", "quoteVolume": "oops"},
            {"symbol": "NOVOL"},
        ]
    )
    assert parse_symbols(raw) == ["BTCUSDT"]


def test_dedupes_when_n_exceeds_rows():
    raw = _ticker([("BTCUSDT", "1"), ("ETHUSDT", "2")])
    assert parse_symbols(raw, n=10) == ["ETHUSDT", "BTCUSDT"]
