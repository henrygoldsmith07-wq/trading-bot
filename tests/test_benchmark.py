from bot.benchmark import equity_metrics, parse_fred_csv, parse_yahoo_json, slice_window

FRED_SAMPLE = """observation_date,SP500
2024-01-02,4709.50
2024-01-03,4704.81
2024-01-04,4688.61
2024-01-05,.
2024-01-08,4763.54
"""

YAHOO_SAMPLE = {
    "chart": {
        "result": [
            {
                "timestamp": [1704171600, 1704258000, 1704344400],
                "indicators": {"quote": [{"close": [4709.5, None, 4704.8]}]},
            }
        ]
    }
}


def test_parse_fred_skips_missing_values():
    rows = parse_fred_csv(FRED_SAMPLE)
    assert len(rows) == 4
    assert rows[0]["close"] == 4709.50
    assert str(rows[0]["date"]) == "2024-01-02"


def test_parse_yahoo_skips_none_closes():
    import json

    rows = parse_yahoo_json(json.dumps(YAHOO_SAMPLE))
    assert len(rows) == 2
    assert rows[1]["close"] == 4704.8


def test_slice_window_filters_by_date():
    import datetime as dt

    rows = parse_fred_csv(FRED_SAMPLE)
    window = slice_window(rows, dt.date(2024, 1, 3), dt.date(2024, 1, 8))
    assert [str(r["date"]) for r in window] == ["2024-01-03", "2024-01-04", "2024-01-08"]


def test_equity_metrics_growth():
    import datetime as dt

    rows = [
        {"date": dt.date(2024, 1, 1), "close": 100.0},
        {"date": dt.date(2024, 6, 30), "close": 110.0},
        {"date": dt.date(2024, 12, 31), "close": 121.0},
    ]
    m = equity_metrics(rows)
    assert abs(m["final"] - 1.21) < 1e-9
    # doubling over ~1 calendar year
    assert m["cagr"] > 0.15
    assert m["max_drawdown"] == 0.0
