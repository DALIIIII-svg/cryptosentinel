"""Unit tests for Phase 3 (history formats, merging, shocks, replay). No network or Docker."""
from __future__ import annotations

import hashlib
import io
import threading
import zipfile

import numpy as np
import pandas as pd

from common.parquet_io import KLINE_COLUMNS, merge_klines, merge_news
from processing.find_shocks import attach_news, detect
from processing.history_formats import (
    live_klines_frame, news_frame_from_coindesk_hf, news_frame_generic, parse_binance_zip, verify_checksum,
)
from replay.replay import build_events, months_between, rebase, run

T0 = 1722816000000  # 2024-08-05 00:00 UTC


def make_zip(rows: list[list], header: bool = False) -> bytes:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore")
    lines += [",".join(map(str, r)) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1m-2024-08.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def binance_row(t: int, close: float = 100.0, unit: int = 1) -> list:
    return [t * unit, close, close + 1, close - 1, close, 10, (t + 59_999) * unit, 1000, 50, 6, 600, 0]


def test_parse_binance_zip_ms_and_microseconds_and_header():
    ms = parse_binance_zip("BTCUSDT", make_zip([binance_row(T0), binance_row(T0 + 60_000)]))
    us = parse_binance_zip("BTCUSDT", make_zip([binance_row(T0, unit=1000)], header=True))
    assert list(ms.columns) == KLINE_COLUMNS
    assert ms["open_time"].tolist() == [T0, T0 + 60_000]
    assert us["open_time"].tolist() == [T0] and us["close_time"].tolist() == [T0 + 59_999]
    assert (ms["source"] == "hist").all() and ms["n_trades"].dtype == "int64"


def test_bad_rows_are_dropped():
    bad = binance_row(T0)
    bad[2], bad[3] = 90, 110  # high < low
    df = parse_binance_zip("BTCUSDT", make_zip([bad, binance_row(T0 + 60_000)]))
    assert df["open_time"].tolist() == [T0 + 60_000]


def test_checksum():
    data = b"abc"
    assert verify_checksum(data, f"{hashlib.sha256(data).hexdigest()}  file.zip\n")
    assert not verify_checksum(data, "deadbeef  file.zip")


def test_merge_klines_prefers_history_and_dedupes():
    hist = parse_binance_zip("BTCUSDT", make_zip([binance_row(T0, close=100)]))
    live = hist.assign(close=999.0, source="ws")
    merged = merge_klines(live, hist)
    assert len(merged) == 1 and merged["close"].iloc[0] == 100 and merged["source"].iloc[0] == "hist"


def test_live_klines_skip_open_and_replayed():
    base = {"symbol": "BTCUSDT", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1, "quote_volume": 1,
            "n_trades": 1, "taker_buy_base": 0.5, "taker_buy_quote": 0.5, "source": "ws"}
    recs = [dict(base, open_time=T0, close_time=T0 + 59_999, is_closed=True),
            dict(base, open_time=T0 + 60_000, close_time=T0 + 119_999, is_closed=False),
            dict(base, open_time=T0 + 120_000, close_time=T0 + 179_999, is_closed=True, replay=True)]
    assert live_klines_frame(recs)["open_time"].tolist() == [T0]


def test_coindesk_hf_news_keeps_only_headline_summary_link():
    chunk = pd.DataFrame([{
        "id": "1", "published_on": "2024-08-05 00:03:10.000000", "title": "Solana slides as market sells off",
        "body": "<p>" + "Long article text. " * 50 + "</p>", "url": "https://ex.com/a",
        "categories": "SOL|MARKET|TRADING|BNB", "source": "coindesk",
    }, {"id": "2", "published_on": None, "title": "no date", "body": "", "url": "https://ex.com/b",
        "categories": "", "source": "x"}])
    df = news_frame_from_coindesk_hf(chunk)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["published_at"] == T0 + 190_000
    assert row["coins"] == ["BNBUSDT", "SOLUSDT"]
    assert len(row["summary"]) <= 300 and row["time_precision"] == "second"


def test_generic_news_detects_date_only():
    df = pd.DataFrame({"URL": ["https://a", "https://b"], "Title": ["Bitcoin up", "XRP down"],
                       "Description": ["", ""], "Date Time": ["2024-08-05", "2024-08-06"], "Coin": ["BTC", "XRP"]})
    out = news_frame_generic(df, "cryptovision")
    assert (out["time_precision"] == "day").all()
    assert out["coins"].tolist() == [["BTCUSDT"], ["XRPUSDT"]]


def test_merge_news_dedupes_on_id():
    a = pd.DataFrame([{"news_id": "x", "published_at": 2, "source": "s", "title": "t", "summary": "",
                       "url": "u", "coins": [], "dataset": "d", "time_precision": "second"}])
    assert len(merge_news(a, a)) == 1


def synthetic_klines(n: int = 600, shock_at: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        price = 100.0
        for i in range(n):
            price *= 1 + rng.normal(0, 0.0005)
            if sym == "SOLUSDT" and i == shock_at:
                price *= 0.9  # -10 % in one minute, SOL only
            t = T0 + i * 60_000
            rows.append({"symbol": sym, "open_time": t, "close_time": t + 59_999, "close": price})
    return pd.DataFrame(rows)


def test_detect_finds_the_coin_specific_shock_with_news():
    shocks = detect(synthetic_klines(), top=5)
    top = shocks.iloc[0]
    assert top["symbol"] == "SOLUSDT" and top["ret_5m_pct"] < -8
    assert abs(top["market_ret_5m_pct"]) < abs(top["ret_5m_pct"])
    news = pd.DataFrame([{"published_at": int(top["end_ms"]) - 20 * 60_000, "coins": ["SOLUSDT"],
                          "title": "Solana outage", "time_precision": "second"},
                         {"published_at": int(top["end_ms"]) + 60_000, "coins": ["SOLUSDT"],
                          "title": "after the move: must not count", "time_precision": "second"}])
    out = attach_news(shocks.head(1), news)
    assert out["coin_news_60m"].iloc[0] == 1 and out["latest_coin_headline"].iloc[0] == "Solana outage"


def test_replay_orders_events_and_rebases():
    k = synthetic_klines(n=5)
    for c in ("open", "high", "low", "volume", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"):
        k[c] = 1
    k["source"] = "hist"
    news = pd.DataFrame([{"news_id": "early", "published_at": T0 - 30 * 60_000, "coins": np.array(["BTCUSDT"]),
                          "title": "t", "source": "s", "summary": "", "url": "u", "dataset": "d", "time_precision": "second"},
                         {"news_id": "mid", "published_at": T0 + 90_000, "coins": np.array([]), "title": "t",
                          "source": "s", "summary": "", "url": "u", "dataset": "d", "time_precision": "second"}])
    ev = build_events(k, news, T0, T0 + 3 * 60_000)
    times = [e[0] for e in ev]
    assert times == sorted(times)
    assert ev[0][2] == "early" and ev[0][0] == T0          # context news sent at the start
    assert sum(1 for e in ev if e[1] == "prices.klines") == 9  # 3 symbols x 3 minutes
    assert all(e[3]["replay"] for e in ev) and isinstance(ev[0][3]["coins"], list)
    rb = rebase(ev, T0, T0 + 10**9)
    assert rb[0][0] == T0 + 10**9 and rb[-1][3]["original_close_time"] < rb[-1][3]["close_time"]

    class FakeProducer:
        def __init__(self): self.n = 0
        def produce(self, *a, **k): self.n += 1
        def poll(self, t): pass
        def flush(self, t=None): return 0
    waits = []
    fp = FakeProducer()
    sent = run(ev, speed=60, stop=threading.Event(), producer=fp, sleep=waits.append)
    assert sent == len(ev) == fp.n
    assert waits and max(waits) <= 3 * 60 / 60 + 0.5  # 3 minutes of history at 60x = about 3 s


def test_months_between_includes_news_lead():
    assert months_between(T0, T0 + 3600_000) == ["2024-08"]
    assert months_between(1722470400000 + 10 * 60_000, 1722470400000 + 3600_000) == ["2024-07", "2024-08"]
