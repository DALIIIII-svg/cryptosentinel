"""Unit tests for Phase 2 parsing, tagging and batching. No network or Docker."""
from __future__ import annotations

from producers.news_producer import SeenSet, parse_cryptopanic, parse_feed
from processing.hot_writer import news_items, price_item
from processing.s3_sink import Batcher
from common.schemas import (
    clean_text, kline_from_rest, kline_from_ws, missing_open_times, news_id, tag_coins,
)

WS_EVENT = {
    "e": "kline", "E": 1731690060005, "s": "BTCUSDT",
    "k": {"t": 1731690000000, "T": 1731690059999, "s": "BTCUSDT", "i": "1m", "f": 1, "L": 2,
          "o": "91000.10", "c": "91050.00", "h": "91100.00", "l": "90990.50", "v": "12.5",
          "n": 840, "x": True, "q": "1137500.0", "V": "7.0", "Q": "637000.0", "B": "0"},
}
REST_ROW = [1731690000000, "91000.10", "91100.00", "90990.50", "91050.00", "12.5",
            1731690059999, "1137500.0", 840, "7.0", "637000.0", "0"]


def test_ws_and_rest_klines_match():
    a, b = kline_from_ws(WS_EVENT), kline_from_rest("BTCUSDT", REST_ROW)
    for field in ("symbol", "open_time", "close_time", "open", "high", "low", "close", "volume",
                  "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "is_closed"):
        assert a[field] == b[field], field
    assert a["close"] == 91050.0 and a["n_trades"] == 840 and a["is_closed"] is True


def test_missing_open_times_only_closed_candles():
    last = 1_000_020_000 // 60_000 * 60_000
    now = last + 3 * 60_000 + 30_000  # 3.5 minutes later: 2 closed candles + 1 open one
    assert missing_open_times(last, now) == [last + 60_000, last + 120_000]
    assert missing_open_times(last, last + 90_000) == []


def test_tag_coins_tickers_and_names():
    assert tag_coins("Bitcoin tops $95K as ETH lags") == ["BTCUSDT", "ETHUSDT"]
    assert tag_coins("Solana and Cardano rally; $PEPE surges") == ["ADAUSDT", "PEPEUSDT", "SOLUSDT"]
    assert tag_coins("Chainlink partners with SWIFT") == ["LINKUSDT"]
    assert tag_coins("Avalanche network upgrade goes live") == ["AVAXUSDT"]


def test_tag_coins_avoids_ordinary_words():
    assert tag_coins("Click the link to read more about the sol-gel process in Canada") == []
    assert tag_coins("An avalanche of liquidations hit the market") == []
    assert tag_coins("Regulators in CANADA discuss stablecoins") == []


def test_clean_text_strips_html_and_cuts():
    assert clean_text("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
    assert len(clean_text("x" * 500, 300)) == 300


def test_news_id_is_stable():
    assert news_id("https://a.com/x ") == news_id("https://a.com/x")


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>Solana ETF filing lifts SOL</title><link>https://ex.com/sol-etf</link>
<description>&lt;p&gt;Asset managers filed &lt;b&gt;today&lt;/b&gt;.&lt;/p&gt;</description>
<pubDate>Sat, 15 Nov 2026 16:00:00 GMT</pubDate></item>
<item><title>No link here</title></item>
</channel></rss>"""


def test_parse_feed():
    items = parse_feed("demo", RSS)
    assert len(items) == 1
    it = items[0]
    assert it["coins"] == ["SOLUSDT"]
    assert it["summary"] == "Asset managers filed today ."
    assert it["published_at"] == 1794758400000  # 2026-11-15 16:00:00 UTC
    assert it["source"] == "demo"


def test_parse_cryptopanic_is_defensive():
    payload = {"results": [
        {"title": "XRP wins case", "original_url": "https://ex.com/xrp", "published_at": "2026-11-15T16:00:00Z",
         "currencies": [{"code": "XRP"}], "votes": {"positive": 3}},
        {"title": "Only slug", "id": 7, "slug": "only-slug"},
        {"title": "Nothing usable"},
    ]}
    items = parse_cryptopanic(payload)
    assert [i["url"] for i in items] == ["https://ex.com/xrp", "https://cryptopanic.com/news/7/only-slug"]
    assert items[0]["coins"] == ["XRPUSDT"] and items[0]["votes"] == {"positive": 3}


def test_seen_set_is_bounded():
    s = SeenSet(maxlen=2)
    assert s.add("a") and s.add("b") and not s.add("a")
    s.add("c")  # evicts "a"
    assert s.add("a")


def test_batcher_flushes_on_size_and_age():
    t = [0.0]
    b = Batcher(max_records=3, max_age_s=60, clock=lambda: t[0])
    b.add("x", {"i": 1})
    assert not b.due()
    b.add("x", {"i": 2}); b.add("y", {"i": 3})
    assert b.due()
    assert b.drain() == {"x": [{"i": 1}, {"i": 2}], "y": [{"i": 3}]}
    b.add("x", {"i": 4})
    t[0] = 61
    assert b.due() and b.size == 1


def test_hot_items():
    k = kline_from_ws(WS_EVENT)
    p = price_item(k)
    assert p["symbol"] == "BTCUSDT" and p["open_time"] == 1731690000000
    assert p["expires_at"] == 1731690059 + 48 * 3600
    n = {"news_id": "abc", "published_at": 1794758400000, "title": "t", "url": "u", "source": "s", "coins": []}
    items = news_items(n)
    assert len(items) == 1 and items[0]["coin"] == "MARKET"
    assert items[0]["sk"] == "1794758400000#abc"
    n["coins"] = ["BTCUSDT", "ETHUSDT"]
    assert [i["coin"] for i in news_items(n)] == ["BTCUSDT", "ETHUSDT"]
