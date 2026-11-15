"""Normalised record formats shared by producers, sinks and (later) feature code.

All times are UTC epoch milliseconds (int). Prices and volumes are floats.
"""
from __future__ import annotations

import hashlib
import html
import re
import time
from calendar import timegm
from typing import Any, Iterable

ONE_MINUTE_MS = 60_000


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------- prices

def kline_from_ws(payload: dict[str, Any]) -> dict[str, Any]:
    """Binance WebSocket kline event (combined-stream 'data' part) -> normalised kline."""
    k = payload["k"]
    return {
        "symbol": k["s"],
        "open_time": int(k["t"]),
        "close_time": int(k["T"]),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "quote_volume": float(k["q"]),
        "n_trades": int(k["n"]),
        "taker_buy_base": float(k["V"]),
        "taker_buy_quote": float(k["Q"]),
        "is_closed": bool(k["x"]),
        "event_time": int(payload.get("E", 0)),
        "ingest_time": now_ms(),
        "source": "ws",
    }


def kline_from_rest(symbol: str, row: list[Any]) -> dict[str, Any]:
    """One row of GET /api/v3/klines -> normalised kline (always a closed candle when used for gap fill)."""
    return {
        "symbol": symbol,
        "open_time": int(row[0]),
        "close_time": int(row[6]),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "quote_volume": float(row[7]),
        "n_trades": int(row[8]),
        "taker_buy_base": float(row[9]),
        "taker_buy_quote": float(row[10]),
        "is_closed": True,
        "event_time": int(row[6]),
        "ingest_time": now_ms(),
        "source": "rest",
    }


def missing_open_times(last_closed_open_time: int, now: int) -> list[int]:
    """Open times of the 1-min candles that closed after `last_closed_open_time` and before `now`."""
    out = []
    t = last_closed_open_time + ONE_MINUTE_MS
    while t + ONE_MINUTE_MS <= now:
        out.append(t)
        t += ONE_MINUTE_MS
    return out


# ---------------------------------------------------------------- news

# Base ticker -> (case-sensitive ticker patterns, case-insensitive names)
COIN_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "BTC": (("BTC", "XBT"), ("bitcoin",)),
    "ETH": (("ETH",), ("ethereum", "ether")),
    "SOL": (("SOL",), ("solana",)),
    "BNB": (("BNB",), ("binance coin", "bnb chain")),
    "XRP": (("XRP",), ("ripple",)),
    "DOGE": (("DOGE",), ("dogecoin",)),
    "ADA": (("ADA",), ("cardano",)),
    "AVAX": (("AVAX",), ()),            # "avalanche" alone is too often an ordinary word
    "LINK": (("LINK",), ("chainlink",)),
    "PEPE": (("PEPE",), ("pepe",)),
}

_TICKER_RE = {
    coin: re.compile(r"(?<![A-Za-z0-9$])\$?(?:" + "|".join(t) + r")(?![A-Za-z0-9])")
    for coin, (t, _) in COIN_ALIASES.items()
}
_NAME_RE = {
    coin: re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    for coin, (_, names) in COIN_ALIASES.items()
    if names
}
_EXTRA_NAME_RE = {"AVAX": re.compile(r"\bAvalanche\b(?=\s+(?:network|blockchain|foundation|token|price|ecosystem|c-chain))", re.IGNORECASE)}


def tag_coins(text: str, quote: str = "USDT") -> list[str]:
    """Return trading pairs mentioned in `text`, e.g. ['BTCUSDT', 'SOLUSDT'] (sorted, unique)."""
    found = set()
    for coin in COIN_ALIASES:
        if _TICKER_RE[coin].search(text):
            found.add(coin)
        elif coin in _NAME_RE and _NAME_RE[coin].search(text):
            found.add(coin)
        elif coin in _EXTRA_NAME_RE and _EXTRA_NAME_RE[coin].search(text):
            found.add(coin)
    return sorted(f"{c}{quote}" for c in found)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str | None, limit: int = 300) -> str:
    """Strip HTML tags/entities, collapse whitespace, cut to `limit` characters."""
    if not raw:
        return ""
    text = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", raw))).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def news_id(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()


def news_item(*, source: str, title: str, url: str, published_at: int | None,
              summary: str | None = None, coins: Iterable[str] | None = None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    title = clean_text(title, 300)
    summary = clean_text(summary, 300)
    tagged = sorted(set(coins or []) | set(tag_coins(f"{title} {summary}")))
    item = {
        "news_id": news_id(url),
        "source": source,
        "title": title,
        "summary": summary,
        "url": url.strip(),
        "published_at": int(published_at) if published_at else now_ms(),
        "coins": tagged,
        "ingest_time": now_ms(),
    }
    if extra:
        item.update(extra)
    return item


def struct_time_to_ms(st: time.struct_time | None) -> int | None:
    """feedparser gives UTC struct_time in *_parsed fields."""
    return timegm(st) * 1000 if st else None
