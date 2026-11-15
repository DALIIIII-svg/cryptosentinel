"""Where the agent's tools get their facts. Two interchangeable sources:

  DynamoSource   hot tables (live and replay): prices-hot, news           (default for the consumer)
  ParquetSource  processed history in S3 (evaluation on past events)

Both return plain pandas/dicts, so the graph does not care which one is used.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from boto3.dynamodb.conditions import Key

from common.aws_clients import dynamodb_resource
from common.config import get_settings
from common.parquet_io import load_klines, load_news

MARKET_KEYS = ("MARKET", "BTCUSDT", "ETHUSDT")  # news about the whole market or its two leaders


def _months(t0: int, t1: int) -> list[str]:
    return list(pd.period_range(pd.to_datetime(t0, unit="ms"), pd.to_datetime(t1, unit="ms"), freq="M").strftime("%Y-%m"))


class ParquetSource:
    def __init__(self) -> None:
        self.symbols = get_settings().symbol_list

    @lru_cache(maxsize=4)
    def _klines(self, months: tuple[str, ...]) -> pd.DataFrame:
        return load_klines(self.symbols, list(months))

    @lru_cache(maxsize=4)
    def _news(self, months: tuple[str, ...]) -> pd.DataFrame:
        return load_news(list(months))

    def candles(self, t0: int, t1: int) -> pd.DataFrame:
        k = self._klines(tuple(_months(t0, t1)))
        return k[(k["open_time"] >= t0) & (k["open_time"] < t1)]

    def news(self, symbol: str, t0: int, t1: int) -> list[dict]:
        n = self._news(tuple(_months(t0, t1)))
        n = n[(n["published_at"] >= t0) & (n["published_at"] <= t1)]
        if "time_precision" in n:
            n = n[n["time_precision"].fillna("second") == "second"]
        keep = n["coins"].apply(lambda c: (len(c) == 0) or symbol in list(c) or any(k in list(c) for k in MARKET_KEYS))
        out = n[keep].sort_values("published_at")
        return [{**r, "coins": list(r["coins"])} for r in out[["news_id", "published_at", "source", "title",
                                                                "summary", "url", "coins"]].to_dict("records")]


class DynamoSource:
    def __init__(self) -> None:
        s = get_settings()
        db = dynamodb_resource()
        self.prices = db.Table(s.table_name("prices-hot"))
        self.news_t = db.Table(s.table_name("news"))
        self.symbols = s.symbol_list

    def candles(self, t0: int, t1: int) -> pd.DataFrame:
        rows = []
        for sym in self.symbols:
            resp = self.prices.query(KeyConditionExpression=Key("symbol").eq(sym) & Key("open_time").between(t0, t1 - 1))
            rows += resp.get("Items", [])
        if not rows:
            return pd.DataFrame(columns=["symbol", "open_time", "close", "volume"])
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "symbol" and c != "replay":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def news(self, symbol: str, t0: int, t1: int) -> list[dict]:
        seen, out = set(), []
        for coin in (symbol, *MARKET_KEYS):
            resp = self.news_t.query(KeyConditionExpression=Key("coin").eq(coin) & Key("sk").between(
                f"{t0:013d}", f"{t1:013d}#~"))
            for it in resp.get("Items", []):
                if it["news_id"] in seen:
                    continue
                seen.add(it["news_id"])
                out.append({"news_id": it["news_id"], "published_at": int(it["published_at"]),
                            "source": it.get("source", ""), "title": it.get("title", ""),
                            "summary": it.get("summary", ""), "url": it.get("url", ""),
                            "coins": list(it.get("coins", []))})
        return sorted(out, key=lambda x: x["published_at"])
