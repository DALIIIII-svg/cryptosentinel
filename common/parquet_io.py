"""Parquet in S3 (SeaweedFS locally): read, write, merge, list partitions.

Processed layout (Hive-style partitions, readable by Athena/Glue later):
  processed/klines/symbol=BTCUSDT/month=2024-08/data.parquet
  processed/news/month=2024-08/data.parquet
"""
from __future__ import annotations

import io
import json
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from common.aws_clients import s3_client
from common.config import get_settings

KLINE_COLUMNS = [
    "symbol", "open_time", "close_time", "open", "high", "low", "close", "volume",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "source",
]
NEWS_COLUMNS = [
    "news_id", "published_at", "source", "title", "summary", "url", "coins", "dataset", "time_precision",
]
# When the same candle comes from several places, keep the most trustworthy one.
SOURCE_RANK = {"hist": 0, "rest": 1, "ws": 2}


def _bucket() -> str:
    return get_settings().s3_bucket


def kline_key(symbol: str, month: str) -> str:
    return f"processed/klines/symbol={symbol}/month={month}/data.parquet"


def news_key(month: str) -> str:
    return f"processed/news/month={month}/data.parquet"


def month_of(ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ms, unit="ms", utc=True).dt.strftime("%Y-%m")


def write_df(key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    s3_client().put_object(Bucket=_bucket(), Key=key, Body=buf.getvalue())


def read_df(key: str) -> pd.DataFrame | None:
    try:
        obj = s3_client().get_object(Bucket=_bucket(), Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()


def exists(key: str) -> bool:
    try:
        s3_client().head_object(Bucket=_bucket(), Key=key)
        return True
    except ClientError:
        return False


def merge_klines(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df = df.assign(_rank=df["source"].map(SOURCE_RANK).fillna(9))
    df = df.sort_values(["open_time", "_rank"]).drop_duplicates("open_time", keep="first")
    return df.drop(columns="_rank")[KLINE_COLUMNS].reset_index(drop=True)


def merge_news(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates("news_id", keep="first").sort_values("published_at")
    return df[NEWS_COLUMNS].reset_index(drop=True)


def upsert_klines(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Merge rows into their (symbol, month) partitions. Returns rows per partition after merge."""
    out = {}
    df = df.assign(_month=month_of(df["open_time"]))
    for (sym, month), part in df.groupby(["symbol", "_month"]):
        key = kline_key(sym, month)
        merged = merge_klines(read_df(key), part.drop(columns="_month")[KLINE_COLUMNS])
        write_df(key, merged)
        out[(sym, month)] = len(merged)
    return out


def upsert_news(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    df = df.assign(_month=month_of(df["published_at"]))
    for month, part in df.groupby("_month"):
        key = news_key(month)
        merged = merge_news(read_df(key), part.drop(columns="_month")[NEWS_COLUMNS])
        write_df(key, merged)
        out[month] = len(merged)
    return out


def load_klines(symbols: Iterable[str], months: Iterable[str]) -> pd.DataFrame:
    frames = [read_df(kline_key(s, m)) for s in symbols for m in months]
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KLINE_COLUMNS)


def load_news(months: Iterable[str]) -> pd.DataFrame:
    frames = [read_df(news_key(m)) for m in months]
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)


# ---- small JSON manifest so incremental jobs skip raw files already processed
def read_manifest(name: str) -> set[str]:
    try:
        obj = s3_client().get_object(Bucket=_bucket(), Key=f"processed/_manifest/{name}.json")
        return set(json.loads(obj["Body"].read()))
    except ClientError:
        return set()


def write_manifest(name: str, keys: set[str]) -> None:
    s3_client().put_object(Bucket=_bucket(), Key=f"processed/_manifest/{name}.json",
                           Body=json.dumps(sorted(keys)).encode())
