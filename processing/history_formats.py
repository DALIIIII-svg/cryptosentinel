"""Parsers for the historical sources. Pure functions (bytes/DataFrames in, DataFrames out)."""
from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd

from common.parquet_io import KLINE_COLUMNS, NEWS_COLUMNS
from common.schemas import clean_text, news_id, tag_coins

BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OUR_TICKERS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "PEPE"}


# ---------------------------------------------------------------- Binance klines
def to_ms(series: pd.Series) -> pd.Series:
    """Binance spot files switched from milliseconds to microseconds in 2025: normalise to ms."""
    s = series.astype("int64")
    return s.where(s < 10**14, s // 1000)


def parse_binance_zip(symbol: str, content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        raw = zf.read(name)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=BINANCE_COLUMNS)
    if not str(df.iloc[0, 0]).strip().lstrip("-").isdigit():  # some files carry a header row
        df = df.iloc[1:]
    df["open_time"] = to_ms(df["open_time"])
    df["close_time"] = to_ms(df["close_time"])
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[col] = df[col].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int64")
    df["symbol"] = symbol
    df["source"] = "hist"
    return clean_klines(df)


def clean_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible rows and duplicates; keep our column order."""
    bad = (df["high"] < df["low"]) | (df["volume"] < 0) | (df["close"] <= 0) | (df["open"] <= 0)
    df = df.loc[~bad, KLINE_COLUMNS]
    return df.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"]).reset_index(drop=True)


def verify_checksum(content: bytes, checksum_file_text: str) -> bool:
    expected = checksum_file_text.split()[0].strip().lower()
    return hashlib.sha256(content).hexdigest() == expected


def live_klines_frame(records: list[dict]) -> pd.DataFrame:
    """Records written by the S3 sink (topic prices.klines) -> kline DataFrame."""
    if not records:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame.from_records(records)
    df = df[df.get("is_closed", True).astype(bool)]
    if "replay" in df:
        df = df[~df["replay"].fillna(False).astype(bool)]
    df["source"] = df.get("source", "ws").fillna("ws")
    return clean_klines(df)


# ---------------------------------------------------------------- news
def _coins_from_categories(cats: str | float) -> list[str]:
    if not isinstance(cats, str):
        return []
    return sorted({f"{c.strip().upper()}USDT" for c in cats.split("|") if c.strip().upper() in OUR_TICKERS})


def news_frame_from_coindesk_hf(chunk: pd.DataFrame, dataset: str = "coindesk_hf") -> pd.DataFrame:
    """Hugging Face 'crypto-news-coindesk-2020-2025' columns:
    id, guid, published_on, title, body, url, imageurl, tags, categories, source, upvotes, downvotes, last_update
    Only the headline, a 300-character summary and the link are kept.
    """
    chunk = chunk.dropna(subset=["url", "title", "published_on"])
    published = pd.to_datetime(chunk["published_on"], utc=True, errors="coerce")
    chunk = chunk.loc[published.notna()]
    published = published.loc[chunk.index]
    rows = []
    for (_, r), ts in zip(chunk.iterrows(), published):
        title = clean_text(str(r["title"]), 300)
        summary = clean_text(str(r.get("body") or ""), 300)
        coins = sorted(set(_coins_from_categories(r.get("categories"))) | set(tag_coins(f"{title} {summary}")))
        rows.append({
            "news_id": news_id(str(r["url"])),
            "published_at": int(ts.value // 1_000_000),
            "source": str(r.get("source") or "unknown"),
            "title": title,
            "summary": summary,
            "url": str(r["url"]).strip(),
            "coins": coins,
            "dataset": dataset,
            "time_precision": "second",
        })
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def news_frame_generic(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Best-effort loader for other news CSVs (e.g. CryptoVision: URL, Title, Description, Date Time, Coin).
    If every timestamp is at midnight, the dataset only has dates: marked time_precision='day',
    and the agent must not use those items to explain minute-level moves.
    """
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_url, c_title = pick("url", "link"), pick("title", "headline")
    c_desc = pick("description", "summary", "body", "text")
    c_time = pick("datetime", "date", "publishedat", "publishedon", "timestamp")
    c_coin = pick("coin", "cryptocurrency", "currency")
    if not (c_url and c_title and c_time):
        raise ValueError(f"cannot find url/title/date columns in {list(df.columns)}")
    ts = pd.to_datetime(df[c_time], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    ts = ts.loc[df.index]
    day_only = bool(((ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)).all())
    rows = []
    for (_, r), t in zip(df.iterrows(), ts):
        title = clean_text(str(r[c_title]), 300)
        summary = clean_text(str(r[c_desc]) if c_desc and isinstance(r[c_desc], str) else "", 300)
        coins = set(tag_coins(f"{title} {summary}"))
        if c_coin and isinstance(r[c_coin], str):
            coins |= set(_coins_from_categories(r[c_coin].replace(",", "|")))
        rows.append({
            "news_id": news_id(str(r[c_url])), "published_at": int(t.value // 1_000_000),
            "source": dataset, "title": title, "summary": summary, "url": str(r[c_url]).strip(),
            "coins": sorted(coins), "dataset": dataset, "time_precision": "day" if day_only else "second",
        })
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def live_news_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.DataFrame.from_records(records)
    if "replay" in df:
        df = df[~df["replay"].fillna(False).astype(bool)]
    df["dataset"] = "live"
    df["time_precision"] = "second"
    for c in NEWS_COLUMNS:
        if c not in df:
            df[c] = None
    return df[NEWS_COLUMNS]
