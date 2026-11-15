"""RAW zone -> PROCESSED zone (Parquet), incremental and idempotent.

Sources handled:
  raw/history/klines/**.zip            Binance monthly files
  raw/prices.klines/dt=*/hour=*/*.gz   live candles from the S3 sink (Phase 2)
  raw/history/news/**.csv.gz           historical news datasets
  raw/news.raw/dt=*/hour=*/*.gz        live news from the S3 sink

Output:
  processed/klines/symbol=X/month=YYYY-MM/data.parquet   (deduplicated on open_time)
  processed/news/month=YYYY-MM/data.parquet               (deduplicated on news_id)

Only candles and news inside HISTORY_START..HISTORY_END are kept from the history files
(live data is always kept). A manifest remembers which raw files were already processed.
This is plain pandas, so the same file can later run as an AWS Glue Python Shell job.

Run:  python -m processing.raw_to_parquet            (then a data quality report is printed)
      python -m processing.raw_to_parquet --rebuild  (ignore the manifest, redo everything)
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict

import pandas as pd

from common.aws_clients import s3_client
from common.config import get_settings
from common.log import get_logger
from common.parquet_io import (
    NEWS_COLUMNS, kline_key, month_of, read_df, read_manifest,
    upsert_klines, upsert_news, write_manifest,
)
from common.s3_io import list_keys, read_jsonl_gz
from processing.history_formats import (
    live_klines_frame, live_news_frame, news_frame_from_coindesk_hf, news_frame_generic, parse_binance_zip,
)

log = get_logger("raw_to_parquet")


def _in_period(ms: pd.Series) -> pd.Series:
    s = get_settings()
    m = month_of(ms)
    return (m >= s.history_start) & (m <= s.history_end)


def process_klines(done: set[str]) -> set[str]:
    s = get_settings()
    client = s3_client()
    new_done = set()
    # history zips: one file = one symbol-month
    for key in sorted(list_keys("raw/history/klines/")):
        if key in done or not key.endswith(".zip"):
            continue
        symbol = key.split("/")[3]
        body = client.get_object(Bucket=s.s3_bucket, Key=key)["Body"].read()
        df = parse_binance_zip(symbol, body)
        df = df[_in_period(df["open_time"])]
        if len(df):
            upsert_klines(df)
        new_done.add(key)
        log.info("klines %s: %d rows", key.rsplit("/", 1)[-1], len(df))
    # live files from the sink: batch them to limit partition rewrites
    live_keys = [k for k in sorted(list_keys("raw/prices.klines/")) if k not in done]
    for i in range(0, len(live_keys), 200):
        batch = live_keys[i:i + 200]
        records = [r for k in batch for r in read_jsonl_gz(k)]
        df = live_klines_frame(records)
        if len(df):
            upsert_klines(df)
        new_done.update(batch)
        log.info("live klines: %d file(s), %d rows", len(batch), len(df))
    return new_done


def process_news(done: set[str]) -> set[str]:
    s = get_settings()
    client = s3_client()
    new_done = set()
    for key in sorted(list_keys("raw/history/news/")):
        if key in done or not key.endswith(".csv.gz"):
            continue
        dataset = key.split("/")[3]
        body = client.get_object(Bucket=s.s3_bucket, Key=key)["Body"].read()
        total = 0
        reader = pd.read_csv(io.BytesIO(body), compression="gzip", chunksize=50_000,
                             dtype=str, keep_default_na=False, on_bad_lines="skip")
        for chunk in reader:
            chunk = chunk.replace({"": None})
            if dataset == "coindesk_hf":
                # filter on the period before the slower row-by-row normalisation
                pub = pd.to_datetime(chunk["published_on"], utc=True, errors="coerce")
                m = pub.dt.strftime("%Y-%m")
                chunk = chunk[(m >= s.history_start) & (m <= s.history_end)]
                frame = news_frame_from_coindesk_hf(chunk) if len(chunk) else pd.DataFrame(columns=NEWS_COLUMNS)
            else:
                frame = news_frame_generic(chunk, dataset)
                frame = frame[_in_period(frame["published_at"])]
            if len(frame) and s.news_time_shift_minutes:
                frame = frame.assign(published_at=frame["published_at"] + s.news_time_shift_minutes * 60_000)
            if len(frame):
                upsert_news(frame)
                total += len(frame)
        new_done.add(key)
        log.info("news %s: %d items in the period", dataset, total)
    live_keys = [k for k in sorted(list_keys("raw/news.raw/")) if k not in done]
    for i in range(0, len(live_keys), 500):
        batch = live_keys[i:i + 500]
        frame = live_news_frame([r for k in batch for r in read_jsonl_gz(k)])
        if len(frame):
            upsert_news(frame)
        new_done.update(batch)
        log.info("live news: %d file(s), %d items", len(batch), len(frame))
    return new_done


def quality_report() -> None:
    """Missing 1-minute candles per symbol and month (expected = minutes between first and last candle)."""
    s = get_settings()
    print("\nData quality: candles per symbol and month (missing = gaps inside the month)")
    print(f"  {'symbol':10s} {'month':8s} {'rows':>7s} {'missing':>8s}")
    worst = 0.0
    for sym in s.symbol_list:
        for month in s.history_months:
            df = read_df(kline_key(sym, month))
            if df is None or not len(df):
                print(f"  {sym:10s} {month:8s} {'-':>7s} {'no data':>8s}")
                continue
            expected = (int(df["open_time"].max()) - int(df["open_time"].min())) // 60_000 + 1
            missing = expected - len(df)
            worst = max(worst, missing / expected)
            if missing:
                print(f"  {sym:10s} {month:8s} {len(df):7d} {missing:8d}")
    print(f"  (months with no gap are not listed) worst gap ratio: {worst:.4%}")

    news = defaultdict(int)
    for month in s.history_months:
        df = read_df(f"processed/news/month={month}/data.parquet")
        news[month] = 0 if df is None else len(df)
    print("\nNews items per month: " + ", ".join(f"{m}: {n}" for m, n in news.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="ignore the manifest and reprocess every raw file")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--rebuild-news", action="store_true",
                    help="delete processed news and rebuild it (after changing NEWS_TIME_SHIFT_MINUTES)")
    args = ap.parse_args()
    if args.rebuild_news:
        s = get_settings()
        client = s3_client()
        for key in list(list_keys("processed/news/")):
            client.delete_object(Bucket=s.s3_bucket, Key=key)
        write_manifest("news", set())
        log.info("processed news deleted; rebuilding with a time shift of %d min", s.news_time_shift_minutes)
    for name, fn in (("klines", process_klines), ("news", process_news)):
        done = set() if args.rebuild else read_manifest(name)
        new = fn(done)
        write_manifest(name, done | new)
        log.info("%s: %d raw file(s) processed this run", name, len(new))
    if not args.no_report:
        quality_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
