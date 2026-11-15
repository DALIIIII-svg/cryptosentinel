"""Download the historical sources into the RAW zone of the bucket (unchanged, for traceability).

  raw/history/klines/{SYMBOL}/{SYMBOL}-1m-YYYY-MM.zip      Binance public monthly files (checksum verified)
  raw/history/news/coindesk_hf/coindesk-crypto-news.csv.gz  Hugging Face CoinDesk news dataset (one file)
  raw/history/news/{name}/{file}.csv.gz                     any extra CSV you pass with --extra-news

Already-downloaded files are skipped, so you can stop and re-run at any time.

Run:  python -m processing.backfill_history                  (prices + news)
      python -m processing.backfill_history --only prices
      python -m processing.backfill_history --extra-news cryptovision=data/downloads/cryptovision.csv
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from common.aws_clients import s3_client
from common.config import get_settings
from common.log import get_logger
from common.parquet_io import exists
from common.s3_io import ensure_bucket
from processing.history_formats import verify_checksum

log = get_logger("backfill_history")
NEWS_RAW_KEY = "raw/history/news/coindesk_hf/coindesk-crypto-news.csv.gz"


def kline_raw_key(symbol: str, month: str) -> str:
    return f"raw/history/klines/{symbol}/{symbol}-1m-{month}.zip"


def backfill_prices() -> int:
    s = get_settings()
    client = s3_client()
    done = failed = 0
    for symbol in s.symbol_list:
        for month in s.history_months:
            key = kline_raw_key(symbol, month)
            if exists(key):
                continue
            url = f"{s.binance_history_url}/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 404:
                    log.warning("not published: %s (symbol not listed yet that month?)", url)
                    continue
                resp.raise_for_status()
                chk = requests.get(url + ".CHECKSUM", timeout=30)
                if chk.ok and not verify_checksum(resp.content, chk.text):
                    raise ValueError("checksum mismatch")
                client.put_object(Bucket=s.s3_bucket, Key=key, Body=resp.content)
                done += 1
                log.info("stored %s (%.1f MB)", key, len(resp.content) / 1e6)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.error("failed %s: %s", url, exc)
    log.info("prices: %d new file(s), %d failure(s)", done, failed)
    return failed


def _upload_gz(local_csv: Path, key: str) -> None:
    s = get_settings()
    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with open(local_csv, "rb") as src, gzip.open(tmp_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        s3_client().upload_file(str(tmp_path), s.s3_bucket, key)
    finally:
        tmp_path.unlink(missing_ok=True)


def backfill_news() -> int:
    s = get_settings()
    if exists(NEWS_RAW_KEY):
        log.info("news dataset already stored at %s", NEWS_RAW_KEY)
        return 0
    cache = Path(os.getenv("CS_DOWNLOAD_DIR", "data/downloads"))
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "coindesk-crypto-news-2020-2025.csv"
    if not local.exists() or local.stat().st_size < 1_000_000:
        log.info("downloading the news dataset (about 240 MB): %s", s.news_dataset_url)
        with requests.get(s.news_dataset_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            part = local.with_suffix(".part")
            got = 0
            with open(part, "wb") as fh:
                for block in resp.iter_content(1 << 20):
                    fh.write(block)
                    got += len(block)
                    if got % (25 << 20) < (1 << 20):
                        log.info("  %d MB", got >> 20)
            part.rename(local)
    log.info("compressing and uploading to %s", NEWS_RAW_KEY)
    _upload_gz(local, NEWS_RAW_KEY)
    log.info("news dataset stored (you can delete %s to free space)", local)
    return 0


def backfill_extra(specs: list[str]) -> int:
    for spec in specs:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not name or not p.exists():
            log.error("--extra-news expects name=path/to/file.csv, got %r", spec)
            return 1
        key = f"raw/history/news/{name}/{p.stem}.csv.gz"
        _upload_gz(p, key)
        log.info("stored %s", key)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["prices", "news"])
    ap.add_argument("--extra-news", action="append", default=[], metavar="NAME=CSV")
    args = ap.parse_args()
    ensure_bucket()
    s = get_settings()
    log.info("period %s .. %s, %d symbols", s.history_start, s.history_end, len(s.symbol_list))
    failures = 0
    if args.only in (None, "prices"):
        failures += backfill_prices()
    if args.only in (None, "news"):
        failures += backfill_news()
    if args.extra_news:
        failures += backfill_extra(args.extra_news)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
