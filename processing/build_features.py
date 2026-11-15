"""Build the training table: processed klines -> processed/features/month=YYYY-MM/data.parquet

For each month, the previous day of candles is loaded as warm-up (rolling windows) and the
first 2 hours of the next month as look-ahead (the target needs the next 60 minutes).
Rows without a complete feature set or target are dropped.

Run:  python -m processing.build_features        (or: make features)
"""
from __future__ import annotations

import sys

import pandas as pd

from common.config import get_settings
from common.log import get_logger
from common.parquet_io import load_klines, write_df
from processing.features import FEATURE_COLUMNS, TARGET_COLUMN, WARMUP_MIN, add_target, compute_features, ready

log = get_logger("build_features")
MS_PER_MIN = 60_000


def month_bounds(month: str) -> tuple[int, int]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return int(start.value // 1_000_000), int(end.value // 1_000_000)


def neighbours(month: str) -> tuple[str, str]:
    p = pd.Period(month, freq="M")
    return str(p - 1), str(p + 1)


def build_month(month: str, symbols: list[str]) -> pd.DataFrame:
    prev_m, next_m = neighbours(month)
    start, end = month_bounds(month)
    k = load_klines(symbols, [prev_m, month, next_m])
    if k.empty:
        return k
    k = k[(k["open_time"] >= start - (WARMUP_MIN + 60) * MS_PER_MIN) & (k["open_time"] < end + 120 * MS_PER_MIN)]
    f = add_target(compute_features(k))
    f = ready(f).dropna(subset=[TARGET_COLUMN])
    f = f[(f["open_time"] >= start) & (f["open_time"] < end)]
    return f.reset_index(drop=True)


def main() -> int:
    s = get_settings()
    total = 0
    for month in s.history_months:
        f = build_month(month, s.symbol_list)
        if f.empty:
            log.warning("%s: no candles", month)
            continue
        write_df(f"processed/features/month={month}/data.parquet", f)
        total += len(f)
        log.info("%s: %d rows x %d features (+ target)", month, len(f), len(FEATURE_COLUMNS))
    log.info("done: %d rows in processed/features/", total)
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
