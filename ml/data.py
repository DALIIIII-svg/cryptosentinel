"""Loading the training table and the fixed time split shared by every model.

Split (never shuffled, so the test months are truly unseen future):
  train      2024-02 .. 2024-10
  validation 2024-11          (early stopping, alert thresholds)
  test       2024-12 .. 2025-01
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_settings
from common.parquet_io import read_df
from processing.features import FEATURE_COLUMNS, TARGET_COLUMN

TRAIN_MONTHS = ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10"]
VAL_MONTHS = ["2024-11"]
TEST_MONTHS = ["2024-12", "2025-01"]
MODELS_DIR = Path(os.getenv("CS_MODELS_DIR", "models"))


def load_features(months: list[str], every: int = 1, symbols: list[str] | None = None) -> pd.DataFrame:
    """Feature rows for `months`. every=N keeps one minute out of N (saves memory for training).
    Floats are stored as float32 (the full year is about 5 million rows)."""
    symbols = symbols or get_settings().symbol_list
    frames = []
    for m in months:
        df = read_df(f"processed/features/month={m}/data.parquet")
        if df is None or df.empty:
            continue
        df = df[df["symbol"].isin(symbols)]
        if every > 1:
            df = df[(df["open_time"] // 60_000) % every == 0]
        cols = ["symbol", "open_time", "close", *FEATURE_COLUMNS, TARGET_COLUMN]
        df = df[[c for c in cols if c in df.columns]].copy()
        for c in df.columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no features for {months}: run `make features` first")
    return pd.concat(frames, ignore_index=True)


def symbol_codes(symbols: pd.Series) -> np.ndarray:
    """Stable integer code per symbol (position in the configured list)."""
    order = {s: i for i, s in enumerate(get_settings().symbol_list)}
    return symbols.map(order).fillna(-1).astype("int16").to_numpy()
