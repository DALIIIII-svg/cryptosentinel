"""THE feature module: imported by offline training AND live scoring, so both compute exactly the
same numbers (no train/serve skew). Pure pandas, no I/O.

Input : 1-minute candles of several symbols (long format: symbol, open_time, open, high, low,
        close, volume, n_trades, taker_buy_base, ...).
Output: one row per (symbol, open_time) with the columns in FEATURE_COLUMNS.

Every feature at minute t uses candles <= t only. The training target (future volatility)
is added by a separate function, add_target(), which is never called in live scoring.

Features (log returns r, 1-min grid; windows counted in minutes):
  ret_1m, ret_5m, ret_15m, ret_60m   log return over the last 1/5/15/60 minutes
  rv_5, rv_15, rv_60, rv_1d          realised volatility = sqrt(sum r^2) over 5/15/60/1440 min
  vol_z, trades_z                    z-score of log(1+volume) / log(1+trades) vs the last 24 h
  imbalance                          2 * taker_buy_base / volume - 1   (-1 all sells .. +1 all buys)
  range_pct                          (high - low) / close
  mkt_ret_1m, mkt_ret_5m, mkt_rv_15  equal-weight market (mean of the coins present)
  idio_ret_5m                        ret_5m - mkt_ret_5m   (coin-specific part of the move)
  hour_sin, hour_cos                 time of day (UTC)
  filled                             1 if the candle was missing and filled (no trades)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE_MS = 60_000
DAY_MIN = 1440
WARMUP_MIN = DAY_MIN                 # rows before this are incomplete (rv_1d, z-scores)
BUFFER_MIN = DAY_MIN + 120           # history the live consumer keeps per symbol

FEATURE_COLUMNS = [
    "ret_1m", "ret_5m", "ret_15m", "ret_60m",
    "rv_5", "rv_15", "rv_60", "rv_1d",
    "vol_z", "trades_z", "imbalance", "range_pct",
    "mkt_ret_1m", "mkt_ret_5m", "mkt_rv_15", "idio_ret_5m",
    "hour_sin", "hour_cos", "filled",
]
TARGET_COLUMN = "y_log_rv_next_60"
EPS = 1e-8


def regularize(k: pd.DataFrame) -> pd.DataFrame:
    """Put each symbol on a complete 1-minute grid. Missing candles: close carried forward,
    open=high=low=close, zero volume/trades, filled=1."""
    out = []
    for sym, g in k.sort_values("open_time").groupby("symbol", sort=False):
        g = g.drop_duplicates("open_time").set_index("open_time")
        grid = np.arange(g.index.min(), g.index.max() + MINUTE_MS, MINUTE_MS)
        g = g.reindex(grid)
        filled = g["close"].isna()
        g["close"] = g["close"].ffill()
        for c in ("open", "high", "low"):
            g[c] = g[c].fillna(g["close"])
        for c in ("volume", "n_trades", "taker_buy_base"):
            g[c] = g[c].fillna(0.0)
        g["filled"] = filled.astype("int8")
        g["symbol"] = sym
        out.append(g.rename_axis("open_time").reset_index())
    cols = ["symbol", "open_time", "open", "high", "low", "close", "volume", "n_trades", "taker_buy_base", "filled"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window, min_periods=60).mean()
    std = x.rolling(window, min_periods=60).std()
    return ((x - mean) / std.replace(0, np.nan)).fillna(0.0)


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"]
    r = np.log(close / close.shift(1))
    r2 = r.pow(2)
    out = pd.DataFrame({"symbol": g["symbol"].values, "open_time": g["open_time"].values,
                        "close": close.values}, index=g.index)
    out["ret_1m"] = r
    for n in (5, 15, 60):
        out[f"ret_{n}m"] = np.log(close / close.shift(n))
    for n, name in ((5, "rv_5"), (15, "rv_15"), (60, "rv_60"), (DAY_MIN, "rv_1d")):
        out[name] = np.sqrt(r2.rolling(n, min_periods=n).sum())
    out["vol_z"] = _rolling_z(np.log1p(g["volume"]), DAY_MIN)
    out["trades_z"] = _rolling_z(np.log1p(g["n_trades"]), DAY_MIN)
    vol = g["volume"]
    out["imbalance"] = np.where(vol > 0, 2 * g["taker_buy_base"] / vol.where(vol > 0, 1) - 1, 0.0)
    out["range_pct"] = (g["high"] - g["low"]) / close
    out["filled"] = g["filled"].values
    return out


def compute_features(klines: pd.DataFrame) -> pd.DataFrame:
    """Features for every (symbol, minute) in `klines`. Rows inside the first day are incomplete
    (NaN in rv_1d etc.); callers drop them with `ready()`."""
    if klines.empty:
        return pd.DataFrame(columns=["symbol", "open_time", "close", *FEATURE_COLUMNS])
    grid = regularize(klines)
    parts = [_per_symbol(g.reset_index(drop=True)) for _, g in grid.groupby("symbol", sort=False)]
    f = pd.concat(parts, ignore_index=True)

    mkt = f.groupby("open_time").agg(mkt_ret_1m=("ret_1m", "mean"), mkt_ret_5m=("ret_5m", "mean")).sort_index()
    mkt["mkt_rv_15"] = np.sqrt(mkt["mkt_ret_1m"].pow(2).rolling(15, min_periods=15).sum())
    f = f.merge(mkt, left_on="open_time", right_index=True, how="left")
    f["idio_ret_5m"] = f["ret_5m"] - f["mkt_ret_5m"]

    hour = (f["open_time"] // 3_600_000) % 24 + ((f["open_time"] // MINUTE_MS) % 60) / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return f[["symbol", "open_time", "close", *FEATURE_COLUMNS]].sort_values(["open_time", "symbol"]).reset_index(drop=True)


def ready(f: pd.DataFrame) -> pd.DataFrame:
    """Rows with every feature available (after the 1-day warm-up)."""
    return f.dropna(subset=FEATURE_COLUMNS)


def add_target(f: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Training target: log realised volatility over the NEXT `horizon` minutes (t+1 .. t+horizon).
    Uses future data on purpose; never call it in live scoring."""
    f = f.sort_values(["symbol", "open_time"]).copy()
    r2 = f.groupby("symbol")["ret_1m"].transform(lambda s: s.pow(2))
    fwd = r2.groupby(f["symbol"]).transform(lambda s: s[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1))
    f[TARGET_COLUMN] = np.log(np.sqrt(fwd) + EPS)
    return f.sort_values(["open_time", "symbol"]).reset_index(drop=True)


class FeatureState:
    """Live version: feed closed candles one by one, get feature rows when a minute is complete
    for every active symbol. Uses compute_features() on a rolling buffer, so the numbers are
    identical to the offline computation (tested in tests/test_features.py)."""

    def __init__(self, buffer_min: int = BUFFER_MIN):
        self.buffer_min = buffer_min
        self.rows: dict[str, list[dict]] = {}
        self.last_emitted: int | None = None

    def add(self, kline: dict, compute: bool = True) -> list[dict]:
        """compute=False only buffers (used for warm-up candles: fast, nothing emitted)."""
        sym = kline["symbol"]
        buf = self.rows.setdefault(sym, [])
        if buf and kline["open_time"] <= buf[-1]["open_time"]:
            return []  # duplicate or late candle
        buf.append(kline)
        if len(buf) > self.buffer_min:
            del buf[: len(buf) - self.buffer_min]
        return self._emit(compute)

    def _emit(self, compute: bool = True) -> list[dict]:
        complete_until = min(buf[-1]["open_time"] for buf in self.rows.values())
        if self.last_emitted is not None and complete_until <= self.last_emitted:
            return []
        if not compute:
            self.last_emitted = complete_until
            return []
        start = complete_until if self.last_emitted is None else self.last_emitted + MINUTE_MS
        frame = pd.DataFrame([r for buf in self.rows.values() for r in buf])
        feats = compute_features(frame)
        out = feats[(feats["open_time"] >= start) & (feats["open_time"] <= complete_until)]
        self.last_emitted = complete_until
        return out.to_dict("records")
