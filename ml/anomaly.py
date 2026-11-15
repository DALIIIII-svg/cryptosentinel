"""Anomaly detection building blocks: features, scores, alert thresholds, evaluation, injections.

Everything is expressed relative to the coin's own recent volatility (rv_1d), so a 2 % move means
more for BTC on a calm day than for PEPE on a wild one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from processing.features import DAY_MIN

EPS = 1e-9
ANOMALY_FEATURES = ["r1n", "r5n", "r15n", "idio5n", "rv5n", "range_n", "vol_z", "trades_z", "imbalance"]
COOLDOWN_MS = 15 * 60_000           # at most one alert per coin per 15 minutes
MATCH_MS = 10 * 60_000              # an alert within +-10 min of an event counts as a detection


def anomaly_frame(f: pd.DataFrame) -> pd.DataFrame:
    """Volatility-normalised features (returns in units of 'usual move over that horizon')."""
    sig = f["rv_1d"].to_numpy(float) / np.sqrt(DAY_MIN) + EPS      # usual 1-minute move
    out = pd.DataFrame(index=f.index)
    out["r1n"] = f["ret_1m"].to_numpy(float) / sig
    out["r5n"] = f["ret_5m"].to_numpy(float) / (sig * np.sqrt(5))
    out["r15n"] = f["ret_15m"].to_numpy(float) / (sig * np.sqrt(15))
    out["idio5n"] = f["idio_ret_5m"].to_numpy(float) / (sig * np.sqrt(5))
    out["rv5n"] = f["rv_5"].to_numpy(float) / (sig * np.sqrt(5))
    out["range_n"] = f["range_pct"].to_numpy(float) / sig
    for c in ("vol_z", "trades_z", "imbalance"):
        out[c] = f[c].to_numpy(float)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-50, 50)


def rule_score(a: pd.DataFrame) -> np.ndarray:
    """Baseline: size of the 5-minute move in standard deviations (the classic 'z > 4' rule)."""
    return np.abs(a["r5n"].to_numpy(float))


def alerts(times: np.ndarray, symbols: np.ndarray, scores: np.ndarray, threshold: float,
           cooldown_ms: int = COOLDOWN_MS) -> pd.DataFrame:
    """Alerts = rows above threshold, at most one per coin per cooldown (the first of a burst)."""
    idx = np.nonzero(scores > threshold)[0]
    if not len(idx):
        return pd.DataFrame(columns=["open_time", "symbol", "score"])
    cand = pd.DataFrame({"open_time": times[idx], "symbol": symbols[idx], "score": scores[idx]}).sort_values("open_time")
    keep, last = [], {}
    for t, s in zip(cand["open_time"].to_numpy(), cand["symbol"].to_numpy()):
        ok = s not in last or t - last[s] >= cooldown_ms
        keep.append(ok)
        if ok:
            last[s] = t
    return cand[np.array(keep)].reset_index(drop=True)


def threshold_for_budget(times: np.ndarray, symbols: np.ndarray, scores: np.ndarray,
                         alerts_per_day: float) -> float:
    """Threshold giving about `alerts_per_day` alerts (all coins together, after cooldown)."""
    days = max((times.max() - times.min()) / 86_400_000, 1e-9)
    target = alerts_per_day * days
    lo, hi = float(np.quantile(scores, 0.5)), float(scores.max())
    for _ in range(30):
        mid = (lo + hi) / 2
        n = len(alerts(times, symbols, scores, mid))
        lo, hi = (mid, hi) if n > target else (lo, mid)
    return hi


def detected(events: pd.DataFrame, alert_df: pd.DataFrame, match_ms: int = MATCH_MS) -> np.ndarray:
    """events: columns end_ms, leader, type. A market-wide event is detected by an alert on any coin,
    a coin-specific one only by an alert on its leader coin."""
    out = []
    for _, e in events.iterrows():
        near = alert_df[(alert_df["open_time"] - e["end_ms"]).abs() <= match_ms]
        if e.get("type") != "market-wide":
            near = near[near["symbol"] == e["leader"]]
        out.append(len(near) > 0)
    return np.array(out, dtype=bool)


# ---------------------------------------------------------------- synthetic injections
INJECTION_TYPES = ("price_jump", "pump", "volume_burst")


def inject(klines: pd.DataFrame, n_per_type: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant anomalies into 1-minute candles (a copy). Returns (candles, injected events).
      price_jump   : a sudden 5-minute move of 6-10 'usual 5-minute moves', staying there
      pump         : +3-6 % over 3 minutes with volume x8 and buying pressure, then back down
      volume_burst : volume and trades x12 for 3 minutes with almost no price change (wash-trading-like)
    Sizes are moderate on purpose: a detector should not catch them trivially."""
    rng = np.random.default_rng(seed)
    k = klines.sort_values(["symbol", "open_time"]).reset_index(drop=True).copy()
    events = []
    syms = k["symbol"].unique()
    t_min, t_max = k["open_time"].min() + 2 * 86_400_000, k["open_time"].max() - 86_400_000
    for kind in INJECTION_TYPES:
        for _ in range(n_per_type):
            sym = rng.choice(syms)
            t = int(rng.integers(t_min // 60_000, t_max // 60_000)) * 60_000
            rows = k.index[(k["symbol"] == sym) & (k["open_time"] >= t)][:30]
            if len(rows) < 30:
                continue
            i = rows[0]
            past = k.loc[(k["symbol"] == sym) & (k["open_time"] < t), "close"].tail(DAY_MIN).to_numpy(float)
            sig5 = float(np.std(np.diff(np.log(past))) * np.sqrt(5)) if len(past) > 100 else 0.003
            if kind == "price_jump":
                move = float(rng.uniform(6, 10)) * sig5 * rng.choice([-1, 1])
                factor = np.exp(np.r_[np.linspace(move / 5, move, 5), np.full(25, move)])
                k.loc[rows, ["open", "high", "low", "close"]] = k.loc[rows, ["open", "high", "low", "close"]].mul(factor, axis=0)
                k.loc[rows[:5], "volume"] *= 3
            elif kind == "pump":
                up = float(rng.uniform(0.03, 0.06))
                path = np.r_[np.linspace(up / 3, up, 3), np.linspace(up * 0.7, 0, 7), np.zeros(20)]
                k.loc[rows, ["open", "high", "low", "close"]] = k.loc[rows, ["open", "high", "low", "close"]].mul(np.exp(path), axis=0)
                k.loc[rows[:3], "volume"] *= 8
                k.loc[rows[:3], "n_trades"] = (k.loc[rows[:3], "n_trades"] * 8).astype("int64")
                k.loc[rows[:3], "taker_buy_base"] = k.loc[rows[:3], "volume"] * 0.9
            else:
                k.loc[rows[:3], "volume"] *= 12
                k.loc[rows[:3], "n_trades"] = (k.loc[rows[:3], "n_trades"] * 12).astype("int64")
                k.loc[rows[:3], "taker_buy_base"] = k.loc[rows[:3], "volume"] * 0.5
            k["high"] = k[["high", "open", "close"]].max(axis=1)
            k["low"] = k[["low", "open", "close"]].min(axis=1)
            events.append({"kind": kind, "leader": sym, "end_ms": int(k.loc[i, "open_time"]) + 3 * 60_000,
                           "type": "coin-specific"})
    return k, pd.DataFrame(events)


def combined_score(rule: np.ndarray, iforest: np.ndarray, scale: dict) -> np.ndarray:
    """Both detectors on one scale: each score divided by its 99.9th percentile on validation data,
    then the larger one. Catches big price moves (rule) AND odd volume/flow patterns (forest)."""
    return np.maximum(rule / scale["rule"], iforest / scale["iforest"])
