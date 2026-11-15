"""Find the biggest price shocks in the historical period and group them into EVENTS.

A shock = a 5-minute log return far outside that coin's usual 5-minute moves (robust z-score),
measured two ways:
  - raw:           the coin's own return            -> catches market-wide crashes
  - idiosyncratic: coin return minus market return  -> catches coin-specific moves
Shocks of different coins within 10 minutes are grouped into ONE event, labelled by how much the
OTHER coins moved compared with the leader (market average excluding the leader):
  - market-wide   : others moved at least 35 % as much as the leader, in the same direction
  - coin-specific : otherwise
For each event: the leader coin, every coin hit, the market move, and the news of the hour before.

Outputs (used for the demo choice and the hand-labelled evaluation):
  eval/shock_events.csv      one row per event (top N market-wide + top N coin-specific)
  eval/shock_candidates.csv  one row per coin shock (detail)

Run:  python -m processing.find_shocks --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_settings
from common.log import get_logger
from common.parquet_io import load_klines, load_news

log = get_logger("find_shocks")
WINDOW_MIN = 5
SEPARATION_MS = 6 * 3600 * 1000      # at most one shock per coin per 6 h
CLUSTER_MS = 10 * 60 * 1000          # shocks of different coins within 10 min = same event
NEWS_LEAD_MS = 60 * 60 * 1000
WIDE_RATIO = 0.35                    # others moved >= 35 % of the leader's move -> market-wide


def five_min_returns(klines: pd.DataFrame) -> pd.DataFrame:
    """Wide table: index = open_time, columns = symbols, values = 5-min log return ending at that candle."""
    close = klines.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
    return np.log(close / close.shift(WINDOW_MIN))


def robust_z(r: pd.Series) -> pd.Series:
    r = r.dropna()
    mad = (r - r.median()).abs().median() * 1.4826
    return (r - r.median()) / mad if mad else r * 0


def detect(klines: pd.DataFrame, top: int, min_z: float = 8.0, min_z_idio: float = 10.0) -> pd.DataFrame:
    """One row per coin shock, strongest first (|z| of raw or idiosyncratic return)."""
    r5 = five_min_returns(klines)
    market = r5.mean(axis=1)
    n_active = r5.notna().sum(axis=1)
    rows = []
    for sym in r5.columns:
        z_raw = robust_z(r5[sym])
        z_idio = robust_z(r5[sym] - market)
        score = pd.concat([z_raw.abs(), z_idio.abs() * (min_z / min_z_idio)], axis=1).max(axis=1)
        cand = score[score >= min_z].sort_values(ascending=False)
        kept: list[int] = []
        for t in cand.index:
            if all(abs(t - k) > SEPARATION_MS for k in kept):
                kept.append(t)
        for t in kept:
            rows.append({
                "symbol": sym,
                "time_utc": pd.to_datetime(t + 60_000, unit="ms", utc=True),  # end of the 5-min window
                "end_ms": int(t + 60_000),
                "ret_5m_pct": round(float(np.expm1(r5.at[t, sym])) * 100, 3),
                "z": round(float(z_raw.get(t, 0.0)), 1),
                "z_idio": round(float(z_idio.get(t, 0.0)), 1),
                "market_ret_5m_pct": round(float(np.expm1(market.at[t])) * 100, 3),
                # market move WITHOUT this coin (the plain mean includes the coin itself)
                "others_ret_5m_pct": round(float(np.expm1(
                    (market.at[t] * n_active.at[t] - r5.at[t, sym]) / max(int(n_active.at[t]) - 1, 1))) * 100, 3),
                "score": float(score.at[t]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).head(top).reset_index(drop=True)


def group_events(shocks: pd.DataFrame) -> pd.DataFrame:
    """Merge shocks of different coins within CLUSTER_MS into events and label each event."""
    if shocks.empty:
        return shocks
    s = shocks.sort_values("end_ms").reset_index(drop=True)
    event_ids, current, start = [], 0, None
    for t in s["end_ms"]:
        if start is None or t - start > CLUSTER_MS:
            current += 1
            start = t
        event_ids.append(current)
    s["event"] = event_ids
    events = []
    for _, g in s.groupby("event"):
        lead = g.loc[g["ret_5m_pct"].abs().idxmax()]
        market = float(lead["market_ret_5m_pct"])
        others = float(lead["others_ret_5m_pct"])
        ratio = others / lead["ret_5m_pct"] if lead["ret_5m_pct"] else 0.0  # same sign -> positive
        wide = ratio >= WIDE_RATIO
        events.append({
            "time_utc": lead["time_utc"],
            "end_ms": int(lead["end_ms"]),
            "type": "market-wide" if wide else "coin-specific",
            "leader": lead["symbol"],
            "leader_ret_5m_pct": lead["ret_5m_pct"],
            "market_ret_5m_pct": market,
            "others_ret_5m_pct": others,
            "n_coins": len(g),
            "coins": " ".join(f"{r.symbol.replace('USDT', '')}:{r.ret_5m_pct:+.1f}%"
                              for r in g.sort_values("ret_5m_pct", key=abs, ascending=False).itertuples()),
            "score": float(g["score"].max()),
        })
    return pd.DataFrame(events).sort_values("score", ascending=False).reset_index(drop=True)


def attach_news(frame: pd.DataFrame, news: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Count news in the hour before each shock/event (only items with a real time of day)."""
    if frame.empty:
        return frame
    news = news[news["time_precision"].fillna("second") == "second"]
    coin_news, all_news, headline = [], [], []
    for _, s in frame.iterrows():
        start = s["end_ms"] - WINDOW_MIN * 60_000 - NEWS_LEAD_MS
        window = news[(news["published_at"] >= start) & (news["published_at"] <= s["end_ms"])]
        sym = s[symbol_col]
        mentions = window[window["coins"].apply(lambda c: sym in list(c) if c is not None else False)]
        coin_news.append(len(mentions))
        all_news.append(len(window))
        headline.append(mentions.sort_values("published_at").iloc[-1]["title"] if len(mentions) else "")
    return frame.assign(coin_news_60m=coin_news, all_news_60m=all_news, latest_coin_headline=headline)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="events of EACH type to keep")
    ap.add_argument("--min-z", type=float, default=8.0)
    args = ap.parse_args()
    s = get_settings()
    log.info("loading candles for %d symbols, %s..%s", len(s.symbol_list), s.history_start, s.history_end)
    klines = load_klines(s.symbol_list, s.history_months)
    if klines.empty:
        log.error("no processed candles: run `make history` then `make parquet` first")
        return 1
    news = load_news(s.history_months)
    shocks = detect(klines, top=100_000, min_z=args.min_z)
    events = group_events(shocks)
    events = pd.concat([events[events["type"] == "market-wide"].head(args.top),
                        events[events["type"] == "coin-specific"].head(args.top)], ignore_index=True)
    events = attach_news(events, news, symbol_col="leader")

    Path("eval").mkdir(exist_ok=True)
    attach_news(shocks.head(200), news).drop(columns=["end_ms", "score"]).to_csv("eval/shock_candidates.csv", index=False)
    events.insert(0, "event_id", [f"E{i + 1:03d}" for i in range(len(events))])
    events.drop(columns=["end_ms", "score"]).to_csv("eval/shock_events.csv", index=False)

    cols = ["event_id", "time_utc", "type", "leader", "leader_ret_5m_pct", "others_ret_5m_pct",
            "n_coins", "coin_news_60m", "all_news_60m"]
    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        for kind in ("market-wide", "coin-specific"):
            part = events[events["type"] == kind]
            print(f"\n=== {kind} events ({len(part)}) ===")
            print(part[cols].to_string(index=False))
    log.info("wrote eval/shock_events.csv (%d events) and eval/shock_candidates.csv", len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
