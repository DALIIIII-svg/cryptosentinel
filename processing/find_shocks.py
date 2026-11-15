"""Find the biggest price shocks in the historical period, with the news just before each one.

Used for three things:
  1. choosing good days for the replay demo,
  2. the candidate list you label by hand for the evaluation (eval/labelled_events.csv),
  3. a first sanity check that prices and news line up in time.

A shock = |5-minute log return| far above that coin's usual 5-minute moves (robust z-score),
at most one shock per coin per 6 hours. For each shock we also report the market move
(mean of the 10 coins) and how many news items mention the coin in the 60 minutes before.

Run:  python -m processing.find_shocks --top 40      (writes eval/shock_candidates.csv)
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
SEPARATION_MS = 6 * 3600 * 1000
NEWS_LEAD_MS = 60 * 60 * 1000


def five_min_returns(klines: pd.DataFrame) -> pd.DataFrame:
    """Wide table: index = open_time, columns = symbols, values = 5-min log return ending at that candle."""
    close = klines.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
    return np.log(close / close.shift(WINDOW_MIN))


def detect(klines: pd.DataFrame, top: int, min_z: float = 8.0) -> pd.DataFrame:
    r5 = five_min_returns(klines)
    market = r5.mean(axis=1)
    rows = []
    for sym in r5.columns:
        r = r5[sym].dropna()
        mad = (r - r.median()).abs().median() * 1.4826  # robust standard deviation
        if not mad:
            continue
        z = (r - r.median()) / mad
        cand = z[z.abs() >= min_z].abs().sort_values(ascending=False)
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
                "z": round(float(z.at[t]), 1),
                "market_ret_5m_pct": round(float(np.expm1(market.at[t])) * 100, 3),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("z", key=lambda s: s.abs(), ascending=False).head(top).reset_index(drop=True)


def attach_news(shocks: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if shocks.empty:
        return shocks
    news = news[news["time_precision"].fillna("second") == "second"]
    coin_news, market_news, headline = [], [], []
    for _, s in shocks.iterrows():
        start = s["end_ms"] - WINDOW_MIN * 60_000 - NEWS_LEAD_MS
        window = news[(news["published_at"] >= start) & (news["published_at"] <= s["end_ms"])]
        mentions = window[window["coins"].apply(lambda c: s["symbol"] in list(c) if c is not None else False)]
        coin_news.append(len(mentions))
        market_news.append(len(window))
        headline.append(mentions.sort_values("published_at").iloc[-1]["title"] if len(mentions) else "")
    return shocks.assign(coin_news_60m=coin_news, all_news_60m=market_news, latest_coin_headline=headline)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-z", type=float, default=8.0)
    ap.add_argument("--out", default="eval/shock_candidates.csv")
    args = ap.parse_args()
    s = get_settings()
    log.info("loading candles for %d symbols, %s..%s", len(s.symbol_list), s.history_start, s.history_end)
    klines = load_klines(s.symbol_list, s.history_months)
    if klines.empty:
        log.error("no processed candles: run `make history` then `make parquet` first")
        return 1
    shocks = attach_news(detect(klines, args.top, args.min_z), load_news(s.history_months))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shocks.drop(columns=["end_ms"]).to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(shocks.drop(columns=["end_ms"]).to_string(index=False))
    log.info("wrote %s (%d shocks)", args.out, len(shocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
