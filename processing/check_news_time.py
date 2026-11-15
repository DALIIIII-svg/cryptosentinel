"""Are the news timestamps really UTC? Two independent checks.

1. Hour-of-day profile of publications (UTC). English crypto media publish most during
   European + US working hours, roughly 12:00-21:00 UTC; a profile shifted by 4-5 hours
   would suggest US Eastern time labelled as UTC.
2. Lag test: correlation between hourly news volume and hourly market activity
   (mean |1-min return| of the 10 coins) when news is shifted by -6..+6 hours.
   News reacts to moves within the same hour or the next, so the best lag should be 0 or +1.
   A best lag of -4/-5 (or +4/+5) means the news clock is shifted by that many hours.

Run:  python -m processing.check_news_time       (or: make newstime)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common.config import get_settings
from common.parquet_io import load_klines, load_news


def hourly_market_activity(klines: pd.DataFrame) -> pd.Series:
    k = klines.sort_values(["symbol", "open_time"])
    k = k.assign(absret=np.log(k["close"] / k.groupby("symbol")["close"].shift(1)).abs())
    hour = (k["open_time"] // 3_600_000) * 3_600_000
    return k.groupby(hour)["absret"].mean().rename("activity")


def hourly_news(news: pd.DataFrame) -> pd.Series:
    hour = (news["published_at"] // 3_600_000) * 3_600_000
    return news.groupby(hour).size().rename("news")


def lag_correlations(activity: pd.Series, news_count: pd.Series, max_lag: int = 6) -> dict[int, float]:
    """corr(activity at hour h, news at hour h + lag). Positive lag = news after the move."""
    df = pd.concat([activity.rename("activity"), news_count.rename("news")], axis=1).fillna({"news": 0}).dropna()
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, 3_600_000)).fillna({"news": 0})
    # Remove the weekly rhythm (both series are busier in US hours and on weekdays),
    # so the correlation comes from event spikes, not from the daily cycle.
    ts = pd.to_datetime(df.index, unit="ms", utc=True)
    how = ts.day_of_week * 24 + ts.hour
    resid = df - df.groupby(how).transform("mean").values
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = resid["news"].shift(-lag)
        out[lag] = float(resid["activity"].corr(shifted))
    return out


def main() -> int:
    s = get_settings()
    news = load_news(s.history_months)
    news = news[news["time_precision"].fillna("second") == "second"]
    if news.empty:
        print("no news with time of day: run `make history` and `make parquet` first")
        return 1
    klines = load_klines(s.symbol_list, s.history_months)

    hours = pd.to_datetime(news["published_at"], unit="ms", utc=True).dt.hour
    profile = hours.value_counts(normalize=True).sort_index()
    print("1. Publications by hour of day (UTC)")
    for h, share in profile.items():
        print(f"   {h:02d}:00  {share * 100:5.1f}%  {'#' * int(share * 300)}")
    peak = int(profile.idxmax())
    busy = profile.loc[12:21].sum()
    print(f"   peak hour {peak:02d}:00 UTC, share published 12:00-21:59 UTC: {busy:.0%}")

    corr = lag_correlations(hourly_market_activity(klines), hourly_news(news))
    best = max(corr, key=corr.get)
    print("\n2. Correlation of market activity with news shifted by N hours (positive = news after the move)")
    for lag, c in corr.items():
        mark = "  <- best" if lag == best else ""
        print(f"   {lag:+d} h  {c:+.3f}{mark}")

    ok = best in (0, 1)
    print("\nVerdict:", "timestamps look like UTC (news follows market activity within the hour)."
          if ok else f"best lag is {best:+d} h: the news clock may be shifted. Tell me this number.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
