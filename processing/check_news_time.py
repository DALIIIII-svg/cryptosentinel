"""Are the news timestamps really UTC? Two independent checks.

1. Hour-of-day profile of publications (UTC). English crypto media publish most during
   European + US working hours, roughly 12:00-21:00 UTC; a profile shifted by 4-5 hours
   would suggest US Eastern time labelled as UTC.
2. Lag test: correlation between hourly news volume and hourly market activity
   (mean |1-min return| of the 10 coins) when news is shifted by -6..+6 hours.
   News reacts to moves within the same hour or the next, so the best lag should be 0 or +1.
   A best lag of -4/-5 (or +4/+5) means the news clock is shifted by that many hours.
3. Event study (the sharpest test): news volume in 30-minute bins from 6 h before to 10 h after
   the biggest market-wide crashes, added up. If the clock is right, coverage jumps in the first
   hour after the crash; a jump that starts N hours later means the clock is N hours late.
4. Headlines around the 5 August 2024 crash, with their timestamps, for a human check.

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


def event_study(news: pd.DataFrame, event_ms: list[int], before_h: int = 6, after_h: int = 10,
                bin_min: int = 30) -> pd.Series:
    """Total news count per bin (index = minutes relative to the event), summed over events."""
    bins = range(-before_h * 60, after_h * 60, bin_min)
    pub = news["published_at"].to_numpy()
    totals = {}
    for b in bins:
        n = 0
        for t in event_ms:
            lo, hi = t + b * 60_000, t + (b + bin_min) * 60_000
            n += int(((pub >= lo) & (pub < hi)).sum())
        totals[b] = n
    return pd.Series(totals)


def jump_start(counts: pd.Series, factor: float = 1.5) -> int | None:
    """First bin from which the next two bins are all > factor x the pre-event baseline (-6 h..-1 h)."""
    base = counts.loc[(counts.index >= -360) & (counts.index < -60)].mean()
    after = counts.loc[counts.index >= -60]
    idx = list(after.index)
    for i in range(len(idx) - 2):
        if all(after.loc[idx[j]] > factor * base for j in (i, i + 1, i + 2)):
            return idx[i]
    return None


def market_wide_events(top: int = 12) -> list[int]:
    try:
        ev = pd.read_csv("eval/shock_events.csv")
    except FileNotFoundError:
        return []
    ev = ev[ev["type"] == "market-wide"].head(top)
    return [int(pd.Timestamp(t).value // 1_000_000) for t in ev["time_utc"]]


KEYWORDS = r"crash|plung|tumbl|sell-?off|liquidat|nikkei|japan|carry trade|bloodbath|slump|dump|drop|below"


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

    events = market_wide_events()
    start = None
    if events:
        counts = event_study(news, events)
        base = counts.loc[(counts.index >= -360) & (counts.index < -60)].mean()
        print(f"\n3. News around the {len(events)} biggest market-wide crashes (30-min bins, baseline {base:.0f})")
        for b, n in counts.items():
            print(f"   {b / 60:+5.1f} h  {n:4d}  {'#' * int(40 * n / max(counts.max(), 1))}")
        start = jump_start(counts)
        print(f"   coverage jump starts at: {'none found' if start is None else f'{start / 60:+.1f} h'}")
    else:
        print("\n3. Event study skipped: run `make shocks` first.")

    crash = int(pd.Timestamp("2024-08-05 01:11", tz="UTC").value // 1_000_000)
    near = news[(news["published_at"] >= crash - 2 * 3_600_000) & (news["published_at"] < crash + 10 * 3_600_000)]
    near = near[near["title"].str.contains(KEYWORDS, case=False, regex=True)].sort_values("published_at")
    print("\n4. Crash-related headlines around 2024-08-05 01:11 UTC (dataset time)")
    for _, r in near.head(40).iterrows():
        print(f"   {pd.to_datetime(r['published_at'], unit='ms', utc=True):%m-%d %H:%M}  {r['title'][:110]}")

    ok = start is not None and -60 <= start <= 60 if events else best in (0, 1)
    print("\nVerdict:", "timestamps look like UTC." if ok else
          "the news clock may be shifted: send me sections 3 and 4 of this output.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
