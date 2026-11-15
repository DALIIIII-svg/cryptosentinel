"""Are the news timestamps really UTC? Two independent checks.

1. Hour-of-day profile of publications (UTC). English crypto media publish most during
   European + US working hours, roughly 12:00-21:00 UTC; a profile shifted by 4-5 hours
   would suggest US Eastern time labelled as UTC.
2. Lag test: correlation between hourly news volume and hourly market activity
   (mean |1-min return| of the 10 coins) when news is shifted by -6..+6 hours.
   News reacts to moves within the same hour or the next, so the best lag should be 0 or +1.
   A best lag of -4/-5 (or +4/+5) means the news clock is shifted by that many hours.
3. Event study: news volume in 30-minute bins from 6 h before to 10 h after the biggest
   market-wide crashes, divided by the volume at the same clock hours on the 28 surrounding days
   (news is naturally quieter at night, so raw counts mislead). Coverage should rise above 1.0x
   within the first hour or two after the crash if the clock is right.
4. Headline vs price (the decisive test): every headline such as "Bitcoin falls below $53K" or
   "BTC tops $70,000" is compared with the minute the real BTC price first crossed that level.
   Typical delay 0..2 h = clock right; about +N h for all of them = clock N h late; negative = early.
5. Headlines around the 5 August 2024 crash, with their timestamps, for a human check.

Run:  python -m processing.check_news_time       (or: make newstime)
"""
from __future__ import annotations

import re
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


def _count(pub: np.ndarray, lo: int, hi: int) -> int:
    return int(np.searchsorted(pub, hi) - np.searchsorted(pub, lo))


def event_study(news: pd.DataFrame, event_ms: list[int], before_h: int = 6, after_h: int = 10,
                bin_min: int = 30, normalise: bool = False, ref_days: int = 14) -> pd.Series:
    """News per bin (index = minutes relative to the event), summed over events.
    normalise=True divides by the count at the same clock time on the +-ref_days surrounding days."""
    bins = range(-before_h * 60, after_h * 60, bin_min)
    pub = np.sort(news["published_at"].to_numpy())
    out = {}
    for b in bins:
        obs = exp = 0.0
        for t in event_ms:
            lo, hi = t + b * 60_000, t + (b + bin_min) * 60_000
            obs += _count(pub, lo, hi)
            if normalise:
                ref = [_count(pub, lo + d * 86_400_000, hi + d * 86_400_000)
                       for d in range(-ref_days, ref_days + 1) if d != 0]
                exp += float(np.mean(ref))
        out[b] = obs / exp if normalise and exp else obs
    return pd.Series(out)


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


_LEVEL = re.compile(
    r"\b(?:bitcoin|btc)\b[^$]{0,60}?\b(?P<verb>below|under|falls? to|drops? to|dips? to|crash(?:es)? to|tumbles? to|"
    r"plunges? to|sinks? to|slides? to|above|over|tops?|hits?|surpass(?:es)?|breaks?|reclaims?|rall(?:y|ies) to|"
    r"surges? to|jumps? to|climbs? to|soars? to)\s+\$(?P<num>\d+(?:[.,]\d+)*)\s*(?P<k>k\b)?",
    re.IGNORECASE)
_DOWN = re.compile(r"below|under|falls?|drops?|dips?|crash|tumbles?|plunges?|sinks?|slides?", re.IGNORECASE)


def parse_level(title: str) -> tuple[float, str] | None:
    """'Bitcoin crashes below $53K' -> (53000.0, 'down'); None when no clear BTC price level."""
    m = _LEVEL.search(title)
    if not m:
        return None
    num = float(m.group("num").replace(",", ""))
    if m.group("k"):
        num *= 1000
    if not 10_000 <= num <= 200_000:  # not a plausible BTC price in 2024
        return None
    return num, "down" if _DOWN.search(m.group("verb")) else "up"


def headline_delays(news: pd.DataFrame, btc: pd.DataFrame, lookback_h: int = 24) -> pd.Series:
    """Minutes between the first BTC crossing of the headline's level and the headline time
    (positive = headline after the crossing). Headlines whose level was not crossed are skipped."""
    btc = btc.sort_values("open_time")
    t, low, high = btc["open_time"].to_numpy(), btc["low"].to_numpy(), btc["high"].to_numpy()
    out = []
    for _, r in news.iterrows():
        parsed = parse_level(r["title"])
        if not parsed:
            continue
        level, direction = parsed
        h = int(r["published_at"])
        i0, i1 = np.searchsorted(t, h - lookback_h * 3_600_000), np.searchsorted(t, h + 6 * 3_600_000)
        if i1 - i0 < 60:
            continue
        lo_w, hi_w = low[i0:i1], high[i0:i1]
        if direction == "down":
            if lo_w[0] <= level:  # already below at the start: no crossing in the window
                continue
            hit = np.nonzero(lo_w <= level)[0]
        else:
            if hi_w[0] >= level:
                continue
            hit = np.nonzero(hi_w >= level)[0]
        if len(hit):
            out.append((h - int(t[i0 + hit[0]])) / 60_000)
    return pd.Series(out, dtype=float)


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
    if events:
        ratio = event_study(news, events, normalise=True)
        print(f"\n3. News around the {len(events)} biggest market-wide crashes, vs the same hours on normal days")
        for b, x in ratio.items():
            print(f"   {b / 60:+5.1f} h  {x:4.2f}x  {'#' * int(20 * x)}")
        first_hours = ratio.loc[(ratio.index >= 0) & (ratio.index < 120)].mean()
        before = ratio.loc[(ratio.index >= -360) & (ratio.index < -60)].mean()
        print(f"   average before (-6..-1 h): {before:.2f}x   first 2 h after: {first_hours:.2f}x")
    else:
        print("\n3. Event study skipped: run `make shocks` first.")

    btc = klines[klines["symbol"] == "BTCUSDT"]
    delays = headline_delays(news, btc)
    print(f"\n4. Headline vs price: {len(delays)} BTC headlines with a price level that was crossed")
    med = None
    if len(delays) >= 10:
        med = float(delays.median())
        q1, q3 = delays.quantile([0.25, 0.75])
        print(f"   delay headline - crossing: median {med:+.0f} min, middle half {q1:+.0f} .. {q3:+.0f} min")
        print(f"   share of headlines BEFORE the crossing (impossible if the clock is right): {(delays < 0).mean():.0%}")
        hist = pd.cut(delays, [-1e9, -120, -30, 0, 30, 60, 120, 240, 360, 1e9],
                      labels=["< -2 h", "-2 h..-30m", "-30..0 m", "0..30 m", "30..60 m", "1..2 h", "2..4 h", "4..6 h", "> 6 h"])
        for label, n in hist.value_counts(sort=False).items():
            print(f"   {label:>11s}  {n:4d}  {'#' * int(60 * n / len(delays))}")
    else:
        print("   too few headlines with a price level to decide")

    crash = int(pd.Timestamp("2024-08-05 01:11", tz="UTC").value // 1_000_000)
    near = news[(news["published_at"] >= crash - 2 * 3_600_000) & (news["published_at"] < crash + 10 * 3_600_000)]
    near = near[near["title"].str.contains(KEYWORDS, case=False, regex=True)].sort_values("published_at")
    print("\n5. Crash-related headlines around 2024-08-05 01:11 UTC (dataset time)")
    for _, r in near.head(40).iterrows():
        print(f"   {pd.to_datetime(r['published_at'], unit='ms', utc=True):%m-%d %H:%M}  {r['title'][:110]}")

    if med is None:
        print("\nVerdict: not enough evidence; send me this output.")
        return 2
    ok = -15 <= med <= 150 and (delays < -30).mean() < 0.2
    print("\nVerdict:", f"timestamps look like UTC (headlines come a median {med:.0f} min after the price move)."
          if ok else f"the clock looks shifted by about {med:+.0f} min: send me this output.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
