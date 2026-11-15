"""Replay a historical period into Kafka as if it were happening now (for demos and tests).

It reads processed candles and news for the window, orders every event by its real time
(a candle at its close time, a news item at its publication time) and publishes them to the
same topics as the live producers (prices.klines, news.raw), waiting between events so that
SPEED x real time is respected. News from the 60 minutes BEFORE the window are sent at the
start, so the agent has the context that existed at that moment.

Every replayed record carries "replay": true. The S3 sink ignores them (no duplicate history);
the hot writer, features and agent treat them like live data.

  python -m replay.replay --start "2024-08-05 00:00" --end "2024-08-05 04:00" --speed 10
  python -m replay.replay --start "2024-08-05 00:00" --end "2024-08-05 04:00" --speed 0     (as fast as possible)
  python -m replay.replay ... --rebase      (shift all times so the window starts now: looks live)
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import pandas as pd

from common.config import get_settings
from common.kafka_io import make_producer, produce_json
from common.log import get_logger
from common.parquet_io import load_klines, load_news
from common.schemas import now_ms
from common.topics import NEWS_RAW, PRICES_KLINES

log = get_logger("replay")
NEWS_LEAD_MS = 60 * 60 * 1000


def parse_utc(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").value // 1_000_000)


def months_between(start_ms: int, end_ms: int) -> list[str]:
    first = pd.to_datetime(start_ms - NEWS_LEAD_MS, unit="ms")
    last = pd.to_datetime(end_ms, unit="ms")
    return list(pd.period_range(first, last, freq="M").strftime("%Y-%m"))


def build_events(klines: pd.DataFrame, news: pd.DataFrame, start_ms: int, end_ms: int) -> list[tuple[int, str, str, dict]]:
    """[(event_time_ms, topic, key, record)] sorted by time. News before `start_ms` get time = start_ms."""
    events = []
    k = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] < end_ms)]
    for rec in k.to_dict("records"):
        rec = {key: (v.item() if hasattr(v, "item") else v) for key, v in rec.items()}
        rec.update(is_closed=True, replay=True)
        events.append((int(rec["close_time"]) + 1, PRICES_KLINES.name, rec["symbol"], rec))
    n = news[(news["published_at"] >= start_ms - NEWS_LEAD_MS) & (news["published_at"] < end_ms)]
    for rec in n.to_dict("records"):
        rec = {key: (list(v) if key == "coins" and v is not None else (v.item() if hasattr(v, "item") else v))
               for key, v in rec.items()}
        rec["replay"] = True
        events.append((max(int(rec["published_at"]), start_ms), NEWS_RAW.name, rec["news_id"], rec))
    events.sort(key=lambda e: (e[0], 0 if e[1] == NEWS_RAW.name else 1))
    return events


def rebase(events: list[tuple[int, str, str, dict]], start_ms: int, now: int) -> list[tuple[int, str, str, dict]]:
    """Shift every timestamp by (now - start) so the replay looks live. Original times are kept."""
    delta = now - start_ms
    out = []
    for t, topic, key, rec in events:
        rec = dict(rec)
        for f in ("open_time", "close_time", "published_at"):
            if f in rec and rec[f] is not None:
                rec[f"original_{f}"] = rec[f]
                rec[f] = int(rec[f]) + delta
        rec["rebased_ms"] = delta
        out.append((t + delta, topic, key, rec))
    return out


def run(events, speed: float, stop: threading.Event, producer=None, sleep=time.sleep) -> int:
    producer = producer or make_producer()
    if not events:
        return 0
    t0_event, t0_wall = events[0][0], time.monotonic()
    sent = 0
    for t, topic, key, rec in events:
        if stop.is_set():
            break
        if speed > 0:
            wait = (t - t0_event) / 1000 / speed - (time.monotonic() - t0_wall)
            if wait > 0:
                sleep(wait)
        produce_json(producer, topic, key, rec)
        sent += 1
        if sent % 500 == 0:
            producer.flush(5)
            log.info("sent %d / %d events (now at %s UTC)", sent, len(events),
                     pd.to_datetime(t, unit="ms").strftime("%Y-%m-%d %H:%M"))
    producer.flush(10)
    return sent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help='UTC, e.g. "2024-08-05 00:00"')
    ap.add_argument("--end", required=True, help='UTC, e.g. "2024-08-05 04:00"')
    ap.add_argument("--speed", type=float, default=10.0, help="x real time; 0 = no waiting")
    ap.add_argument("--rebase", action="store_true", help="shift times so the window starts now")
    ap.add_argument("--symbols", help="comma list, default = all configured symbols")
    args = ap.parse_args()

    s = get_settings()
    start_ms, end_ms = parse_utc(args.start), parse_utc(args.end)
    if end_ms <= start_ms:
        log.error("--end must be after --start")
        return 1
    symbols = [x.strip().upper() for x in args.symbols.split(",")] if args.symbols else s.symbol_list
    months = months_between(start_ms, end_ms)
    klines, news = load_klines(symbols, months), load_news(months)
    events = build_events(klines, news, start_ms, end_ms)
    if not events:
        log.error("nothing to replay in that window: did you run `make history` and `make parquet`?")
        return 1
    if args.rebase:
        events = rebase(events, start_ms, now_ms())
    n_k = sum(1 for e in events if e[1] == PRICES_KLINES.name)
    minutes = (end_ms - start_ms) / 60_000
    pace = f"{args.speed:g}x real time (about {minutes / args.speed:.1f} min)" if args.speed else "full speed"
    log.info("replaying %d candles and %d news items (%.0f min of history) at %s",
             n_k, len(events) - n_k, minutes, pace)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    sent = run(events, args.speed, stop)
    log.info("done: %d events sent", sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
