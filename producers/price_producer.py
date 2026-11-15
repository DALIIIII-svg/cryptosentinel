"""Binance price producer: closed 1-minute candles for every symbol -> Kafka `prices.klines`.

- One combined WebSocket on the market-data-only host (works from AWS US regions too).
- Only CLOSED candles (k.x == true) are published, keyed by symbol.
- Reconnects with exponential backoff, and on purpose every 23 h (Binance drops at 24 h).
- After every (re)connection, missed candles are fetched from the REST API (gap fill),
  so a Wi-Fi cut or a laptop sleep leaves no hole.

Run:  python -m producers.price_producer
"""
from __future__ import annotations

import asyncio
import json
import signal
import time

import requests
import websockets

from common.config import get_settings
from common.kafka_io import make_producer, produce_json
from common.log import get_logger
from common.schemas import ONE_MINUTE_MS, kline_from_rest, kline_from_ws, missing_open_times, now_ms
from common.topics import PRICES_KLINES

log = get_logger("price_producer")

PLANNED_RECONNECT_S = 23 * 3600
REST_LIMIT = 1000  # Binance max rows per klines request
STARTUP_BACKFILL_MIN = 120


class PriceProducer:
    def __init__(self) -> None:
        self.s = get_settings()
        self.symbols = self.s.symbol_list
        self.producer = make_producer()
        # symbol -> open_time of the last published closed candle.
        # At start-up we pretend the last candle was STARTUP_BACKFILL_MIN minutes ago, so the first
        # gap fill also covers a restart (duplicates are harmless: tables and Glue dedupe on keys).
        start = (now_ms() // ONE_MINUTE_MS) * ONE_MINUTE_MS - (STARTUP_BACKFILL_MIN + 1) * ONE_MINUTE_MS
        self.last_closed: dict[str, int] = {sym: start for sym in self.symbols}
        self.published = 0
        self.stop = asyncio.Event()

    # ------------------------------------------------------------ publishing
    def publish(self, kline: dict) -> None:
        sym = kline["symbol"]
        if kline["open_time"] <= self.last_closed.get(sym, -1):
            return  # duplicate (gap fill and stream can overlap)
        produce_json(self.producer, PRICES_KLINES.name, sym, kline)
        self.last_closed[sym] = kline["open_time"]
        self.published += 1

    # ------------------------------------------------------------ gap fill
    def _fetch_rest(self, symbol: str, start_ms: int) -> list[list]:
        url = f"{self.s.binance_rest_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": REST_LIMIT}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def gap_fill(self) -> None:
        now = now_ms()
        for sym in self.symbols:
            last = self.last_closed[sym]
            missing = missing_open_times(last, now)
            if not missing:
                continue
            filled = 0
            start = missing[0]
            try:
                while start <= missing[-1]:
                    rows = self._fetch_rest(sym, start)
                    if not rows:
                        break
                    for row in rows:
                        k = kline_from_rest(sym, row)
                        if k["close_time"] < now:
                            self.publish(k)
                            filled += 1
                    start = int(rows[-1][0]) + ONE_MINUTE_MS
                    if len(rows) < REST_LIMIT:
                        break
            except requests.RequestException as exc:
                log.warning("gap fill failed for %s: %s", sym, exc)
            if filled:
                log.info("gap fill %s: %d candle(s)", sym, filled)
        self.producer.flush(10)

    # ------------------------------------------------------------ stream
    def stream_url(self) -> str:
        streams = "/".join(f"{s.lower()}@kline_1m" for s in self.symbols)
        return f"{self.s.binance_ws_url}/stream?streams={streams}"

    async def run(self) -> None:
        backoff = 1
        while not self.stop.is_set():
            try:
                log.info("connecting to %s (%d symbols)", self.s.binance_ws_url, len(self.symbols))
                async with websockets.connect(
                    self.stream_url(), ping_interval=20, ping_timeout=30, open_timeout=20, max_queue=2048
                ) as ws:
                    backoff = 1
                    await asyncio.to_thread(self.gap_fill)
                    connected_at = time.monotonic()
                    last_report = connected_at
                    log.info("connected")
                    while not self.stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            log.warning("no message for 30 s, reconnecting")
                            break
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        if data.get("e") == "kline" and data["k"].get("x"):
                            self.publish(kline_from_ws(data))
                        self.producer.poll(0)
                        if time.monotonic() - last_report > 300:
                            log.info("published %d closed candles so far", self.published)
                            last_report = time.monotonic()
                        if time.monotonic() - connected_at > PLANNED_RECONNECT_S:
                            log.info("planned reconnect (23 h)")
                            break
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as exc:
                log.warning("connection problem: %s: %s; retry in %d s", type(exc).__name__, exc, backoff)
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60)
        self.producer.flush(10)
        log.info("stopped after %d candles", self.published)


def main() -> None:
    pp = PriceProducer()
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, pp.stop.set)
        except NotImplementedError:  # Windows without WSL
            pass
    loop.run_until_complete(pp.run())


if __name__ == "__main__":
    main()
