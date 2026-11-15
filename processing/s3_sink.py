"""S3 sink: Kafka -> raw zone of the bucket, as gzip JSON Lines.

    raw/{topic}/dt=YYYY-MM-DD/hour=HH/part-{flush time}-{random}.jsonl.gz

Flushes every SINK_FLUSH_SECONDS (60) or SINK_MAX_RECORDS (5000), whichever comes first.
Kafka offsets are committed only AFTER the files are written (at-least-once: a crash
can duplicate a few records, never lose them; Glue removes duplicates later).

Run:  python -m processing.s3_sink
"""
from __future__ import annotations

import signal
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from common.config import get_settings
from common.kafka_io import decode, make_consumer
from common.log import get_logger
from common.s3_io import put_jsonl_gz, raw_key
from common.topics import NEWS_RAW, PRICES_KLINES

log = get_logger("s3_sink")

SINK_TOPICS = [PRICES_KLINES.name, NEWS_RAW.name]


class Batcher:
    """Buffers records per topic and decides when to flush. Pure logic, easy to test."""

    def __init__(self, max_records: int, max_age_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_records = max_records
        self.max_age_s = max_age_s
        self.clock = clock
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.started = clock()

    def add(self, topic: str, record: dict) -> None:
        self.buffers[topic].append(record)

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.buffers.values())

    def due(self) -> bool:
        return self.size >= self.max_records or (self.clock() - self.started) >= self.max_age_s

    def drain(self) -> dict[str, list[dict]]:
        out = {t: recs for t, recs in self.buffers.items() if recs}
        self.buffers = defaultdict(list)
        self.started = self.clock()
        return out


def write_batches(batches: dict[str, list[dict]], when: datetime | None = None) -> dict[str, int]:
    when = when or datetime.now(timezone.utc)
    counts = {}
    for topic, records in batches.items():
        key = raw_key(topic, when, f"{when:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}")
        counts[topic] = put_jsonl_gz(key, records)
    return counts


def main() -> None:
    s = get_settings()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    consumer = make_consumer("s3-sink", SINK_TOPICS)
    batcher = Batcher(s.sink_max_records, s.sink_flush_seconds)
    log.info("sinking %s to s3://%s/raw/ every %d s", SINK_TOPICS, s.s3_bucket, s.sink_flush_seconds)

    def flush() -> None:
        batches = batcher.drain()
        if batches:
            counts = write_batches(batches)
            consumer.commit(asynchronous=False)
            log.info("wrote %s", ", ".join(f"{t}: {n}" for t, n in counts.items()))

    try:
        while not stop.is_set():
            msg = consumer.poll(1.0)
            if msg is not None and not msg.error():
                record = decode(msg.value())
                if not record.get("replay"):  # replayed history is already stored
                    batcher.add(msg.topic(), record)
            elif msg is not None and msg.error():
                log.warning("kafka error: %s", msg.error())
            if batcher.due():
                flush()
        flush()
    finally:
        consumer.close()
        log.info("stopped")


if __name__ == "__main__":
    main()
