"""Live features: Kafka prices.klines -> FeatureState -> Kafka features.1m (key = symbol).

Works the same for live candles and replayed history. A minute is published once every
active symbol has its candle for that minute.

Run:  python -m processing.feature_consumer        (or: make features-live)
"""
from __future__ import annotations

import math
import signal
import threading

from common.kafka_io import decode, make_consumer, make_producer, produce_json
from common.log import get_logger
from common.topics import FEATURES_1M, PRICES_KLINES
from processing.features import FeatureState

log = get_logger("feature_consumer")


def clean(row: dict) -> dict:
    """NaN/inf are not valid JSON: send them as null."""
    out = {}
    for k, v in row.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = None if isinstance(v, float) and not math.isfinite(v) else v
    return out


def main() -> None:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    # each run starts from the newest candles: a feature buffer is rebuilt in memory
    consumer = make_consumer("feature-consumer", [PRICES_KLINES.name], **{"auto.offset.reset": "latest"})
    producer = make_producer()
    state = FeatureState()
    sent = 0
    log.info("reading %s, writing %s (first full features after 24 h of candles; replay a longer window to warm up)",
             PRICES_KLINES.name, FEATURES_1M.name)
    try:
        while not stop.is_set():
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            k = decode(msg.value())
            replay = bool(k.get("replay"))
            for row in state.add(k, compute=not k.get("warmup")):
                row = clean(row)
                row["replay"] = replay
                produce_json(producer, FEATURES_1M.name, row["symbol"], row)
                sent += 1
            consumer.commit(asynchronous=True)
            if sent and sent % 1000 == 0:
                log.info("published %d feature rows", sent)
    finally:
        producer.flush(10)
        consumer.close()
        log.info("stopped after %d feature rows", sent)


if __name__ == "__main__":
    main()
