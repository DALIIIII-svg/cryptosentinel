"""Hot writer: Kafka -> DynamoDB hot tables (what the agent and the dashboard read).

  prices.klines -> <prefix>-prices-hot  (one item per symbol + minute, expires after 48 h)
  news.raw      -> <prefix>-news        (one item per tagged coin, 'MARKET' if none; 30 days)

Writes are idempotent (same keys overwrite), so replays and duplicates are harmless.

Run:  python -m processing.hot_writer
"""
from __future__ import annotations

import signal
import threading
from typing import Any

from common.aws_clients import dynamodb_resource
from common.config import get_settings
from common.dynamo_io import NEWS_TTL_SECONDS, PRICES_HOT_TTL_SECONDS, to_dynamo
from common.kafka_io import decode, make_consumer
from common.log import get_logger
from common.schemas import now_ms
from common.topics import NEWS_RAW, PRICES_KLINES

log = get_logger("hot_writer")

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "n_trades",
                "taker_buy_base", "taker_buy_quote", "close_time")


def price_item(k: dict[str, Any]) -> dict[str, Any]:
    item = {"symbol": k["symbol"], "open_time": k["open_time"]}
    if k.get("replay"):
        item["replay"] = True
    item.update({f: k[f] for f in PRICE_FIELDS if f in k})
    # TTL counts from when we store it, so replayed history is not deleted at once
    item["expires_at"] = max(k["close_time"] // 1000, now_ms() // 1000) + PRICES_HOT_TTL_SECONDS
    return to_dynamo(item)


def news_items(n: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "news_id": n["news_id"],
        "published_at": n["published_at"],
        "title": n["title"],
        "summary": n.get("summary", ""),
        "url": n["url"],
        "source": n["source"],
        "coins": list(n.get("coins") or []),
        "time_precision": n.get("time_precision") or "second",
        "expires_at": max(n["published_at"] // 1000, now_ms() // 1000) + NEWS_TTL_SECONDS,
    }
    if n.get("votes"):
        base["votes"] = n["votes"]
    targets = n.get("coins") or ["MARKET"]
    return [to_dynamo({**base, "coin": c, "sk": f"{n['published_at']:013d}#{n['news_id']}"}) for c in targets]


def main() -> None:
    s = get_settings()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    db = dynamodb_resource()
    prices = db.Table(s.table_name("prices-hot"))
    news = db.Table(s.table_name("news"))
    consumer = make_consumer("hot-writer", [PRICES_KLINES.name, NEWS_RAW.name])
    log.info("writing to %s and %s", prices.name, news.name)
    written = 0

    try:
        while not stop.is_set():
            msgs = consumer.consume(num_messages=500, timeout=1.0)
            price_batch, news_batch = [], []
            for m in msgs:
                if m.error():
                    log.warning("kafka error: %s", m.error())
                    continue
                value = decode(m.value())
                if m.topic() == PRICES_KLINES.name:
                    price_batch.append(price_item(value))
                else:
                    news_batch.extend(news_items(value))
            if price_batch:
                with prices.batch_writer(overwrite_by_pkeys=["symbol", "open_time"]) as bw:
                    for it in price_batch:
                        bw.put_item(Item=it)
            if news_batch:
                with news.batch_writer(overwrite_by_pkeys=["coin", "sk"]) as bw:
                    for it in news_batch:
                        bw.put_item(Item=it)
            if msgs:
                consumer.commit(asynchronous=False)
                written += len(price_batch) + len(news_batch)
                if price_batch or news_batch:
                    log.debug("wrote %d prices, %d news items", len(price_batch), len(news_batch))
    finally:
        consumer.close()
        log.info("stopped after %d items", written)


if __name__ == "__main__":
    main()
