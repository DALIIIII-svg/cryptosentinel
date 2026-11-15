"""Thin wrappers around confluent-kafka so every component connects the same way."""
from __future__ import annotations

import json
from typing import Any, Iterable

from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from common.config import get_settings
from common.topics import ALL_TOPICS, RETENTION_MS_24H, Topic


def _base_conf() -> dict[str, Any]:
    return {"bootstrap.servers": get_settings().kafka_bootstrap}


def make_producer(**overrides: Any) -> Producer:
    conf = {
        **_base_conf(),
        "acks": "all",
        "enable.idempotence": True,
        "linger.ms": 50,
        "compression.type": "lz4",
    }
    conf.update(overrides)
    return Producer(conf)


def make_consumer(group_id: str, topics: Iterable[str], **overrides: Any) -> Consumer:
    conf = {
        **_base_conf(),
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        # Components that write elsewhere (S3, DynamoDB) commit manually after a successful write.
        "enable.auto.commit": False,
    }
    conf.update(overrides)
    consumer = Consumer(conf)
    consumer.subscribe(list(topics))
    return consumer


def encode(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")


def decode(raw: bytes | None) -> Any:
    return None if raw is None else json.loads(raw.decode("utf-8"))


def produce_json(producer: Producer, topic: str, key: str | None, value: Any) -> None:
    producer.produce(topic, key=key.encode("utf-8") if key else None, value=encode(value))
    producer.poll(0)  # serve delivery callbacks


def ensure_topics(topics: Iterable[Topic] = ALL_TOPICS, timeout: float = 30.0) -> list[str]:
    """Create missing topics. Returns the names that were created."""
    admin = AdminClient(_base_conf())
    existing = set(admin.list_topics(timeout=timeout).topics)
    to_create = [
        NewTopic(
            t.name,
            num_partitions=t.partitions,
            replication_factor=1,
            config={"retention.ms": str(RETENTION_MS_24H)},
        )
        for t in topics
        if t.name not in existing
    ]
    if not to_create:
        return []
    futures = admin.create_topics(to_create)
    created = []
    for name, fut in futures.items():
        try:
            fut.result(timeout=timeout)
            created.append(name)
        except KafkaException as exc:  # already created by someone else
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise
    return created
