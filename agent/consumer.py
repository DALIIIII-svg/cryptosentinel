"""Agent service: Kafka alerts.anomaly -> RiskAnalyst -> DynamoDB alerts + Kafka alerts.explained + notification.

Facts come from the DynamoDB hot tables (filled by the hot writer, live or replay), so the hot writer
must be running. Run:  python -m agent.consumer      (or: make agent)
"""
from __future__ import annotations

import signal
import threading

from agent.graph import RiskAnalyst
from agent.notify import send
from agent.sources import DynamoSource
from common.aws_clients import dynamodb_resource
from common.config import get_settings
from common.dynamo_io import to_dynamo
from common.kafka_io import decode, make_consumer, make_producer, produce_json
from common.log import get_logger
from common.topics import ALERTS_ANOMALY, ALERTS_EXPLAINED

log = get_logger("agent_consumer")


def main() -> None:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    s = get_settings()
    analyst = RiskAnalyst(DynamoSource())
    table = dynamodb_resource().Table(s.table_name("alerts"))
    consumer = make_consumer("agent", [ALERTS_ANOMALY.name], **{"auto.offset.reset": "latest"})
    producer = make_producer()
    done = 0
    log.info("agent ready (%s), waiting for %s", analyst.model, ALERTS_ANOMALY.name)
    try:
        while not stop.is_set():
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            alert = decode(msg.value())
            rec = analyst.explain(alert)
            table.put_item(Item=to_dynamo({k: v for k, v in rec.items() if v is not None}))
            produce_json(producer, ALERTS_EXPLAINED.name, rec["symbol"], rec)
            producer.poll(0)
            consumer.commit(asynchronous=False)
            send(rec)
            done += 1
            log.info("explained %s in %d ms: %s (%.0f%%), %d citation(s), guardrails %s",
                     rec["alert_id"], rec["latency_ms"], rec["cause"], rec["confidence"] * 100,
                     len(rec["citations"]), rec["guardrails"] or "none")
    finally:
        producer.flush(10)
        consumer.close()
        log.info("stopped after %d explained alerts", done)


if __name__ == "__main__":
    main()
