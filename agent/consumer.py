"""Agent service: Kafka alerts.anomaly -> RiskAnalyst -> DynamoDB alerts + Kafka alerts.explained + notification.

Alerts arriving within GROUP_WAIT_S of each other are grouped: the coins of one market-wide move are
explained ONCE (by the coin that moved most) and the others reuse that explanation. During a crash
that is 1 LLM call instead of 10 (free-tier rate limits, faster alerts, one event = one story).

Facts come from the DynamoDB hot tables (filled by the hot writer, live or replay), so the hot writer
must be running. Run:  python -m agent.consumer      (or: make agent)
"""
from __future__ import annotations

import signal
import threading

import time

from agent.graph import RiskAnalyst, follower_record, group_alerts
from agent.notify import send
from agent.sources import DynamoSource
from common.aws_clients import dynamodb_resource
from common.config import get_settings
from common.dynamo_io import to_dynamo
from common.kafka_io import decode, make_consumer, make_producer, produce_json
from common.log import get_logger
from common.topics import ALERTS_ANOMALY, ALERTS_EXPLAINED

log = get_logger("agent_consumer")
GROUP_WAIT_S = 2.0


def main() -> None:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    s = get_settings()
    analyst = RiskAnalyst(DynamoSource())
    table = dynamodb_resource().Table(s.table_name("alerts"))
    consumer = make_consumer("agent", [ALERTS_ANOMALY.name], **{"auto.offset.reset": "latest"})
    producer = make_producer()
    done = llm_calls = 0
    pending: list[dict] = []
    first_at = 0.0
    log.info("agent ready (%s), waiting for %s", analyst.model, ALERTS_ANOMALY.name)

    def publish(rec: dict) -> None:
        table.put_item(Item=to_dynamo({k: v for k, v in rec.items() if v is not None}))
        produce_json(producer, ALERTS_EXPLAINED.name, rec["symbol"], rec)
        producer.poll(0)

    try:
        while not stop.is_set():
            msg = consumer.poll(0.5)
            if msg is not None and not msg.error():
                if not pending:
                    first_at = time.time()
                pending.append(decode(msg.value()))
            if not pending or time.time() - first_at < GROUP_WAIT_S:
                continue
            for group in group_alerts(pending):
                rec = analyst.explain(group[0])
                llm_calls += 1
                publish(rec)
                send(rec)
                done += 1
                log.info("explained %s in %d ms: %s (%.0f%%), %d citation(s), guardrails %s%s",
                         rec["alert_id"], rec["latency_ms"], rec["cause"], rec["confidence"] * 100,
                         len(rec["citations"]), rec["guardrails"] or "none",
                         f", +{len(group) - 1} coin(s) grouped" if len(group) > 1 else "")
                for a in group[1:]:
                    publish(follower_record(rec, a))
                    done += 1
            pending = []
            consumer.commit(asynchronous=False)
    finally:
        producer.flush(10)
        consumer.close()
        log.info("stopped after %d explained alerts (%d LLM calls)", done, llm_calls)


if __name__ == "__main__":
    main()
