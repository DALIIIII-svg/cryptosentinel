"""Phase 1 check: write and read back one Kafka message, one S3 object, one DynamoDB item.

    python -m infra.local.smoke
Exit code 0 = the local stack works end to end.
"""
from __future__ import annotations

import sys
import time
import uuid

from common.aws_clients import dynamodb_client, s3_client
from common.config import get_settings
from common.kafka_io import decode, ensure_topics, make_consumer, make_producer, produce_json
from common.s3_io import put_jsonl_gz, read_jsonl_gz
from common.topics import SMOKE


def check_kafka(token: str) -> None:
    ensure_topics([SMOKE])
    producer = make_producer()
    produce_json(producer, SMOKE.name, "smoke", {"token": token})
    if producer.flush(15) != 0:
        raise RuntimeError("Kafka producer could not deliver the message")

    consumer = make_consumer(f"smoke-{token}", [SMOKE.name])
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            if decode(msg.value()).get("token") == token:
                return
        raise RuntimeError("message not read back from Kafka within 30 s")
    finally:
        consumer.close()


def check_s3(token: str) -> None:
    s = get_settings()
    key = f"smoke/{token}.jsonl.gz"
    put_jsonl_gz(key, [{"token": token}])
    rows = read_jsonl_gz(key)
    s3_client().delete_object(Bucket=s.s3_bucket, Key=key)
    if rows != [{"token": token}]:
        raise RuntimeError(f"S3 returned {rows!r}")


def check_dynamodb(token: str) -> None:
    s = get_settings()
    table = s.table_name("alerts")
    client = dynamodb_client()
    item = {"alert_id": {"S": f"smoke-{token}"}, "symbol": {"S": "SMOKE"}, "ts": {"N": "0"}}
    client.put_item(TableName=table, Item=item)
    got = client.get_item(TableName=table, Key={"alert_id": item["alert_id"]}).get("Item")
    client.delete_item(TableName=table, Key={"alert_id": item["alert_id"]})
    if got != item:
        raise RuntimeError(f"DynamoDB returned {got!r}")


def main() -> int:
    token = uuid.uuid4().hex[:12]
    failures = 0
    for label, fn in (("Kafka", check_kafka), ("S3", check_s3), ("DynamoDB", check_dynamodb)):
        try:
            fn(token)
            print(f"[PASS] {label}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
    print("Smoke test passed." if not failures else f"Smoke test failed ({failures} check(s)).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
