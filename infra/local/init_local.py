"""Create everything the local stack needs: Kafka topics, the S3 bucket, DynamoDB tables.

Idempotent: safe to run as often as you like.
    python -m infra.local.init_local
"""
from __future__ import annotations

import sys
import time

from common.config import get_settings
from common.dynamo_io import create_tables
from common.kafka_io import ensure_topics
from common.s3_io import ensure_bucket


def _retry(label: str, fn, attempts: int = 20, delay: float = 3.0):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything while services boot
            if i == attempts:
                print(f"[FAIL] {label}: {exc}")
                raise
            print(f"[wait] {label} not ready ({type(exc).__name__}), retry {i}/{attempts}...")
            time.sleep(delay)


def main() -> int:
    s = get_settings()
    print(f"Environment: {s.cs_env} | Kafka {s.kafka_bootstrap} | S3 {s.s3_endpoint_url or 'AWS'} "
          f"| DynamoDB {s.dynamo_endpoint_url or 'AWS'}")

    created = _retry("Kafka topics", ensure_topics)
    print(f"[ok] Kafka topics ready (created: {created or 'none, all existed'})")

    made = _retry("S3 bucket", ensure_bucket)
    print(f"[ok] S3 bucket '{s.s3_bucket}' ready ({'created' if made else 'existed'})")

    tables = _retry("DynamoDB tables", create_tables)
    print(f"[ok] DynamoDB tables ready (created: {tables or 'none, all existed'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
