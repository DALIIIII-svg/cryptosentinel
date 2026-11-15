"""S3 helpers (work against SeaweedFS locally and S3 on AWS)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from botocore.exceptions import ClientError

from common.aws_clients import s3_client
from common.config import get_settings


def ensure_bucket(bucket: str | None = None) -> bool:
    """Create the bucket if missing. Returns True if it was created."""
    s = get_settings()
    bucket = bucket or s.s3_bucket
    client = s3_client()
    try:
        client.head_bucket(Bucket=bucket)
        return False
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
    if s.aws_region == "us-east-1" or s.s3_endpoint_url:
        client.create_bucket(Bucket=bucket)
    else:
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": s.aws_region},
        )
    return True


def raw_key(topic: str, ts: datetime, suffix: str) -> str:
    """raw/{topic}/dt=YYYY-MM-DD/hour=HH/part-{suffix}.jsonl.gz (UTC partitions)."""
    ts = ts.astimezone(timezone.utc)
    return f"raw/{topic}/dt={ts:%Y-%m-%d}/hour={ts:%H}/part-{suffix}.jsonl.gz"


def put_jsonl_gz(key: str, records: Iterable[dict[str, Any]], bucket: str | None = None) -> int:
    """Write records as gzip JSON Lines. Returns the number of records written."""
    lines = [json.dumps(r, separators=(",", ":"), default=str) for r in records]
    body = gzip.compress(("\n".join(lines) + "\n").encode("utf-8")) if lines else gzip.compress(b"")
    s3_client().put_object(
        Bucket=bucket or get_settings().s3_bucket,
        Key=key,
        Body=body,
        ContentType="application/x-ndjson",
        ContentEncoding="gzip",
    )
    return len(lines)


def read_jsonl_gz(key: str, bucket: str | None = None) -> list[dict[str, Any]]:
    obj = s3_client().get_object(Bucket=bucket or get_settings().s3_bucket, Key=key)
    text = gzip.decompress(obj["Body"].read()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def list_keys(prefix: str, bucket: str | None = None) -> Iterator[str]:
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket or get_settings().s3_bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            yield item["Key"]
