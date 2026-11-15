"""DynamoDB tables (hot data) and helpers. Works against DynamoDB Local and AWS.

Tables (names get the TABLE_PREFIX, e.g. cryptosentinel-alerts):
  prices-hot : PK symbol (S), SK open_time (N)            TTL expires_at (48 h)
  news       : PK coin (S),   SK sk (S) = published_at#news_id   TTL expires_at (30 days)
  forecasts  : PK symbol (S), SK open_time (N)            TTL expires_at (48 h)  - live scores
  alerts     : PK alert_id (S); GSI by-symbol on symbol (S) + ts (N)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

from common.aws_clients import dynamodb_client
from common.config import get_settings

PRICES_HOT_TTL_SECONDS = 48 * 3600
NEWS_TTL_SECONDS = 30 * 24 * 3600

TABLE_SPECS: dict[str, dict[str, Any]] = {
    "prices-hot": {
        "KeySchema": [
            {"AttributeName": "symbol", "KeyType": "HASH"},
            {"AttributeName": "open_time", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "open_time", "AttributeType": "N"},
        ],
        "ttl_attribute": "expires_at",
    },
    "news": {
        "KeySchema": [
            {"AttributeName": "coin", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "coin", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "ttl_attribute": "expires_at",
    },
    "forecasts": {
        "KeySchema": [
            {"AttributeName": "symbol", "KeyType": "HASH"},
            {"AttributeName": "open_time", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "open_time", "AttributeType": "N"},
        ],
        "ttl_attribute": "expires_at",
    },
    "alerts": {
        "KeySchema": [{"AttributeName": "alert_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "alert_id", "AttributeType": "S"},
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "ts", "AttributeType": "N"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "by-symbol",
                "KeySchema": [
                    {"AttributeName": "symbol", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        "ttl_attribute": None,
    },
}


def create_tables(tags: list[dict[str, str]] | None = None) -> list[str]:
    """Create missing tables (on-demand billing). Returns names created."""
    s = get_settings()
    client = dynamodb_client()
    existing = set(client.list_tables().get("TableNames", []))
    created: list[str] = []
    for short, spec in TABLE_SPECS.items():
        name = s.table_name(short)
        if name in existing:
            continue
        kwargs: dict[str, Any] = {
            "TableName": name,
            "KeySchema": spec["KeySchema"],
            "AttributeDefinitions": spec["AttributeDefinitions"],
            "BillingMode": "PAY_PER_REQUEST",
        }
        if spec.get("GlobalSecondaryIndexes"):
            kwargs["GlobalSecondaryIndexes"] = spec["GlobalSecondaryIndexes"]
        if tags and not s.dynamo_endpoint_url:
            kwargs["Tags"] = tags
        client.create_table(**kwargs)
        client.get_waiter("table_exists").wait(TableName=name)
        if spec.get("ttl_attribute"):
            try:
                client.update_time_to_live(
                    TableName=name,
                    TimeToLiveSpecification={"Enabled": True, "AttributeName": spec["ttl_attribute"]},
                )
            except ClientError:
                pass  # TTL is best-effort locally
        created.append(name)
    return created


def to_dynamo(value: Any) -> Any:
    """Convert floats to Decimal recursively (boto3 resource API refuses floats)."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: to_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_dynamo(v) for v in value]
    return value
