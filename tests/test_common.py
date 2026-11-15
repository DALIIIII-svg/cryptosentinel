"""Unit tests for the shared modules. No Docker or network needed."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from common import config
from common.aws_clients import client_kwargs
from common.dynamo_io import TABLE_SPECS, to_dynamo
from common.s3_io import raw_key
from common.topics import ALL_TOPICS


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_blank_endpoints_become_none(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    monkeypatch.setenv("DYNAMO_ENDPOINT_URL", "  ")
    s = config.get_settings()
    assert s.s3_endpoint_url is None
    assert s.dynamo_endpoint_url is None


def test_symbol_list_is_clean(monkeypatch):
    monkeypatch.setenv("SYMBOLS", " btcusdt, ETHUSDT ,,solusdt ")
    assert config.get_settings().symbol_list == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_table_name_uses_prefix(monkeypatch):
    monkeypatch.setenv("TABLE_PREFIX", "cryptosentinel")
    assert config.get_settings().table_name("alerts") == "cryptosentinel-alerts"


def test_local_endpoint_gets_dummy_credentials(monkeypatch):
    monkeypatch.setenv("LOCAL_AWS_ACCESS_KEY_ID", "local")
    kw = client_kwargs("http://localhost:8333")
    assert kw["endpoint_url"] == "http://localhost:8333"
    assert kw["aws_access_key_id"] == "local"


def test_aws_endpoint_uses_default_credential_chain():
    kw = client_kwargs(None)
    assert "endpoint_url" not in kw
    assert "aws_access_key_id" not in kw


def test_raw_key_is_partitioned_in_utc():
    ts = datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)
    assert raw_key("prices.klines", ts, "abc") == (
        "raw/prices.klines/dt=2026-11-15/hour=23/part-abc.jsonl.gz"
    )


def test_topics_are_unique_and_named():
    names = [t.name for t in ALL_TOPICS]
    assert len(names) == len(set(names)) == 6
    assert all(t.partitions >= 1 for t in ALL_TOPICS)


def test_table_specs_declare_every_key_attribute():
    for spec in TABLE_SPECS.values():
        declared = {a["AttributeName"] for a in spec["AttributeDefinitions"]}
        keys = {k["AttributeName"] for k in spec["KeySchema"]}
        for gsi in spec.get("GlobalSecondaryIndexes", []):
            keys |= {k["AttributeName"] for k in gsi["KeySchema"]}
        assert keys == declared  # DynamoDB refuses unused or missing attribute definitions


def test_to_dynamo_converts_floats_recursively():
    out = to_dynamo({"a": 1.5, "b": [0.1, {"c": 2.0}], "d": "x", "e": 3})
    assert out == {"a": Decimal("1.5"), "b": [Decimal("0.1"), {"c": Decimal("2.0")}], "d": "x", "e": 3}
