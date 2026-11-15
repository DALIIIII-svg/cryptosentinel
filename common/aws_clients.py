"""boto3 clients that point at local emulators or real AWS depending on config."""
from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config

from common.config import get_settings


def client_kwargs(endpoint_url: str | None) -> dict[str, Any]:
    """Keyword arguments for boto3.client / boto3.resource.

    With an endpoint (local emulator) we send dummy credentials, so a real
    ~/.aws/credentials file is never needed for local work.
    With no endpoint (AWS), boto3 uses its normal credential chain
    (~/.aws/credentials on a laptop, the instance profile on EC2).
    """
    s = get_settings()
    kwargs: dict[str, Any] = {"region_name": s.aws_region}
    if endpoint_url:
        kwargs.update(
            endpoint_url=endpoint_url,
            aws_access_key_id=s.local_aws_access_key_id,
            aws_secret_access_key=s.local_aws_secret_access_key,
        )
    return kwargs


def s3_client():
    s = get_settings()
    # Path-style addressing is required by SeaweedFS and harmless on AWS.
    cfg = Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client("s3", config=cfg, **client_kwargs(s.s3_endpoint_url))


def dynamodb_client():
    s = get_settings()
    return boto3.client("dynamodb", **client_kwargs(s.dynamo_endpoint_url))


def dynamodb_resource():
    s = get_settings()
    return boto3.resource("dynamodb", **client_kwargs(s.dynamo_endpoint_url))
