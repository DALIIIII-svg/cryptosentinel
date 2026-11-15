"""Single source of configuration for every CryptoSentinel component.

The same code runs locally and on AWS. Only the env file changes:
  - locally:  .env.local  (copied from .env.example)
  - on AWS:   .env.aws    (copied from .env.aws.example)

Pick the file with the CS_ENV_FILE environment variable (default: .env.local).
Real environment variables always win over the file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_file() -> Path:
    return REPO_ROOT / os.getenv("CS_ENV_FILE", ".env.local")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_file()),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    cs_env: str = "local"
    aws_region: str = "us-east-1"

    # Kafka
    kafka_bootstrap: str = "localhost:9092"

    # S3 / SeaweedFS
    s3_bucket: str = "cryptosentinel"
    s3_endpoint_url: str | None = None

    # DynamoDB / DynamoDB Local
    dynamo_endpoint_url: str | None = None
    table_prefix: str = "cryptosentinel"

    # Dummy credentials, only sent to local emulators (never to real AWS)
    local_aws_access_key_id: str = "local"
    local_aws_secret_access_key: str = "local"

    # Market data
    symbols: str = (
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,PEPEUSDT"
    )
    binance_ws_url: str = "wss://data-stream.binance.vision"
    binance_rest_url: str = "https://data-api.binance.vision"

    # News
    cryptopanic_token: str | None = None

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Models / notifications
    scorer: str = "local"
    notifier: str = "console"
    sns_topic_arn: str | None = None

    @field_validator(
        "s3_endpoint_url",
        "dynamo_endpoint_url",
        "cryptopanic_token",
        "groq_api_key",
        "sns_topic_arn",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    @property
    def is_local(self) -> bool:
        return self.cs_env.lower() == "local"

    def table_name(self, short: str) -> str:
        """'alerts' -> 'cryptosentinel-alerts'."""
        return f"{self.table_prefix}-{short}" if self.table_prefix else short


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
