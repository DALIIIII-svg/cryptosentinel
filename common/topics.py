"""Kafka topics: the one place where names and partition counts are defined."""
from __future__ import annotations

from dataclasses import dataclass

RETENTION_MS_24H = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class Topic:
    name: str
    partitions: int
    description: str


PRICES_KLINES = Topic("prices.klines", 3, "1-min candles from Binance, key = symbol")
PRICES_TRADES = Topic("prices.trades", 3, "aggregated trades from Binance, key = symbol")
NEWS_RAW = Topic("news.raw", 1, "normalised news items (RSS, CryptoPanic)")
FEATURES_1M = Topic("features.1m", 3, "per-symbol feature vectors every minute")
ALERTS_ANOMALY = Topic("alerts.anomaly", 1, "anomaly events from the live scorer")
ALERTS_EXPLAINED = Topic("alerts.explained", 1, "alerts explained by the agent")

ALL_TOPICS: tuple[Topic, ...] = (
    PRICES_KLINES,
    PRICES_TRADES,
    NEWS_RAW,
    FEATURES_1M,
    ALERTS_ANOMALY,
    ALERTS_EXPLAINED,
)

# Used only by the smoke test.
SMOKE = Topic("smoke.test", 1, "smoke test, safe to delete")
