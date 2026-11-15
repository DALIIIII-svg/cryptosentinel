"""Live scoring: Kafka features.1m -> scorer -> Kafka alerts.anomaly (key = symbol).

At most one alert per coin per 15 minutes (the first minute of a burst), like in training.
Each alert carries what the agent needs to start: the move, the market move, volume and order-flow
context and the volatility forecast. Replayed data stays marked "replay": true.

Run:  python -m processing.live_scorer        (or: make score-live)
"""
from __future__ import annotations

import signal
import threading

import numpy as np

from common.aws_clients import dynamodb_resource
from common.config import get_settings
from common.dynamo_io import PRICES_HOT_TTL_SECONDS, to_dynamo
from common.kafka_io import decode, make_consumer, make_producer, produce_json
from common.log import get_logger
from common.schemas import now_ms
from common.topics import ALERTS_ANOMALY, FEATURES_1M
from ml.anomaly import COOLDOWN_MS
from ml.scorer import get_scorer
from processing.features import FEATURE_COLUMNS

log = get_logger("live_scorer")


def pct(log_ret) -> float | None:
    return None if log_ret is None else round(float(np.expm1(log_ret)) * 100, 3)


def build_alert(feat: dict, sc: dict) -> dict:
    return {
        "alert_id": f"{feat['symbol']}-{feat['open_time']}",
        "symbol": feat["symbol"],
        "ts": int(feat["open_time"]) + 60_000,           # end of the minute that triggered
        "open_time": int(feat["open_time"]),
        "close": feat.get("close"),
        "anomaly_score": round(float(sc["anomaly_score"]), 3),
        "threshold": round(float(sc["anomaly_threshold"]), 3),
        "z_5m": round(float(sc["z_5m"]), 2),
        "ret_5m_pct": pct(feat.get("ret_5m")),
        "ret_15m_pct": pct(feat.get("ret_15m")),
        "market_ret_5m_pct": pct(feat.get("mkt_ret_5m")),
        "idio_ret_5m_pct": pct(feat.get("idio_ret_5m")),
        "vol_z": feat.get("vol_z"),
        "trades_z": feat.get("trades_z"),
        "imbalance": feat.get("imbalance"),
        "vol_forecast_60m_pct": round(float(sc["vol_forecast_pct"]), 3),
        "replay": bool(feat.get("replay")),
        "detected_at": now_ms(),
    }


def main() -> None:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    scorer = get_scorer()
    forecasts = dynamodb_resource().Table(get_settings().table_name("forecasts"))
    consumer = make_consumer("live-scorer", [FEATURES_1M.name], **{"auto.offset.reset": "latest"})
    producer = make_producer()
    last_alert: dict[str, int] = {}
    scored = sent = 0
    log.info("scoring %s -> %s (threshold %.3f)", FEATURES_1M.name, ALERTS_ANOMALY.name, scorer.threshold)
    try:
        while not stop.is_set():
            msgs = consumer.consume(num_messages=100, timeout=1.0)
            feats = [decode(m.value()) for m in msgs if not m.error()]
            feats = [f for f in feats if all(f.get(c) is not None for c in FEATURE_COLUMNS)]
            if feats:
                result = scorer.score(feats).to_dict("records")
                scored += len(feats)
                # every score goes to the forecasts table (dashboard: forecast vs realised volatility)
                with forecasts.batch_writer(overwrite_by_pkeys=["symbol", "open_time"]) as bw:
                    for feat, sc in zip(feats, result):
                        bw.put_item(Item=to_dynamo({
                            "symbol": feat["symbol"], "open_time": int(feat["open_time"]),
                            "close": feat.get("close"), "vol_forecast_pct": float(sc["vol_forecast_pct"]),
                            "anomaly_score": float(sc["anomaly_score"]), "z_5m": float(sc["z_5m"]),
                            "expires_at": now_ms() // 1000 + PRICES_HOT_TTL_SECONDS}))
                for feat, sc in zip(feats, result):
                    if not sc["is_anomaly"]:
                        continue
                    t, sym = int(feat["open_time"]), feat["symbol"]
                    if sym in last_alert and t - last_alert[sym] < COOLDOWN_MS:
                        continue
                    last_alert[sym] = t
                    alert = build_alert(feat, sc)
                    produce_json(producer, ALERTS_ANOMALY.name, sym, alert)
                    sent += 1
                    log.info("ALERT %s at %s: 5-min move %+.2f%% (z %+.1f), market %+.2f%%, score %.2f",
                             sym, alert["open_time"], alert["ret_5m_pct"] or 0, alert["z_5m"],
                             alert["market_ret_5m_pct"] or 0, alert["anomaly_score"])
            if msgs:
                consumer.commit(asynchronous=True)
                producer.poll(0)
    finally:
        producer.flush(10)
        consumer.close()
        log.info("stopped: %d rows scored, %d alerts", scored, sent)


if __name__ == "__main__":
    main()
