"""Anomaly detector: Isolation Forest vs the classic 'z > threshold' rule, at the SAME alert budget.

Three detectors: the rule, the forest alone, and "combined" (the larger of the two, each on its
own scale), which the live scorer uses. Each detector gets a threshold that produces ALERTS_PER_DAY alerts (all 10 coins together, after a
15-minute cooldown per coin) on the validation month. Then, on unseen months, we measure:
  1. recall on the real labelled shock events (eval/shock_events.csv) in Nov 2024 .. Jan 2025
  2. recall on planted anomalies (price jumps, pumps, volume bursts) in the test months
Comparing at a fixed alert budget is fair: any detector can catch everything with enough alerts.

Locally the model is scikit-learn's IsolationForest; on AWS it is replaced by SageMaker's
Random Cut Forest (same idea: isolation of rare points), with the same features and evaluation.

Writes models/anomaly_iforest.joblib, models/anomaly_meta.json, eval/results_anomaly.csv.
Run:  python -m ml.train_anomaly      (or: make train-anomaly)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from common.config import get_settings
from common.log import get_logger
from common.parquet_io import load_klines
from common.schemas import to_epoch_ms
from ml.anomaly import (
    ANOMALY_FEATURES, INJECTION_TYPES, alerts, anomaly_frame, combined_score, detected, inject, rule_score,
    threshold_for_budget,
)
from ml.data import MODELS_DIR, TEST_MONTHS, TRAIN_MONTHS, VAL_MONTHS, load_features
from processing.features import compute_features, ready

log = get_logger("train_anomaly")
ALERTS_PER_DAY = 10
TRAIN_SAMPLE = 300_000


def load_events(months: list[str]) -> pd.DataFrame:
    try:
        ev = pd.read_csv("eval/shock_events.csv")
    except FileNotFoundError:
        log.warning("eval/shock_events.csv missing: run `make shocks` first; skipping real events")
        return pd.DataFrame(columns=["event_id", "end_ms", "leader", "type"])
    ev["end_ms"] = to_epoch_ms(ev["time_utc"])
    month = pd.to_datetime(ev["end_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
    return ev[month.isin(months)].reset_index(drop=True)


def score_frame(f: pd.DataFrame, model: IsolationForest) -> tuple[np.ndarray, np.ndarray]:
    a = anomaly_frame(f)
    return rule_score(a), -model.score_samples(a[ANOMALY_FEATURES].to_numpy(np.float32))


def main() -> int:
    t0 = time.time()
    s = get_settings()
    train = load_features(TRAIN_MONTHS, every=2)
    sample = train.sample(min(TRAIN_SAMPLE, len(train)), random_state=0)
    model = IsolationForest(n_estimators=300, max_samples=2048, random_state=0, n_jobs=-1)
    model.fit(anomaly_frame(sample)[ANOMALY_FEATURES].to_numpy(np.float32))
    del train
    log.info("Isolation Forest trained on %d rows", len(sample))

    # thresholds from the alert budget on the validation month
    val = load_features(VAL_MONTHS, every=1)
    vt, vs = val["open_time"].to_numpy(), val["symbol"].to_numpy()
    v_rule, v_if = score_frame(val, model)
    scale = {"rule": float(np.quantile(v_rule, 0.999)), "iforest": float(np.quantile(v_if, 0.999))}
    v_comb = combined_score(v_rule, v_if, scale)
    thr = {"rule": threshold_for_budget(vt, vs, v_rule, ALERTS_PER_DAY),
           "iforest": threshold_for_budget(vt, vs, v_if, ALERTS_PER_DAY),
           "combined": threshold_for_budget(vt, vs, v_comb, ALERTS_PER_DAY)}
    log.info("thresholds for %d alerts/day: rule |z| > %.2f, iforest > %.4f, combined > %.3f",
             ALERTS_PER_DAY, thr["rule"], thr["iforest"], thr["combined"])

    rows = []
    # 1) real labelled events, validation + test months (unseen by training)
    eval_months = VAL_MONTHS + TEST_MONTHS
    events = load_events(eval_months)
    if len(events):
        f = pd.concat([val, load_features(TEST_MONTHS, every=1)], ignore_index=True)
        t, sym = f["open_time"].to_numpy(), f["symbol"].to_numpy()
        r_s, i_s = score_frame(f, model)
        for name, sc in (("rule", r_s), ("iforest", i_s), ("combined", combined_score(r_s, i_s, scale))):
            al = alerts(t, sym, sc, thr[name])
            hit = detected(events, al)
            for kind in ("market-wide", "coin-specific"):
                m = (events["type"] == kind).to_numpy()
                rows.append({"test": f"real events: {kind}", "detector": name, "events": int(m.sum()),
                             "recall": round(float(hit[m].mean()), 3) if m.any() else None,
                             "alerts_per_day": round(len(al) / max((t.max() - t.min()) / 86_400_000, 1), 1)})
        del f

    # 2) planted anomalies in the test months
    k = load_klines(s.symbol_list, TEST_MONTHS)
    k_inj, planted = inject(k, n_per_type=25, seed=1)
    f = ready(compute_features(k_inj))
    t, sym = f["open_time"].to_numpy(), f["symbol"].to_numpy()
    r_s, i_s = score_frame(f, model)
    for name, sc in (("rule", r_s), ("iforest", i_s), ("combined", combined_score(r_s, i_s, scale))):
        al = alerts(t, sym, sc, thr[name])
        hit = detected(planted, al)
        for kind in INJECTION_TYPES:
            m = (planted["kind"] == kind).to_numpy()
            rows.append({"test": f"planted: {kind}", "detector": name, "events": int(m.sum()),
                         "recall": round(float(hit[m].mean()), 3), "alerts_per_day": None})

    res = pd.DataFrame(rows)
    Path("eval").mkdir(exist_ok=True)
    res.to_csv("eval/results_anomaly.csv", index=False)
    table = res.pivot_table(index="test", columns="detector", values="recall", aggfunc="first")
    table["events"] = res.groupby("test")["events"].first()
    print(f"\nRecall at the same alert budget ({ALERTS_PER_DAY} alerts/day for all coins), unseen months")
    print(table.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "anomaly_iforest.joblib")
    (MODELS_DIR / "anomaly_meta.json").write_text(json.dumps({
        "features": ANOMALY_FEATURES, "alerts_per_day": ALERTS_PER_DAY, "thresholds": thr, "scale": scale,
        "live_detector": "combined",
        "train_months": TRAIN_MONTHS, "val_months": VAL_MONTHS, "test_months": TEST_MONTHS,
        "results": res.to_dict("records"),
    }, indent=2))
    log.info("saved %s/anomaly_iforest.joblib (%.0f s)", MODELS_DIR, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
