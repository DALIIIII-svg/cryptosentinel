"""One interface for both models, so the rest of the system never knows where they run.

  LocalScorer      loads the files written by `make train-vol` / `make train-anomaly` (models/)
  SageMakerScorer  calls the SageMaker endpoints (Phase 7; same inputs and outputs)

score(features) takes feature rows (dicts or a DataFrame from processing.features) and returns
one row per input with:
  anomaly_score, anomaly_threshold, is_anomaly   (combined detector, alert budget from training)
  z_5m                                           (5-minute move in usual-move units, signed)
  iforest_score
  vol_forecast_pct                               (expected volatility over the next 60 min, in %)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from common.config import get_settings
from ml.anomaly import ANOMALY_FEATURES, anomaly_frame, combined_score, rule_score
from ml.data import MODELS_DIR, symbol_codes
from processing.features import FEATURE_COLUMNS


class LocalScorer:
    def __init__(self, models_dir: Path | str = MODELS_DIR):
        d = Path(models_dir)
        missing = [p for p in ("anomaly_iforest.joblib", "anomaly_meta.json", "vol_xgb.json", "vol_meta.json")
                   if not (d / p).exists()]
        if missing:
            raise FileNotFoundError(f"missing {missing} in {d}: run `make train` first")
        self.forest = joblib.load(d / "anomaly_iforest.joblib")
        self.anomaly_meta = json.loads((d / "anomaly_meta.json").read_text())
        self.vol = xgb.Booster()
        self.vol.load_model(d / "vol_xgb.json")
        self.vol_meta = json.loads((d / "vol_meta.json").read_text())
        self.threshold = float(self.anomaly_meta["thresholds"]["combined"])
        self.scale = self.anomaly_meta["scale"]

    def score(self, rows) -> pd.DataFrame:
        f = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        f = f.reset_index(drop=True)
        a = anomaly_frame(f)
        rule = rule_score(a)
        forest = -self.forest.score_samples(a[ANOMALY_FEATURES].to_numpy(np.float32))
        comb = combined_score(rule, forest, self.scale)
        x = np.column_stack([f[FEATURE_COLUMNS].to_numpy(np.float32), symbol_codes(f["symbol"]).astype(np.float32)])
        log_rv = self.vol.predict(xgb.DMatrix(x, feature_names=self.vol_meta["features"]))
        return pd.DataFrame({
            "symbol": f["symbol"], "open_time": f["open_time"],
            "anomaly_score": comb, "anomaly_threshold": self.threshold, "is_anomaly": comb > self.threshold,
            "z_5m": a["r5n"].to_numpy(float), "iforest_score": forest,
            "vol_forecast_pct": np.exp(log_rv.astype(float)) * 100,
        })


class SageMakerScorer:
    def __init__(self) -> None:
        raise NotImplementedError("SageMaker endpoints are deployed in Phase 7; use SCORER=local for now")


def get_scorer():
    return SageMakerScorer() if get_settings().scorer == "sagemaker" else LocalScorer()
