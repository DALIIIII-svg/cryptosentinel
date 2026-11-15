"""Volatility forecaster: XGBoost vs three baselines, on log realised volatility of the next 60 minutes.

  naive    : next hour's volatility = last hour's            (persistence)
  HAR-RV   : linear regression on log rv over 5 min, 15 min, 1 h and 1 day (standard benchmark)
  GARCH    : GARCH(1,1) per coin, fitted on October 2024 1-minute returns
  XGBoost  : gradient-boosted trees on the 19 features + coin id

Trained on 2024-02..2024-10, early-stopped on 2024-11, scored on 2024-12..2025-01 (unseen).
Writes models/vol_xgb.json, models/vol_meta.json and eval/results_vol.csv.

Run:  python -m ml.train_vol      (or: make train-vol)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common.log import get_logger
from ml import garch
from ml.data import MODELS_DIR, TEST_MONTHS, TRAIN_MONTHS, VAL_MONTHS, load_features, symbol_codes
from ml.metrics import qlike, r2, rmse
from processing.features import FEATURE_COLUMNS, TARGET_COLUMN

log = get_logger("train_vol")
EPS = 1e-8
XGB_FEATURES = [*FEATURE_COLUMNS, "symbol_code"]
HAR_COLUMNS = ["rv_5", "rv_15", "rv_60", "rv_1d"]
GARCH_FIT_MONTH = "2024-10"


def xgb_matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURE_COLUMNS].to_numpy(np.float32)
    return np.column_stack([x, symbol_codes(df["symbol"]).astype(np.float32)])


def har_design(df: pd.DataFrame) -> np.ndarray:
    logs = [np.log(df[c].to_numpy(float) + EPS) for c in HAR_COLUMNS]
    return np.column_stack([np.ones(len(df)), *logs])


def fit_har(train: pd.DataFrame) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(har_design(train), train[TARGET_COLUMN].to_numpy(float), rcond=None)
    return coef


def garch_predictions(fit_df: pd.DataFrame, test_full: pd.DataFrame) -> pd.Series:
    """GARCH forecast for every test row (needs every minute, in order, per coin)."""
    preds = pd.Series(np.nan, index=test_full.index)
    for sym, g in test_full.groupby("symbol"):
        r_fit = fit_df.loc[fit_df["symbol"] == sym].sort_values("open_time")["ret_1m"].to_numpy(float)
        if len(r_fit) < 5000:
            continue
        params = garch.fit(r_fit)
        g = g.sort_values("open_time")
        preds.loc[g.index] = garch.forecast_log_rv(g["ret_1m"].to_numpy(float), params)
        log.info("GARCH %s: omega=%.4g alpha=%.3f beta=%.3f", sym, params.omega, params.alpha, params.beta)
    return preds


def evaluate(name: str, y: np.ndarray, yhat: np.ndarray) -> dict:
    ok = np.isfinite(yhat)
    return {"model": name, "rmse_log_rv": round(rmse(y[ok], yhat[ok]), 4), "qlike": round(qlike(y[ok], yhat[ok]), 4),
            "r2": round(r2(y[ok], yhat[ok]), 4), "n": int(ok.sum())}


def main() -> int:
    t0 = time.time()
    log.info("loading features (train every 3rd minute, validation every 5th, test every minute)")
    train = load_features(TRAIN_MONTHS, every=3)
    val = load_features(VAL_MONTHS, every=5)
    test_full = load_features(TEST_MONTHS, every=1)
    test = test_full[(test_full["open_time"] // 60_000) % 5 == 0]
    log.info("rows: train %d, val %d, test %d (scored)", len(train), len(val), len(test))

    model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=7, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=50, tree_method="hist",
                             early_stopping_rounds=50, n_jobs=-1, eval_metric="rmse")
    model.fit(xgb_matrix(train), train[TARGET_COLUMN].to_numpy(np.float32),
              eval_set=[(xgb_matrix(val), val[TARGET_COLUMN].to_numpy(np.float32))], verbose=200)
    log.info("XGBoost: best iteration %d", model.best_iteration)

    coef = fit_har(train)
    garch_fit = load_features([GARCH_FIT_MONTH], every=1)
    gpred = garch_predictions(garch_fit, test_full).loc[test.index].to_numpy(float)

    y = test[TARGET_COLUMN].to_numpy(float)
    results = pd.DataFrame([
        evaluate("naive (last hour)", y, np.log(test["rv_60"].to_numpy(float) + EPS)),
        evaluate("HAR-RV", y, har_design(test) @ coef),
        evaluate("GARCH(1,1)", y, gpred),
        evaluate("XGBoost", y, model.predict(xgb_matrix(test))),
    ])
    Path("eval").mkdir(exist_ok=True)
    results.to_csv("eval/results_vol.csv", index=False)
    print("\nVolatility forecast, test months", ", ".join(TEST_MONTHS), "(lower RMSE/QLIKE is better)")
    print(results.to_string(index=False))

    imp = pd.Series(model.feature_importances_, index=XGB_FEATURES).sort_values(ascending=False)
    print("\nTop features (XGBoost gain share):")
    for f, v in imp.head(8).items():
        print(f"  {f:14s} {v:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "vol_xgb.json")
    (MODELS_DIR / "vol_meta.json").write_text(json.dumps({
        "features": XGB_FEATURES, "target": TARGET_COLUMN, "har_coef": coef.tolist(),
        "train_months": TRAIN_MONTHS, "val_months": VAL_MONTHS, "test_months": TEST_MONTHS,
        "best_iteration": int(model.best_iteration), "test_results": results.to_dict("records"),
    }, indent=2))
    log.info("saved %s/vol_xgb.json (%.0f s)", MODELS_DIR, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
