"""Phase 5 tests: GARCH, metrics, alert budget/cooldown, event matching, injections, scorer."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import IsolationForest

from ml import garch
from ml.anomaly import (
    ANOMALY_FEATURES, alerts, anomaly_frame, combined_score, detected, inject, rule_score, threshold_for_budget,
)
from ml.metrics import qlike, r2, rmse
from processing.features import FEATURE_COLUMNS, compute_features, ready
from tests.test_features import MINUTE_MS, T0, candles


def simulate_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = rng.normal(0, np.sqrt(s2))
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r / garch.SCALE


def test_garch_fit_recovers_parameters():
    p = garch.fit(simulate_garch(20000, 0.02, 0.08, 0.90))
    assert p.alpha == pytest.approx(0.08, abs=0.03) and p.beta == pytest.approx(0.90, abs=0.04)


def test_garch_filter_matches_the_recursion():
    r = simulate_garch(500, 0.02, 0.1, 0.85, seed=1) * garch.SCALE
    p = garch.GarchParams(0.02, 0.1, 0.85)
    s2 = garch.filter_variance(r, p, sigma2_0=0.3)
    loop = [0.3]
    for t in range(1, len(r)):
        loop.append(p.omega + p.alpha * r[t - 1] ** 2 + p.beta * loop[-1])
    np.testing.assert_allclose(s2, loop, rtol=1e-10)
    f = garch.forecast_log_rv(r / garch.SCALE, p, horizon=60)
    assert np.isfinite(f).all() and len(f) == len(r)


def test_metrics():
    y = np.array([-5.0, -4.0, -6.0])
    assert rmse(y, y) == 0 and qlike(y, y) == pytest.approx(0) and r2(y, y) == 1
    assert qlike(y, y - 0.5) > 0  # under-forecasting volatility is penalised


def test_alerts_respect_cooldown_and_budget():
    times = np.arange(0, 2 * 86_400_000, 60_000)
    symbols = np.array(["A"] * len(times))
    scores = np.zeros(len(times))
    scores[100:110] = 5            # a burst = one alert
    scores[2000] = 6
    al = alerts(times, symbols, scores, 1.0)
    assert al["open_time"].tolist() == [times[100], times[2000]]
    rng = np.random.default_rng(0)
    noisy = rng.exponential(1, len(times))
    thr = threshold_for_budget(times, symbols, noisy, alerts_per_day=10)
    assert 15 <= len(alerts(times, symbols, noisy, thr)) <= 25


def test_detected_market_wide_vs_coin_specific():
    ev = pd.DataFrame([{"end_ms": 1_000_000, "leader": "XRPUSDT", "type": "coin-specific"},
                       {"end_ms": 1_000_000, "leader": "ETHUSDT", "type": "market-wide"}])
    al = pd.DataFrame({"open_time": [1_000_000 + 120_000], "symbol": ["BTCUSDT"], "score": [3.0]})
    assert detected(ev, al).tolist() == [False, True]


def test_injected_price_jump_is_detected_by_the_rule():
    k = candles(4 * 1440, symbols=("BTCUSDT", "ETHUSDT"))
    k_inj, planted = inject(k, n_per_type=3, seed=2)
    assert set(planted["kind"]) == {"price_jump", "pump", "volume_burst"}
    f = ready(compute_features(k_inj))
    a = anomaly_frame(f)
    jumps = planted[planted["kind"] == "price_jump"]
    for _, e in jumps.iterrows():
        near = (f["symbol"] == e["leader"]) & ((f["open_time"] - e["end_ms"]).abs() <= 5 * MINUTE_MS)
        if near.any():
            assert rule_score(a)[near.to_numpy()].max() > 4


@pytest.fixture()
def tiny_models(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
    from common import config
    config.get_settings.cache_clear()
    f = ready(compute_features(candles(1440 + 300)))
    a = anomaly_frame(f)[ANOMALY_FEATURES].to_numpy(np.float32)
    joblib.dump(IsolationForest(n_estimators=50, random_state=0).fit(a), tmp_path / "anomaly_iforest.joblib")
    (tmp_path / "anomaly_meta.json").write_text(json.dumps(
        {"thresholds": {"combined": 1.5}, "scale": {"rule": 3.0, "iforest": 0.6}}))
    x = np.column_stack([f[FEATURE_COLUMNS].to_numpy(np.float32), np.zeros(len(f), np.float32)])
    names = [*FEATURE_COLUMNS, "symbol_code"]
    booster = xgb.train({"max_depth": 2}, xgb.DMatrix(x, label=np.full(len(f), -5.0), feature_names=names), 5)
    booster.save_model(tmp_path / "vol_xgb.json")
    (tmp_path / "vol_meta.json").write_text(json.dumps({"features": names}))
    yield tmp_path, f
    config.get_settings.cache_clear()


def test_local_scorer_round_trip(tiny_models):
    from ml.scorer import LocalScorer
    path, f = tiny_models
    sc = LocalScorer(path)
    rows = f.tail(6).copy()
    rows.iloc[-1, rows.columns.get_loc("ret_5m")] = 0.2  # a huge 5-minute move on the last row
    out = sc.score(rows.to_dict("records"))
    assert list(out["symbol"]) == list(rows["symbol"]) and out["is_anomaly"].iloc[-1]
    assert (out["vol_forecast_pct"] > 0).all()
    assert combined_score(np.array([6.0]), np.array([0.3]), {"rule": 3.0, "iforest": 0.6})[0] == 2.0


def test_missing_models_message(tmp_path):
    from ml.scorer import LocalScorer
    with pytest.raises(FileNotFoundError, match="make train"):
        LocalScorer(tmp_path)


def test_load_events_reads_times_in_milliseconds(tmp_path, monkeypatch):
    from ml.train_anomaly import load_events
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval").mkdir()
    pd.DataFrame([{"event_id": "E1", "time_utc": "2024-12-05 22:29:00+00:00", "type": "market-wide", "leader": "BTCUSDT"},
                  {"event_id": "E2", "time_utc": "2024-08-05 01:11:00+00:00", "type": "market-wide", "leader": "ETHUSDT"}]
                 ).to_csv(tmp_path / "eval" / "shock_events.csv", index=False)
    ev = load_events(["2024-12"])
    assert ev["event_id"].tolist() == ["E1"] and ev["end_ms"].iloc[0] == 1733437740000
