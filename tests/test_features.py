"""Phase 4 tests: no look-ahead, live/offline parity, target, gap filling; events and news-time check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processing.check_news_time import event_study, headline_delays, jump_start, lag_correlations, parse_level
from processing.features import (
    DAY_MIN, FEATURE_COLUMNS, MINUTE_MS, TARGET_COLUMN, FeatureState, add_target, compute_features, ready, regularize,
)
from processing.find_shocks import group_events

T0 = 1722816000000  # 2024-08-05 00:00 UTC


def candles(n: int, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for j, sym in enumerate(symbols):
        price = 100.0 * (j + 1)
        for i in range(n):
            o = price
            price *= float(np.exp(rng.normal(0, 0.001)))
            vol = float(rng.lognormal(3, 0.5))
            rows.append({"symbol": sym, "open_time": T0 + i * MINUTE_MS, "close_time": T0 + i * MINUTE_MS + 59_999,
                         "open": o, "high": max(o, price) * 1.0005, "low": min(o, price) * 0.9995, "close": price,
                         "volume": vol, "n_trades": int(vol * 10), "taker_buy_base": vol * float(rng.uniform(0.2, 0.8)),
                         "quote_volume": vol * price, "taker_buy_quote": 0.0, "source": "hist"})
    return pd.DataFrame(rows)


def test_no_look_ahead():
    k = candles(DAY_MIN + 200)
    t = T0 + (DAY_MIN + 100) * MINUTE_MS
    base = compute_features(k)
    k2 = k.copy()
    future = k2["open_time"] > t
    k2.loc[future, ["close", "high", "low", "volume"]] *= 3.0  # wild future
    changed = compute_features(k2)
    a = base[base["open_time"] == t].set_index("symbol")[FEATURE_COLUMNS]
    b = changed[changed["open_time"] == t].set_index("symbol")[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a, b)


def test_live_matches_offline():
    n_warm, n_live = DAY_MIN + 60, 40
    k = candles(n_warm + n_live)
    offline = ready(compute_features(k)).set_index(["symbol", "open_time"])[FEATURE_COLUMNS]
    state = FeatureState()
    live_rows = []
    for rec in k.sort_values(["open_time", "symbol"]).to_dict("records"):
        live_rows += state.add(rec, compute=rec["open_time"] >= T0 + n_warm * MINUTE_MS)
    live = pd.DataFrame(live_rows).set_index(["symbol", "open_time"])[FEATURE_COLUMNS]
    assert len(live) == n_live * 3
    np.testing.assert_allclose(live.to_numpy(float), offline.loc[live.index].to_numpy(float), rtol=1e-9, atol=1e-12)


def test_target_is_future_realised_volatility():
    k = candles(200, symbols=("BTCUSDT",))
    f = add_target(compute_features(k), horizon=60)
    i = 50
    r = np.log(k["close"] / k["close"].shift(1))
    expected = np.log(np.sqrt((r.iloc[i + 1:i + 61] ** 2).sum()) + 1e-8)
    assert f.loc[f["open_time"] == T0 + i * MINUTE_MS, TARGET_COLUMN].iloc[0] == pytest.approx(expected)
    assert f[TARGET_COLUMN].tail(60).isna().all()  # no future = no target


def test_gaps_are_filled_and_flagged():
    k = candles(10, symbols=("BTCUSDT",)).drop(index=[4, 5])
    g = regularize(k)
    assert len(g) == 10 and g["filled"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert g.loc[4, "close"] == g.loc[3, "close"] and g.loc[4, "volume"] == 0


def test_feature_ranges():
    f = ready(compute_features(candles(DAY_MIN + 30)))
    assert len(f) and f["imbalance"].between(-1, 1).all() and (f["range_pct"] >= 0).all()
    assert f["hour_sin"].between(-1, 1).all()
    np.testing.assert_allclose(f["idio_ret_5m"], f["ret_5m"] - f["mkt_ret_5m"])


def test_group_events_labels_market_wide_and_coin_specific():
    t = T0 + 3_600_000
    shocks = pd.DataFrame([
        {"symbol": "BTCUSDT", "end_ms": t, "time_utc": t, "ret_5m_pct": -6.0, "market_ret_5m_pct": -7.0, "others_ret_5m_pct": -7.1, "score": 50},
        {"symbol": "ETHUSDT", "end_ms": t + 60_000, "time_utc": t, "ret_5m_pct": -9.0, "market_ret_5m_pct": -7.0, "others_ret_5m_pct": -6.8, "score": 60},
        {"symbol": "SOLUSDT", "end_ms": t + 120_000, "time_utc": t, "ret_5m_pct": -8.0, "market_ret_5m_pct": -7.0, "others_ret_5m_pct": -6.9, "score": 55},
        # many coins wobble with it, but the others barely moved -> still coin-specific
        {"symbol": "XRPUSDT", "end_ms": t + 86_400_000, "time_utc": t, "ret_5m_pct": -9.0, "market_ret_5m_pct": -1.5, "others_ret_5m_pct": -0.7, "score": 40},
        {"symbol": "ADAUSDT", "end_ms": t + 86_460_000, "time_utc": t, "ret_5m_pct": -3.0, "market_ret_5m_pct": -1.5, "others_ret_5m_pct": -1.3, "score": 10},
        {"symbol": "DOGEUSDT", "end_ms": t + 86_520_000, "time_utc": t, "ret_5m_pct": -3.0, "market_ret_5m_pct": -1.5, "others_ret_5m_pct": -1.3, "score": 10},
    ])
    ev = group_events(shocks).set_index("leader")
    assert ev.loc["ETHUSDT", "type"] == "market-wide" and ev.loc["ETHUSDT", "n_coins"] == 3
    assert ev.loc["XRPUSDT", "type"] == "coin-specific" and ev.loc["XRPUSDT", "n_coins"] == 3


def test_lag_check_finds_a_known_shift():
    rng = np.random.default_rng(3)
    hours = np.arange(24 * 60) * 3_600_000 + T0
    spikes = rng.exponential(1.0, len(hours)) ** 3
    activity = pd.Series(spikes, index=hours)
    true_news = pd.Series(np.roll(spikes, 1) * 5 + rng.normal(0, 0.1, len(hours)), index=hours)  # 1 h after
    assert max(lag_correlations(activity, true_news), key=lag_correlations(activity, true_news).get) == 1
    shifted = pd.Series(true_news.values, index=hours - 5 * 3_600_000)  # clock 5 h early
    corr = lag_correlations(activity, shifted)
    assert max(corr, key=corr.get) == -4


def test_event_study_detects_a_late_clock():
    rng = np.random.default_rng(5)
    events = [T0 + i * 7 * 86_400_000 for i in range(8)]
    background = rng.integers(T0 - 86_400_000, T0 + 60 * 86_400_000, 20000)
    def news_with_delay(delay_min):
        burst = [e + (delay_min + int(x)) * 60_000 for e in events for x in rng.uniform(0, 120, 60)]
        return pd.DataFrame({"published_at": np.r_[background, burst]})
    assert jump_start(event_study(news_with_delay(5), events)) in (-30, 0)
    assert 180 <= jump_start(event_study(news_with_delay(210), events)) <= 210  # clock 3.5 h late


def test_parse_level():
    assert parse_level("Bitcoin crashes below $53K, wiping out $600M in leveraged longs") == (53000.0, "down")
    assert parse_level("BTC tops $70,000 for the first time since March") == (70000.0, "up")
    assert parse_level("Bitcoin ETF inflows reach $500M") is None
    assert parse_level("Ethereum falls below $2,800") is None


def test_headline_delays_measure_the_clock():
    t = np.arange(48 * 60) * MINUTE_MS + T0
    price = np.where(np.arange(len(t)) < 30 * 60, 60000.0, 52000.0)  # crosses below 53K at hour 30
    btc = pd.DataFrame({"open_time": t, "low": price, "high": price})
    cross = T0 + 30 * 3_600_000
    news = pd.DataFrame({"title": ["Bitcoin falls below $53K"] * 2, "published_at": [cross + 20 * MINUTE_MS, cross + 210 * MINUTE_MS]})
    assert headline_delays(news, btc).tolist() == [20.0, 210.0]
