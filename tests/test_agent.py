"""Phase 6 tests: the agent graph and its guardrails, with a fake data source and scripted LLM answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent import graph as graph_mod
from agent import llm
from agent.graph import RiskAnalyst, market_facts
from agent.notify import format_alert

T = 1722820260000  # 2024-08-05 01:11 UTC
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


class FakeSource:
    def __init__(self, crash_all: bool, news: list[dict]):
        self.crash_all, self._news = crash_all, news

    def candles(self, t0, t1):
        rows = []
        for sym in SYMS:
            price = 100.0
            for t in range(t0, t1, 60_000):
                if t >= T - 5 * 60_000 and (self.crash_all or sym == "XRPUSDT"):
                    price *= 0.98
                rows.append({"symbol": sym, "open_time": t, "close": price, "volume": 1.0})
        return pd.DataFrame(rows)

    def news(self, symbol, t0, t1):
        return [n for n in self._news if t0 <= n["published_at"] <= t1]


def item(i, minutes_before, title, coins):
    return {"news_id": f"id{i}", "published_at": T - minutes_before * 60_000, "source": "test",
            "title": title, "summary": "", "url": f"https://ex.com/{i}", "coins": coins}


NEWS = [item(1, 50, "Global markets slide on rate fears", []),
        item(2, 20, "Ripple loses appeal in SEC case", ["XRPUSDT"]),
        item(3, -10, "XRP crashes after court news", ["XRPUSDT"])]   # published AFTER the move


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    from common import config
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def alert(symbol="XRPUSDT"):
    return {"alert_id": "a1", "symbol": symbol, "ts": T, "z_5m": -9.0, "vol_z": 1.0}


def test_market_wide_move_is_classified_market_wide():
    rec = RiskAnalyst(FakeSource(crash_all=True, news=NEWS)).explain(alert("ETHUSDT"))
    assert rec["is_market_wide"] and rec["cause"] == "market_wide"
    assert rec["ret_5m_pct"] < -8


def test_coin_specific_move_cites_only_news_before_the_move():
    rec = RiskAnalyst(FakeSource(crash_all=False, news=NEWS)).explain(alert())
    assert not rec["is_market_wide"] and rec["cause"] == "news_driven"
    assert [c["url"] for c in rec["citations"]] == ["https://ex.com/2"]
    assert all(c["published_at"] <= T for c in rec["citations"])
    assert rec["news_considered"] == 2  # the item published after the move was never shown


def scripted(monkeypatch, answers):
    calls = []

    def fake_complete(system, user):
        calls.append(user)
        return answers[min(len(calls) - 1, len(answers) - 1)]
    monkeypatch.setattr(llm, "complete", fake_complete)
    return calls


def test_invented_citation_triggers_retry_then_is_removed(monkeypatch):
    bad = json.dumps({"cause": "news_driven", "confidence": 0.9, "risk_level": "high",
                      "summary": "Caused by an article that does not exist.", "cited_news_ids": ["N9"]})
    calls = scripted(monkeypatch, [bad, bad])
    rec = RiskAnalyst(FakeSource(False, NEWS)).explain(alert())
    assert len(calls) == 2 and "rejected" in calls[1]
    assert rec["cause"] == "unexplained" and rec["citations"] == [] and rec["confidence"] <= 0.3
    assert "news_driven_without_citation" in rec["guardrails"]


def test_garbage_answers_fall_back_safely(monkeypatch):
    scripted(monkeypatch, ["I think it went down.", "still no json"])
    rec = RiskAnalyst(FakeSource(True, NEWS)).explain(alert("ETHUSDT"))
    assert rec["cause"] == "market_wide" and "llm_fallback" in rec["guardrails"]


def test_llm_error_does_not_crash(monkeypatch):
    def boom(system, user):
        raise TimeoutError("network down")
    monkeypatch.setattr(llm, "complete", boom)
    rec = RiskAnalyst(FakeSource(False, NEWS)).explain(alert())
    assert rec["cause"] == "unexplained" and rec["attempts"] == 2


def test_extract_json_handles_fences():
    assert llm.extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_market_facts_ratio():
    src = FakeSource(crash_all=False, news=[])
    f = market_facts(src.candles(T - 30 * 60_000, T), "XRPUSDT", T)
    assert f["coin_ret_5m_pct"] < -8 and abs(f["others_ret_5m_pct"]) < 0.01 and not f["is_market_wide"]


def test_notification_text():
    rec = RiskAnalyst(FakeSource(False, NEWS)).explain(alert())
    subject, body = format_alert(rec)
    assert "XRP" in subject and "news-driven" in subject
    assert "https://ex.com/2" in body and "not investment advice" in body


def test_eval_faithfulness():
    from eval.eval_agent import faithfulness
    out = pd.DataFrame([{"event_id": "E1", "citations": json.dumps([{"published_at": T - 1, "url": "u"},
                                                                      {"published_at": T + 1, "url": "u"}])}])
    assert faithfulness(out, pd.DataFrame({"event_id": ["E1"], "ts": [T]})) == 0.5


def test_rule_first_scorer(tmp_path, tiny_models):
    from ml.scorer import LocalScorer
    path, f = tiny_models
    meta = json.loads((path / "anomaly_meta.json").read_text())
    meta.update(live_detector="rule_first", thresholds={"combined": 1.5, "rule_first_rule": 5.0, "rule_first_forest": 10.0})
    (path / "anomaly_meta.json").write_text(json.dumps(meta))
    sc = LocalScorer(path)
    rows = f.tail(3).copy()
    rows.iloc[-1, rows.columns.get_loc("ret_5m")] = 0.2
    out = sc.score(rows)
    assert sc.detector == "rule_first" and out["is_anomaly"].tolist() == [False, False, True]


def test_market_facts_come_from_the_alert_when_present():
    # candles say "flat" (hot writer behind) but the alert carries the real moves of every coin
    src = FakeSource(crash_all=False, news=[])
    a = {**alert("BTCUSDT"), "coins_moved_5m_pct": {"BTCUSDT": -6.0, "ETHUSDT": -9.0, "SOLUSDT": -8.0, "XRPUSDT": -7.0}}
    rec = RiskAnalyst(src).explain(a)
    assert rec["ret_5m_pct"] == -6.0 and rec["is_market_wide"] and rec["cause"] == "market_wide"


def test_build_alert_carries_every_coin_move():
    from processing.live_scorer import build_alert, coins_moved
    rows = [{"symbol": "BTCUSDT", "open_time": 0, "ret_5m": -0.06}, {"symbol": "ETHUSDT", "open_time": 0, "ret_5m": -0.09}]
    moved = coins_moved(rows)[0]
    al = build_alert({**rows[0], "close": 1.0}, {"anomaly_score": 2, "anomaly_threshold": 1, "z_5m": -9, "vol_forecast_pct": 1},
                     moved)
    assert al["ret_5m_pct"] == pytest.approx(-5.824, abs=1e-3) and al["others_ret_5m_pct"] == pytest.approx(-8.607, abs=1e-3)
