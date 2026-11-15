"""The risk-analyst agent as a LangGraph state machine.

    gather_context -> search_news -> classify (LLM) -> validate --ok--> finalize
                                          ^               | invalid (1 retry)
                                          +---------------+

Design rule: CODE gathers the facts (prices, market move, news before the move) and CHECKS the answer;
the LLM is called once (twice on a bad answer) to judge the cause and write the explanation.
Guardrails enforced in code, whatever the LLM says:
  - only news published BEFORE the move, and returned by search_news, may be cited
  - every citation must point to one of those items (unknown ids are removed)
  - "news_driven" without a valid citation becomes "unexplained"
  - no trading advice is requested or accepted; the output must match the Explanation schema
"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from agent import llm
from agent.schemas import Explanation
from common.config import get_settings
from common.log import get_logger

log = get_logger("agent")
NEWS_LOOKBACK_MS = 65 * 60_000         # the 5-minute move + the hour before it
MAX_NEWS = 25
WIDE_RATIO = 0.35                       # same rule as processing/find_shocks.py

SYSTEM_PROMPT = """You are a crypto market risk analyst. A monitoring system detected an abnormal price
move. Using ONLY the facts provided, decide the most likely cause and explain it briefly.

Causes (choose exactly one):
- news_driven: a news item published before the move plausibly explains it (cite it).
- market_wide: the whole market moved together; the coin followed the market.
- possible_manipulation: a sharp coin-specific move with no explaining news, often with unusual
  volume or one-sided trading (a possible pump or dump). Say "possible", never accuse anyone.
- unexplained: none of the above is supported by the facts.

Rules:
- Cite only news ids from news_before_move (like "N3"), at most 3, and only if they support your cause.
- Headlines that merely mention the coin are not a cause; prefer "unexplained" to guessing.
- No trading advice, no price predictions.
- Answer with ONE JSON object and nothing else:
{"cause": "...", "confidence": 0.0-1.0, "risk_level": "low|medium|high",
 "summary": "2-3 sentences for a trader", "cited_news_ids": ["N1"]}"""


class AgentState(TypedDict, total=False):
    alert: dict
    facts: dict
    news: list[dict]
    news_ids: dict[str, dict]
    attempts: int
    raw: str
    errors: list[str]
    explanation: dict
    record: dict


def _pct(x: float | None) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(np.expm1(x)) * 100, 3)


def market_facts(candles: pd.DataFrame, symbol: str, t_end: int) -> dict:
    """5-minute moves ending at t_end for every coin, the coin's own move and the others' average."""
    c = candles.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
    c = c[c.index < t_end]
    if len(c) < 6 or symbol not in c:
        return {"coin_ret_5m_pct": None, "others_ret_5m_pct": None, "btc_ret_5m_pct": None,
                "is_market_wide": False, "coins_moved": {}}
    r5 = np.log(c.iloc[-1] / c.iloc[-6])
    own = float(r5[symbol])
    others = r5.drop(symbol).dropna()
    oth = float(others.mean()) if len(others) else 0.0
    wide = own != 0 and (oth / own) >= WIDE_RATIO
    return {
        "coin_ret_5m_pct": _pct(own), "others_ret_5m_pct": _pct(oth),
        "btc_ret_5m_pct": _pct(float(r5.get("BTCUSDT", np.nan))) if "BTCUSDT" in r5 else None,
        "is_market_wide": bool(wide),
        "coins_moved": {s.replace("USDT", ""): _pct(float(v)) for s, v in r5.dropna().items()},
    }


def price_path(candles: pd.DataFrame, symbol: str, t_end: int) -> dict:
    c = candles[(candles["symbol"] == symbol) & (candles["open_time"] < t_end)].sort_values("open_time")
    if c.empty:
        return {}
    close = c["close"].to_numpy(float)
    last = close[-1]

    def ago(minutes: int) -> float | None:
        return round((last / close[-1 - minutes] - 1) * 100, 3) if len(close) > minutes else None

    return {"last_price": float(last), "change_5m_pct": ago(5), "change_15m_pct": ago(15),
            "change_60m_pct": ago(60), "high_60m": float(c["close"].tail(60).max()), "low_60m": float(c["close"].tail(60).min())}


class RiskAnalyst:
    """Builds and runs the graph with a data source (DynamoSource or ParquetSource)."""

    def __init__(self, source):
        self.source = source
        self.model = f"{get_settings().llm_provider}:{get_settings().llm_model}"
        g = StateGraph(AgentState)
        g.add_node("gather_context", self.gather_context)
        g.add_node("search_news", self.search_news)
        g.add_node("classify", self.classify)
        g.add_node("validate", self.validate)
        g.add_node("finalize", self.finalize)
        g.set_entry_point("gather_context")
        g.add_edge("gather_context", "search_news")
        g.add_edge("search_news", "classify")
        g.add_edge("classify", "validate")
        g.add_conditional_edges("validate", self.route, {"retry": "classify", "done": "finalize"})
        g.add_edge("finalize", END)
        self.graph = g.compile()

    # ------------------------------------------------------------------ nodes
    def gather_context(self, state: AgentState) -> AgentState:
        a = state["alert"]
        t = int(a["ts"])
        candles = self.source.candles(t - 90 * 60_000, t)
        facts = {
            "symbol": a["symbol"],
            "move_end_utc": pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
            "move": {**price_path(candles, a["symbol"], t),
                     "z_score_5m": a.get("z_5m"), "volume_z": a.get("vol_z"),
                     "buy_sell_imbalance": a.get("imbalance"),
                     "expected_volatility_next_60m_pct": a.get("vol_forecast_60m_pct")},
            "market": market_facts(candles, a["symbol"], t),
        }
        return {"facts": facts, "attempts": 0, "errors": []}

    def search_news(self, state: AgentState) -> AgentState:
        a = state["alert"]
        t = int(a["ts"])
        items = self.source.news(a["symbol"], t - NEWS_LOOKBACK_MS, t)
        items = [n for n in items if int(n["published_at"]) <= t][-MAX_NEWS:]   # guardrail: before the move
        ids = {}
        listing = []
        for i, n in enumerate(items, 1):
            nid = f"N{i}"
            ids[nid] = n
            listing.append({"id": nid, "minutes_before_move": round((t - int(n["published_at"])) / 60_000),
                            "source": n.get("source", ""), "title": n.get("title", ""),
                            "summary": (n.get("summary") or "")[:200], "coins": list(n.get("coins") or [])})
        facts = {**state["facts"], "news_before_move": listing}
        return {"facts": facts, "news": items, "news_ids": ids}

    def classify(self, state: AgentState) -> AgentState:
        user = "FACTS:\n" + json.dumps(state["facts"], ensure_ascii=False, default=str)
        if state.get("errors"):
            user += "\n\nYour previous answer was rejected: " + "; ".join(state["errors"]) + ". Answer again with valid JSON."
        try:
            raw = llm.complete(SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001 - network/model errors become a validation error
            raw = f"LLM_ERROR: {type(exc).__name__}: {exc}"
        return {"raw": raw, "attempts": state.get("attempts", 0) + 1}

    def validate(self, state: AgentState) -> AgentState:
        errors: list[str] = []
        try:
            exp = Explanation.model_validate(llm.extract_json(state["raw"]))
        except (ValueError, ValidationError) as exc:
            return {"errors": [f"invalid answer ({type(exc).__name__}): {str(exc)[:200]}"]}
        unknown = [i for i in exp.cited_news_ids if i not in state["news_ids"]]
        if unknown:
            errors.append(f"cited ids not in news_before_move: {unknown}")
        if exp.cause == "news_driven" and not [i for i in exp.cited_news_ids if i in state["news_ids"]]:
            errors.append("news_driven requires at least one valid citation")
        return {"errors": errors, "explanation": exp.model_dump()}

    def route(self, state: AgentState) -> str:
        return "retry" if state.get("errors") and state.get("attempts", 0) < 2 else "done"

    def finalize(self, state: AgentState) -> AgentState:
        a, facts = state["alert"], state["facts"]
        exp = state.get("explanation")
        guard = []
        if exp is None:  # the LLM never produced valid JSON
            wide = facts["market"]["is_market_wide"]
            exp = {"cause": "market_wide" if wide else "unexplained", "confidence": 0.2, "risk_level": "medium",
                   "summary": "Automatic fallback: the language model did not return a valid answer. "
                              + ("Most coins moved together." if wide else "No explanation could be verified."),
                   "cited_news_ids": []}
            guard.append("llm_fallback")
        valid = [i for i in exp["cited_news_ids"] if i in state["news_ids"]]
        if len(valid) != len(exp["cited_news_ids"]):
            guard.append("removed_invalid_citations")
        if exp["cause"] == "news_driven" and not valid:
            exp = {**exp, "cause": "unexplained", "confidence": min(exp["confidence"], 0.3)}
            guard.append("news_driven_without_citation")
        citations = [{"id": i, "title": state["news_ids"][i]["title"], "url": state["news_ids"][i]["url"],
                      "source": state["news_ids"][i].get("source", ""),
                      "published_at": int(state["news_ids"][i]["published_at"])} for i in valid]
        record = {
            "alert_id": a.get("alert_id") or f"{a['symbol']}-{a['ts']}",
            "symbol": a["symbol"], "ts": int(a["ts"]),
            "cause": exp["cause"], "confidence": float(exp["confidence"]), "risk_level": exp["risk_level"],
            "summary": exp["summary"], "citations": citations,
            "ret_5m_pct": facts["market"].get("coin_ret_5m_pct"),
            "others_ret_5m_pct": facts["market"].get("others_ret_5m_pct"),
            "is_market_wide": facts["market"].get("is_market_wide"),
            "news_considered": len(state.get("news", [])),
            "anomaly_score": a.get("anomaly_score"), "vol_forecast_60m_pct": a.get("vol_forecast_60m_pct"),
            "replay": bool(a.get("replay")), "model": self.model, "attempts": state.get("attempts", 0),
            "guardrails": guard,
        }
        return {"record": record}

    # ------------------------------------------------------------------ entry point
    def explain(self, alert: dict[str, Any]) -> dict:
        t0 = time.time()
        out = self.graph.invoke({"alert": alert})
        rec = out["record"]
        rec["latency_ms"] = int((time.time() - t0) * 1000)
        return rec
