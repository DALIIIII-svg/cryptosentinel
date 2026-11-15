"""The LLM behind the agent, behind one function: complete(system, user) -> text.

  LLM_PROVIDER=groq     free Groq API (OpenAI-compatible), model LLM_MODEL (default openai/gpt-oss-120b)
  LLM_PROVIDER=ollama   a local model through Ollama's OpenAI-compatible endpoint
  LLM_PROVIDER=bedrock  Amazon Bedrock (Phase 7, needs langchain-aws)
  LLM_PROVIDER=fake     deterministic rules, no network (tests, offline demos)

LangChain chat models are used so that switching provider is a configuration change.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from common.config import get_settings
from common.log import get_logger

log = get_logger("llm")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@lru_cache(maxsize=1)
def _chat_model():
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "groq":
        from langchain_openai import ChatOpenAI
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is empty: add it to .env.local (free key at console.groq.com)")
        extra = {"reasoning_effort": "low"} if "gpt-oss" in s.llm_model else {}
        return ChatOpenAI(model=s.llm_model, api_key=s.groq_api_key, base_url=GROQ_BASE_URL,
                          temperature=0, timeout=60, max_retries=3, **extra)
    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=s.llm_model, api_key="ollama", base_url=f"{s.ollama_base_url}/v1",
                          temperature=0, timeout=180, max_retries=1)
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse  # installed in Phase 7
        return ChatBedrockConverse(model=s.llm_model, region_name=s.aws_region, temperature=0)
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")


def complete(system: str, user: str) -> str:
    if get_settings().llm_provider.lower() == "fake":
        return fake_answer(user)
    msg = _chat_model().invoke([("system", system), ("user", user)])
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def extract_json(text: str) -> dict:
    """First JSON object in the text (models sometimes wrap it in prose or code fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the answer")
    return json.loads(m.group(0))


def fake_answer(user_prompt: str) -> str:
    """Rule-based stand-in for the LLM: reads the FACTS block of the prompt."""
    facts = extract_json(user_prompt.split("FACTS:", 1)[1])
    news = facts.get("news_before_move", [])
    coin_news = [n for n in news if facts["symbol"] in n.get("coins", [])]
    if facts["market"]["is_market_wide"]:
        cause, cited = "market_wide", [n["id"] for n in news[:2]]
    elif coin_news:
        cause, cited = "news_driven", [coin_news[-1]["id"]]
    elif (facts["move"].get("volume_z") or 0) > 3:
        cause, cited = "possible_manipulation", []
    else:
        cause, cited = "unexplained", []
    return json.dumps({"cause": cause, "confidence": 0.6, "risk_level": "medium",
                       "summary": f"Rule-based answer: {cause.replace('_', ' ')} for {facts['symbol']}.",
                       "cited_news_ids": cited})
