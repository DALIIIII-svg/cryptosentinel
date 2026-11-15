"""What the agent must return, checked with Pydantic (the LLM's JSON is never trusted as is)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Cause = Literal["news_driven", "market_wide", "possible_manipulation", "unexplained"]
CAUSES: tuple[str, ...] = ("news_driven", "market_wide", "possible_manipulation", "unexplained")


class Explanation(BaseModel):
    cause: Cause
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    summary: str = Field(min_length=10, max_length=700)
    cited_news_ids: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("summary")
    @classmethod
    def _strip(cls, v: str) -> str:
        return " ".join(v.split())
