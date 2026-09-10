from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Research question")
    scope: Literal["all", "paper"] = "all"
    paper_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("paper_id")
    @classmethod
    def normalize_paper_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def model_post_init(self, __context) -> None:
        if self.scope == "paper" and not self.paper_id:
            raise ValueError("paper_id is required when scope='paper'")
        if self.scope == "all" and self.paper_id:
            raise ValueError("paper_id must be omitted when scope='all'")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
