"""Schemas for generated answers and their retrieved source provenance."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.retrieval.schema import SearchResult


class AnswerCitation(BaseModel):
    """A model-emitted source label resolved to a retrieved chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(..., pattern=r"^S[1-9]\d*$")
    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)


class GeneratedAnswer(BaseModel):
    """Answer text plus the complete retrieval context and citation audit."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: list[SearchResult] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    unresolved_citation_labels: list[str] = Field(default_factory=list)
    model_id: str = Field(..., min_length=1)
    abstained: bool = Field(
        ...,
        description="True only when retrieval returned no source and generation was skipped.",
    )
    duration_ms: float = Field(..., ge=0.0)

    @field_validator("query", "answer", "model_id")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty or whitespace-only.")
        return value

    @field_validator("duration_ms")
    @classmethod
    def validate_finite_duration(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("duration_ms must be finite.")
        return value
