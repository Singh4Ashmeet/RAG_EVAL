"""Canonical retrieval schemas and data structures."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.normalize.schema import Chunk


class RetrievalStrategy(StrEnum):
    """Supported retrieval techniques."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SearchResult(BaseModel):
    """Scored and ranked retrieval candidate linked to a canonical Chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk = Field(
        ...,
        description="Retrieved canonical document chunk.",
    )
    score: float = Field(
        ...,
        description="Relevance or similarity score assigned by the retriever or reranker.",
    )
    rank: int = Field(
        ...,
        ge=0,
        description="Position rank in the result set; must be non-negative.",
    )
    strategy: RetrievalStrategy = Field(
        ...,
        description="Retrieval strategy that produced this candidate.",
    )

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Score must be a finite float, got: {v}")
        return v
