"""Canonical document ingestion schemas and contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.normalize.schema import Chunk, NormalizedDocument


class IngestionStatus(StrEnum):
    """Status outcomes for document ingestion processing."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class IngestionResult(BaseModel):
    """Outcome of document ingestion, linking normalized output, chunks, and errors."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        min_length=1,
        description="Canonical document identifier.",
    )
    status: IngestionStatus = Field(
        ...,
        description="Overall ingestion outcome status.",
    )
    source_path: Path = Field(
        ...,
        description="Path to the source file processed by the pipeline.",
    )
    document: NormalizedDocument | None = Field(
        default=None,
        description="Extracted canonical document, if parsing produced output.",
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Extracted canonical chunks, if chunking produced output.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages or failure details recorded during processing.",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Pipeline processing duration in milliseconds.",
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id cannot be empty or whitespace-only.")
        return v

    @field_validator("duration_ms")
    @classmethod
    def validate_finite_duration(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"duration_ms must be a finite float, got: {v}")
        return v
