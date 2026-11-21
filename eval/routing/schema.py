"""Validated schemas for Phase 11 independent routing dataset."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RoutingSplit(StrEnum):
    """Dataset partition assignment with document-level separation."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class QueryCategory(StrEnum):
    """Taxonomy of queries for retrieval routing diversity."""

    DIRECT_FACTUAL = "direct_factual"
    TECHNICAL_IDENTIFIER = "technical_identifier"
    NUMERICAL = "numerical"
    PROCEDURAL = "procedural"
    SPECIFIC_DETAIL = "specific_detail"
    CONFIGURATION = "configuration"
    MULTI_STEP = "multi_step"


class OracleStrategy(StrEnum):
    """Optimal retrieval strategy identified by the oracle evaluator."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    AMBIGUOUS = "ambiguous"
    NO_CLEAR_WINNER = "no_clear_winner"


class RetrievalMetrics(BaseModel):
    """Standardized retrieval metrics at specified cutoffs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)
    mrr_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)


class RoutingExample(BaseModel):
    """An independent query example for retrieval strategy routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(..., min_length=1)
    source_document_id: str = Field(..., min_length=1)
    source_document: str = Field(..., min_length=1)
    source_element_ids: list[str] = Field(..., min_length=1)
    source_chunk_ids: list[str] = Field(..., min_length=1)
    generation_method: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    query_category: QueryCategory
    expected_answer: str = Field(..., min_length=1)
    candidate_retrieval_strategies: list[str] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid"]
    )
    oracle_strategy: OracleStrategy = Field(default=OracleStrategy.AMBIGUOUS)
    generation_seed: int
    dataset_version: str = Field(default="1.0.0")
    split: RoutingSplit
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", "expected_answer", "example_id", "source_document")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only.")
        return value.strip()

    @field_validator("source_chunk_ids", "source_element_ids")
    @classmethod
    def validate_id_list(cls, values: list[str]) -> list[str]:
        if any(not v.strip() for v in values):
            raise ValueError("ID list cannot contain empty or whitespace-only items.")
        return [v.strip() for v in values]


class OracleLabelRecord(BaseModel):
    """Comprehensive oracle evaluation record for a query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    source_document: str = Field(..., min_length=1)
    split: RoutingSplit
    bm25_metrics: RetrievalMetrics
    dense_metrics: RetrievalMetrics
    hybrid_metrics: RetrievalMetrics
    oracle_strategy: OracleStrategy
    is_tie: bool
    margin: float = Field(ge=0.0)


class RoutingDatasetManifest(BaseModel):
    """Full manifest cataloging dataset artifacts, distributions, and integrity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    dataset_version: str
    created_at: str
    generation_seed: int
    total_examples: int
    split_counts: dict[str, int]
    document_splits: dict[str, list[str]]
    oracle_distribution: dict[str, int]
    category_distribution: dict[str, int]
    length_distribution: dict[str, int]
    source_document_count: int
    files: dict[str, str]
    leakage_summary: dict[str, Any]


class LeakageAuditReport(BaseModel):
    """Audit report demonstrating zero contamination with protected evaluation data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_timestamp: str
    protected_queries_count: int
    evaluated_queries_count: int
    exact_matches_found: int
    normalized_matches_found: int
    fuzzy_matches_found: int
    max_fuzzy_similarity: float
    fuzzy_threshold: float
    source_separation_verified: bool
    protected_dataset_isolation_verified: bool
    status: str
    diagnostics: list[str]
