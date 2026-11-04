"""Validated query/reference dataset contracts."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class AnnotationStatus(StrEnum):
    """Lifecycle and verification state of an evaluation case."""

    VERIFIED = "verified"
    PARTIALLY_ANNOTATED = "partially_annotated"
    UNVERIFIED = "unverified"


class Answerability(StrEnum):
    """Whether the query is answerable from the designated corpus."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    UNDETERMINED = "undetermined"


class QuestionCategory(StrEnum):
    """Structural or behavioral category of the evaluation query."""

    DIRECT_FACTUAL = "direct_factual"
    MULTI_PARAGRAPH = "multi_paragraph"
    MULTI_EVIDENCE = "multi_evidence"
    SPECIFIC_DETAIL = "specific_detail"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"
    MISLEADING_TERMINOLOGY = "misleading_terminology"


class Difficulty(StrEnum):
    """Subjective retrieval and synthesis difficulty."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CorpusTrack(StrEnum):
    """Identifies the independent golden corpus track."""

    TRACK_A = "track_a"
    TRACK_B = "track_b"


class AnnotationProvenance(BaseModel):
    """Auditable provenance and verification metadata for an evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annotator: str = Field(..., min_length=1)
    verified_at: datetime
    source_documents: list[str] = Field(..., min_length=1)
    verification_notes: str | None = None
    evidence_references: list[str] = Field(default_factory=list)

    @field_validator("annotator")
    @classmethod
    def validate_annotator(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("annotator cannot be whitespace-only.")
        return value.strip()

    @field_validator("source_documents")
    @classmethod
    def validate_source_documents(cls, values: list[str]) -> list[str]:
        if any(not val.strip() for val in values):
            raise ValueError("source_documents cannot contain empty or whitespace-only names.")
        normalized = [val.strip() for val in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_documents cannot contain duplicates.")
        return normalized

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_references(cls, values: list[str]) -> list[str]:
        if any(not val.strip() for val in values):
            raise ValueError("evidence_references cannot contain empty strings.")
        return [val.strip() for val in values]


class EvaluationCase(BaseModel):
    """One query and any available retrieval or answer ground truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_element_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # M3 annotation metadata: defaults ensure legacy records remain unverified
    status: AnnotationStatus = Field(default=AnnotationStatus.UNVERIFIED)
    answerability: Answerability = Field(default=Answerability.UNDETERMINED)
    category: QuestionCategory | None = None
    difficulty: Difficulty | None = None
    provenance: AnnotationProvenance | None = None
    track: CorpusTrack | None = None

    @field_validator("query_id", "query")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty or whitespace-only.")
        return value.strip()

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_relevant_chunk_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Relevant chunk IDs cannot be empty or whitespace-only.")
        normalized = [value.strip() for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Relevant chunk IDs must be unique within a query.")
        return normalized

    @field_validator("relevant_element_ids")
    @classmethod
    def validate_relevant_element_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Relevant source element IDs cannot be empty or whitespace-only.")
        normalized = [value.strip() for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Relevant source element IDs must be unique within a query.")
        return normalized

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("reference_answer cannot be whitespace-only.")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_ground_truth_and_status(self) -> Self:
        # Verified cases must carry explicit audit provenance
        if self.status == AnnotationStatus.VERIFIED and self.provenance is None:
            raise ValueError(
                "Evaluation cases marked as verified must include verification provenance."
            )

        # Answerable cases (or undetermined cases) must have at least one ground-truth label
        if self.answerability != Answerability.UNANSWERABLE:
            if (
                not self.relevant_chunk_ids
                and not self.relevant_element_ids
                and self.reference_answer is None
            ):
                raise ValueError(
                    "Each evaluation case must include relevant_chunk_ids, relevant_element_ids, "
                    "reference_answer, or a combination unless answerability is unanswerable."
                )
        return self


class EvaluationDataset(BaseModel):
    """Dataset identity based on its file name and exact source-byte digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    cases: list[EvaluationCase] = Field(..., min_length=1)
    track: CorpusTrack | None = None

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset_id cannot be empty or whitespace-only.")
        return value.strip()

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest.")
        return value

    @model_validator(mode="after")
    def validate_query_ids_unique(self) -> Self:
        ids = [case.query_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("query_id values must be unique within a dataset.")
        return self
