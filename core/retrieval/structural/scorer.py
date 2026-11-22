"""Deterministic structural scorer combining individual structural signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.normalize.schema import Chunk, NormalizedDocument
from core.retrieval.structural.signals import (
    StructuralChunkMetadata,
    build_chunk_structural_metadata,
    compute_doc_context_signal,
    compute_element_type_signal,
    compute_heading_signal,
    compute_section_signal,
    compute_table_schema_signal,
    tokenize_text,
)


@dataclass
class StructuralScoreResult:
    """Detailed score result with component breakdowns."""

    total_score: float  # Bounded in [0.0, 1.0]
    s_heading: float  # Bounded in [0.0, 1.0]
    s_section: float  # Bounded in [0.0, 1.0]
    s_element_type: float  # Bounded in [0.0, 1.0]
    s_table_schema: float  # Bounded in [0.0, 1.0]
    s_doc_context: float  # Bounded in [0.0, 1.0]
    active_signals: list[str] = field(default_factory=list)
    metadata: StructuralChunkMetadata | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_score": round(self.total_score, 4),
            "s_heading": round(self.s_heading, 4),
            "s_section": round(self.s_section, 4),
            "s_element_type": round(self.s_element_type, 4),
            "s_table_schema": round(self.s_table_schema, 4),
            "s_doc_context": round(self.s_doc_context, 4),
            "active_signals": self.active_signals,
        }


DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "heading": 0.25,
    "section": 0.25,
    "element_type": 0.20,
    "table_schema": 0.15,
    "doc_context": 0.15,
}


class StructuralScorer:
    """Deterministic structural relevance scorer.

    Combines independent structural signals into a bounded score in [0.0, 1.0].
    Preserves original chunk text and does not interact with dense/sparse embeddings.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        documents: dict[str, NormalizedDocument] | None = None,
    ) -> None:
        self.weights = dict(weights or DEFAULT_SIGNAL_WEIGHTS)
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.documents = documents or {}
        self._metadata_cache: dict[str, StructuralChunkMetadata] = {}

    def get_chunk_metadata(self, chunk: Chunk) -> StructuralChunkMetadata:
        """Get or compute cached structural metadata for a chunk."""
        if chunk.chunk_id not in self._metadata_cache:
            doc = self.documents.get(chunk.document_id)
            self._metadata_cache[chunk.chunk_id] = build_chunk_structural_metadata(chunk, doc)
        return self._metadata_cache[chunk.chunk_id]

    def score(
        self,
        query: str,
        chunk: Chunk,
        document: NormalizedDocument | None = None,
    ) -> StructuralScoreResult:
        """Compute bounded structural score for (query, chunk)."""
        if document is not None:
            meta = build_chunk_structural_metadata(chunk, document)
        else:
            meta = self.get_chunk_metadata(chunk)

        query_tokens = tokenize_text(query)

        s_heading = compute_heading_signal(query_tokens, meta)
        s_section = compute_section_signal(query_tokens, meta)
        s_elem = compute_element_type_signal(query, meta)
        s_table = compute_table_schema_signal(query_tokens, meta)
        s_doc = compute_doc_context_signal(query_tokens, meta)

        raw_components = {
            "heading": s_heading,
            "section": s_section,
            "element_type": s_elem,
            "table_schema": s_table,
            "doc_context": s_doc,
        }

        active = [k for k, v in raw_components.items() if v > 0.0]

        total = sum(self.weights.get(k, 0.0) * v for k, v in raw_components.items())
        total = max(0.0, min(1.0, total))

        return StructuralScoreResult(
            total_score=total,
            s_heading=s_heading,
            s_section=s_section,
            s_element_type=s_elem,
            s_table_schema=s_table,
            s_doc_context=s_doc,
            active_signals=active,
            metadata=meta,
        )
