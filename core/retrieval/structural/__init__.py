"""Structural metadata scoring and fusion package."""

from __future__ import annotations

from core.retrieval.structural.fusion import (
    StructuralFusionStrategy,
    fuse_structural,
)
from core.retrieval.structural.scorer import (
    DEFAULT_SIGNAL_WEIGHTS,
    StructuralScorer,
    StructuralScoreResult,
)
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

__all__ = [
    "DEFAULT_SIGNAL_WEIGHTS",
    "StructuralChunkMetadata",
    "StructuralFusionStrategy",
    "StructuralScoreResult",
    "StructuralScorer",
    "build_chunk_structural_metadata",
    "compute_doc_context_signal",
    "compute_element_type_signal",
    "compute_heading_signal",
    "compute_section_signal",
    "compute_table_schema_signal",
    "fuse_structural",
    "tokenize_text",
]
