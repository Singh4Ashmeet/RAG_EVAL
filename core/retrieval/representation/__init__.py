"""Document-structure-enriched representations for retrieval."""

from __future__ import annotations

from core.retrieval.representation.base import ChunkRepresentation
from core.retrieval.representation.baseline import BaselineRepresentation
from core.retrieval.representation.structural import (
    FullStructuralRepresentation,
    HeadingEnrichedRepresentation,
    extract_document_title,
    extract_section_hierarchies,
    extract_table_and_element_metadata,
)


def create_representation(name: str = "baseline", **kwargs) -> ChunkRepresentation:
    """Factory to create a chunk representation by name."""
    norm_name = name.lower().strip()
    if norm_name in ("baseline", "raw", "none", "a"):
        return BaselineRepresentation()
    elif norm_name in ("heading", "heading_enriched", "hierarchy", "b"):
        return HeadingEnrichedRepresentation(**kwargs)
    elif norm_name in ("full", "full_structural", "structural", "c"):
        return FullStructuralRepresentation(**kwargs)
    else:
        raise ValueError(
            f"Unknown representation '{name}'. "
            "Expected: 'baseline', 'heading_enriched', 'full_structural'."
        )


__all__ = [
    "BaselineRepresentation",
    "ChunkRepresentation",
    "FullStructuralRepresentation",
    "HeadingEnrichedRepresentation",
    "create_representation",
    "extract_document_title",
    "extract_section_hierarchies",
    "extract_table_and_element_metadata",
]
