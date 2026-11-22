"""Deterministic structural signals extracted from document and chunk metadata.

Derives structural relevance signals without altering chunk text or embedding vectors:
1. S_heading: Overlap between query terms and direct heading / title
2. S_section: Overlap between query terms and section hierarchy path
3. S_element_type: Compatibility between query intent and chunk element types
4. S_table_schema: Overlap between query terms and table headers/captions
5. S_doc_context: Overlap between query terms and document filename/title
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.normalize.schema import Chunk, ElementType, NormalizedDocument
from core.retrieval.representation.structural import (
    extract_document_title,
    extract_section_hierarchies,
    extract_table_and_element_metadata,
)

# Standard deterministic stopwords for lexical matching
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "on",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
    }
)

# Deterministic regex for query intent categories
_NUMERICAL_QUERY_RE = re.compile(
    r"\b(number|port|count|how many|percentage|percent|ratio|rate|metric|score|"
    r"dimension|size|bytes|kb|mb|gb|limit|maximum|minimum|max|min|total|\d+)\b",
    re.IGNORECASE,
)
_PROCEDURAL_QUERY_RE = re.compile(
    r"\b(how to|step|steps|procedure|command|deploy|run|configure|setup|install|"
    r"guide|instruction|instructions|process|workflow)\b",
    re.IGNORECASE,
)
_TABLE_QUERY_RE = re.compile(
    r"\b(table|column|row|schema|headers|matrix|breakdown|comparison|grid)\b",
    re.IGNORECASE,
)


def tokenize_text(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens excluding stopwords."""
    if not text:
        return set()
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return {t for t in raw_tokens if len(t) > 1 and t not in STOPWORDS}


def token_overlap_score(query_tokens: set[str], target_tokens: set[str]) -> float:
    """Compute bounded overlap score between query tokens and target tokens in [0.0, 1.0].

    Uses containment ratio: fraction of query content tokens found in target.
    """
    if not query_tokens or not target_tokens:
        return 0.0
    matching = query_tokens.intersection(target_tokens)
    return min(1.0, len(matching) / len(query_tokens))


@dataclass
class StructuralChunkMetadata:
    """Aggregated structural context for a chunk."""

    document_id: str
    chunk_id: str
    document_title: str
    section_path: list[str] = field(default_factory=list)
    element_types: list[ElementType] = field(default_factory=list)
    table_headers: list[str] = field(default_factory=list)
    table_caption: str = ""
    page_numbers: list[int] = field(default_factory=list)
    has_table: bool = False
    has_heading: bool = False
    has_list: bool = False
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def section_str(self) -> str:
        return " > ".join(self.section_path) if self.section_path else ""


def build_chunk_structural_metadata(
    chunk: Chunk,
    document: NormalizedDocument | None = None,
) -> StructuralChunkMetadata:
    """Extract structural metadata for a chunk from its parent NormalizedDocument."""
    doc_id = chunk.document_id
    chunk_id = chunk.chunk_id

    if document is None:
        pages = sorted({loc.page_number for loc in chunk.locations if loc.page_number})
        return StructuralChunkMetadata(
            document_id=doc_id,
            chunk_id=chunk_id,
            document_title="Unknown Document",
            page_numbers=pages,
        )

    doc_title = extract_document_title(document)
    sections_map = extract_section_hierarchies(document)
    elem_types_map, _ = extract_table_and_element_metadata(document)

    # Determine deepest section path from contributing source elements
    section_path: list[str] = []
    for elem_id in chunk.source_element_ids:
        if elem_id in sections_map and sections_map[elem_id]:
            if len(sections_map[elem_id]) >= len(section_path):
                section_path = sections_map[elem_id]

    # Determine element types
    elem_types: list[ElementType] = [
        elem_types_map[eid] for eid in chunk.source_element_ids if eid in elem_types_map
    ]

    # Check for tables and table metadata
    table_headers: list[str] = []
    table_caption = ""
    has_table = False
    has_heading = False
    has_list = False

    elem_dict = {e.element_id: e for e in document.elements}
    for eid in chunk.source_element_ids:
        elem = elem_dict.get(eid)
        if elem is None:
            continue
        if elem.table_data is not None:
            has_table = True
            if elem.table_data.headers:
                table_headers.extend(str(h) for h in elem.table_data.headers)
            if elem.table_data.caption:
                table_caption = elem.table_data.caption
        if elem.element_type in (ElementType.HEADER, ElementType.HEADING):
            has_heading = True
        if elem.element_type == ElementType.LIST_ITEM:
            has_list = True

    pages = sorted({loc.page_number for loc in chunk.locations if loc.page_number})

    return StructuralChunkMetadata(
        document_id=doc_id,
        chunk_id=chunk_id,
        document_title=doc_title,
        section_path=section_path,
        element_types=elem_types,
        table_headers=table_headers,
        table_caption=table_caption,
        page_numbers=pages,
        has_table=has_table,
        has_heading=has_heading,
        has_list=has_list,
        custom_metadata=dict(chunk.metadata),
    )


def compute_heading_signal(
    query_tokens: set[str],
    metadata: StructuralChunkMetadata,
) -> float:
    """Compute S_heading: Overlap between query terms and the direct heading / title."""
    if not query_tokens:
        return 0.0

    target_tokens: set[str] = set()
    if metadata.section_path:
        # Use direct (leaf) heading
        leaf_heading = metadata.section_path[-1]
        target_tokens.update(tokenize_text(leaf_heading))

    if not target_tokens:
        # Fall back to document title tokens
        target_tokens.update(tokenize_text(metadata.document_title))

    return token_overlap_score(query_tokens, target_tokens)


def compute_section_signal(
    query_tokens: set[str],
    metadata: StructuralChunkMetadata,
) -> float:
    """Compute S_section: Overlap between query terms and full section hierarchy path."""
    if not query_tokens or not metadata.section_path:
        return 0.0

    target_tokens: set[str] = set()
    for heading in metadata.section_path:
        target_tokens.update(tokenize_text(heading))

    return token_overlap_score(query_tokens, target_tokens)


def compute_element_type_signal(
    query: str,
    metadata: StructuralChunkMetadata,
) -> float:
    """Compute S_element_type: Compatibility between query intent and chunk element types.

    Returns bounded score in [0.0, 1.0]:
    - 1.0: Strong intent match (e.g. table query with table chunk, procedural with list)
    - 0.5: Neutral / standard text match
    - 0.0: Mismatched structure (e.g. table query on pure heading chunk)
    """
    if not query.strip():
        return 0.5

    is_numerical = bool(_NUMERICAL_QUERY_RE.search(query))
    is_procedural = bool(_PROCEDURAL_QUERY_RE.search(query))
    is_table_query = bool(_TABLE_QUERY_RE.search(query))

    if is_table_query:
        return 1.0 if metadata.has_table else 0.2

    if is_numerical:
        if metadata.has_table:
            return 1.0
        # If chunk contains digits
        if any(c.isdigit() for c in metadata.document_title + metadata.section_str):
            return 0.8
        return 0.5

    if is_procedural:
        if metadata.has_list:
            return 1.0
        return 0.6

    # General / factual query: neutral baseline
    return 0.5


def compute_table_schema_signal(
    query_tokens: set[str],
    metadata: StructuralChunkMetadata,
) -> float:
    """Compute S_table_schema: Overlap between query terms and table headers / captions.

    Returns 0.0 if chunk contains no table.
    """
    if not metadata.has_table or not query_tokens:
        return 0.0

    target_tokens: set[str] = set()
    for h in metadata.table_headers:
        target_tokens.update(tokenize_text(h))
    if metadata.table_caption:
        target_tokens.update(tokenize_text(metadata.table_caption))

    return token_overlap_score(query_tokens, target_tokens)


def compute_doc_context_signal(
    query_tokens: set[str],
    metadata: StructuralChunkMetadata,
) -> float:
    """Compute S_doc_context: Overlap between query terms and document title / topic."""
    if not query_tokens or not metadata.document_title:
        return 0.0

    target_tokens = tokenize_text(metadata.document_title)
    return token_overlap_score(query_tokens, target_tokens)
