"""Unit tests for Phase 16 Document-Structure-Enriched Retrieval & Representation Forensics."""

from __future__ import annotations

from datetime import UTC, datetime

from core.normalize.schema import (
    BoundingBox,
    Chunk,
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
    TableData,
)
from core.retrieval.representation import (
    BaselineRepresentation,
    FullStructuralRepresentation,
    HeadingEnrichedRepresentation,
    create_representation,
    extract_document_title,
    extract_section_hierarchies,
    extract_table_and_element_metadata,
)


def _make_sample_doc() -> NormalizedDocument:
    """Helper to create a fully specified NormalizedDocument for tests."""
    meta = DocumentMetadata(
        document_id="doc_sample_123",
        filename="system_architecture.md",
        content_sha256="a" * 64,
        byte_size=1024,
        created_at=datetime.now(UTC),
        processed_at=datetime.now(UTC),
    )

    elements = [
        ExtractedElement(
            element_id="e0001",
            element_type=ElementType.HEADER,
            content="# System Architecture",
            order_index=0,
            location=PageLocation(page_number=1, bbox=BoundingBox(x0=0, y0=0, x1=1, y1=0.1)),
        ),
        ExtractedElement(
            element_id="e0002",
            element_type=ElementType.TEXT,
            content="This document describes the core system architecture.",
            order_index=1,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="e0003",
            element_type=ElementType.HEADER,
            content="## Subsystem Components",
            order_index=2,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="e0004",
            element_type=ElementType.TEXT,
            content="The retrieval engine indexes vectors into SQLite.",
            order_index=3,
            location=PageLocation(page_number=2),
        ),
        ExtractedElement(
            element_id="e0005",
            element_type=ElementType.TABLE,
            content="Table of node specifications and limits.",
            order_index=4,
            location=PageLocation(page_number=2),
            table_data=TableData(
                headers=["Node ID", "RAM (GB)", "vCPU"],
                rows=[["worker-1", "32", "8"], ["worker-2", "64", "16"]],
                caption="Resource Allocations",
            ),
        ),
    ]

    return NormalizedDocument(metadata=meta, elements=elements, page_count=2)


def _make_sample_chunk(
    chunk_id: str = "doc_sample_123_c0001",
    elem_ids: list[str] | None = None,
    text: str = "The retrieval engine indexes vectors into SQLite.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc_sample_123",
        text=text,
        chunk_index=1,
        source_element_ids=elem_ids or ["e0004"],
        locations=[PageLocation(page_number=2)],
        metadata={"chunker": "ElementAwareChunker"},
    )


class TestRepresentationExtraction:
    """Tests for document title, heading hierarchy, and structural metadata extraction."""

    def test_extract_document_title_from_header(self) -> None:
        doc = _make_sample_doc()
        title = extract_document_title(doc)
        assert title == "System Architecture"

    def test_extract_document_title_fallback_filename(self) -> None:
        meta = DocumentMetadata(
            document_id="doc_test",
            filename="user_manual_v2.pdf",
            content_sha256="b" * 64,
        )
        doc = NormalizedDocument(metadata=meta, elements=[])
        title = extract_document_title(doc)
        assert title == "User Manual V2"

    def test_extract_section_hierarchies(self) -> None:
        doc = _make_sample_doc()
        hierarchies = extract_section_hierarchies(doc)

        assert hierarchies["e0001"] == ["System Architecture"]
        assert hierarchies["e0002"] == ["System Architecture"]
        assert hierarchies["e0003"] == ["System Architecture", "Subsystem Components"]
        assert hierarchies["e0004"] == ["System Architecture", "Subsystem Components"]
        assert hierarchies["e0005"] == ["System Architecture", "Subsystem Components"]

    def test_extract_table_metadata(self) -> None:
        doc = _make_sample_doc()
        elem_types, tables = extract_table_and_element_metadata(doc)

        assert elem_types["e0001"] == ElementType.HEADER
        assert elem_types["e0005"] == ElementType.TABLE
        assert "Resource Allocations" in tables["e0005"]
        assert "Node ID | RAM (GB) | vCPU" in tables["e0005"]


class TestChunkRepresentations:
    """Tests for Baseline, HeadingEnriched, and FullStructural representations."""

    def test_baseline_representation_preserves_text_verbatim(self) -> None:
        rep = BaselineRepresentation()
        assert rep.name == "baseline"

        doc = _make_sample_doc()
        chunk = _make_sample_chunk()

        transformed = rep.transform(chunk, doc)
        assert transformed == chunk.text

    def test_heading_enriched_representation(self) -> None:
        rep = HeadingEnrichedRepresentation(separator=" > ")
        assert rep.name == "heading_enriched"

        doc = _make_sample_doc()
        chunk = _make_sample_chunk()

        transformed = rep.transform(chunk, doc)

        assert "[DOCUMENT]\nSystem Architecture" in transformed
        assert "[SECTION]\nSystem Architecture > Subsystem Components" in transformed
        assert f"[CONTENT]\n{chunk.text}" in transformed
        assert chunk.text in transformed

    def test_full_structural_representation_table_and_pages(self) -> None:
        rep = FullStructuralRepresentation(separator=" > ")
        assert rep.name == "full_structural"

        doc = _make_sample_doc()
        table_chunk = _make_sample_chunk(
            chunk_id="doc_sample_123_c0002",
            elem_ids=["e0005"],
            text="Table of node specifications and limits.",
        )

        transformed = rep.transform(table_chunk, doc)

        assert "[DOCUMENT]\nSystem Architecture" in transformed
        assert "[SECTION]\nSystem Architecture > Subsystem Components" in transformed
        assert "[CONTENT]\nTable of node specifications and limits." in transformed
        assert "[STRUCTURE]" in transformed
        assert "Type: table" in transformed
        assert "Page: 2" in transformed
        assert "Resource Allocations" in transformed
        assert "Node ID | RAM (GB) | vCPU" in transformed

    def test_representation_determinism(self) -> None:
        rep_b = HeadingEnrichedRepresentation()
        rep_c = FullStructuralRepresentation()

        doc = _make_sample_doc()
        chunk = _make_sample_chunk()

        # Invocations must yield bitwise identical output
        out_b1 = rep_b.transform(chunk, doc)
        out_b2 = rep_b.transform(chunk, doc)
        assert out_b1 == out_b2

        out_c1 = rep_c.transform(chunk, doc)
        out_c2 = rep_c.transform(chunk, doc)
        assert out_c1 == out_c2

    def test_missing_document_or_metadata(self) -> None:
        rep_b = HeadingEnrichedRepresentation()
        rep_c = FullStructuralRepresentation()

        chunk = _make_sample_chunk()

        # When document is None
        out_b = rep_b.transform(chunk, None)
        assert "[DOCUMENT]\nUnknown Document" in out_b
        assert "[SECTION]\nGeneral Content" in out_b
        assert f"[CONTENT]\n{chunk.text}" in out_b

        out_c = rep_c.transform(chunk, None)
        assert "[DOCUMENT]\nUnknown Document" in out_c
        assert "[SECTION]\nGeneral Content" in out_c
        assert f"[CONTENT]\n{chunk.text}" in out_c

    def test_factory_function(self) -> None:
        rep_a = create_representation("baseline")
        assert isinstance(rep_a, BaselineRepresentation)

        rep_b = create_representation("heading_enriched")
        assert isinstance(rep_b, HeadingEnrichedRepresentation)

        rep_c = create_representation("full_structural")
        assert isinstance(rep_c, FullStructuralRepresentation)
