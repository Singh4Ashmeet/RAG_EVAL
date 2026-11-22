"""Unit tests for Phase 18 element-centric and boundary-aware chunkers."""

from __future__ import annotations

import pytest

from core.chunking import (
    BoundaryAwareChunker,
    ElementAtomicChunker,
    ElementAwareChunker,
    ElementHeadingAttachedChunker,
)
from core.normalize.schema import (
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
    TableData,
)


@pytest.fixture
def multi_element_document() -> NormalizedDocument:
    """Fixture with headings, paragraphs, lists, and tables."""
    elements = [
        ExtractedElement(
            element_id="elem_h1",
            element_type=ElementType.HEADER,
            content="# Distributed Storage Specifications",
            order_index=0,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_p1",
            element_type=ElementType.TEXT,
            content="The cluster implements replicated NVMe storage pools across failure domains.",
            order_index=1,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_p2",
            element_type=ElementType.TEXT,
            content="Heartbeat intervals are tuned to 250 milliseconds with quorum leasing.",
            order_index=2,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_table",
            element_type=ElementType.TABLE,
            content="Storage nodes configuration table.",
            order_index=3,
            location=PageLocation(page_number=2),
            table_data=TableData(
                headers=["Node ID", "Capacity", "Port", "Status"],
                rows=[["node-01", "2TB", "8080", "Active"], ["node-02", "4TB", "8080", "Active"]],
                caption="Storage Node Allocations",
            ),
        ),
        ExtractedElement(
            element_id="elem_h2",
            element_type=ElementType.HEADER,
            content="## Failover Runbook",
            order_index=4,
            location=PageLocation(page_number=2),
        ),
        ExtractedElement(
            element_id="elem_l1",
            element_type=ElementType.LIST_ITEM,
            content="1. Check node heartbeat status in console.",
            order_index=5,
            location=PageLocation(page_number=2),
        ),
        ExtractedElement(
            element_id="elem_l2",
            element_type=ElementType.LIST_ITEM,
            content="2. Issue quorum failover command via CLI.",
            order_index=6,
            location=PageLocation(page_number=2),
        ),
    ]

    return NormalizedDocument(
        metadata=DocumentMetadata(
            document_id="doc_storage_test",
            filename="storage_spec.md",
        ),
        elements=elements,
        page_count=2,
    )


class TestElementCentricChunkers:
    """Tests for ElementAtomicChunker and ElementHeadingAttachedChunker."""

    def test_element_atomic_chunking(self, multi_element_document: NormalizedDocument) -> None:
        chunker = ElementAtomicChunker(max_chunk_size=1000)
        chunks = chunker.chunk(multi_element_document)

        # 7 non-empty elements -> exactly 7 atomic chunks
        assert len(chunks) == 7

        # Check deterministic IDs
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc_storage_test_ea{i:04d}"
            assert len(c.source_element_ids) == 1

        # Check element preservation
        assert chunks[0].text == "# Distributed Storage Specifications"
        assert chunks[3].text == "Storage nodes configuration table."
        assert chunks[3].source_element_ids == ["elem_table"]
        assert chunks[3].metadata["element_type"] == "table"

    def test_element_heading_attached_chunking(
        self, multi_element_document: NormalizedDocument
    ) -> None:
        chunker = ElementHeadingAttachedChunker(max_chunk_size=1200)
        chunks = chunker.chunk(multi_element_document)

        # 7 elements (2 headings + 5 body elements) -> 7 chunks
        assert len(chunks) == 7

        # Check deterministic IDs
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc_storage_test_eh{i:04d}"

        # Headings alone
        assert chunks[0].text == "# Distributed Storage Specifications"
        assert chunks[0].metadata.get("is_heading") is True

        # Body paragraph under H1 has H1 attached
        assert chunks[1].text.startswith("# Distributed Storage Specifications\n\n")
        assert "The cluster implements replicated NVMe" in chunks[1].text
        assert "elem_h1" in chunks[1].source_element_ids
        assert "elem_p1" in chunks[1].source_element_ids

        # Table under H1 has H1 attached
        assert chunks[3].text.startswith("# Distributed Storage Specifications\n\n")
        assert "elem_table" in chunks[3].source_element_ids

        # List under H2 has H2 attached
        assert chunks[5].text.startswith("## Failover Runbook\n\n")
        assert "elem_h2" in chunks[5].source_element_ids
        assert "elem_l1" in chunks[5].source_element_ids

    def test_oversized_element_splitting(self) -> None:
        long_content = "Word " * 400  # 2000 chars
        long_doc = NormalizedDocument(
            metadata=DocumentMetadata(document_id="doc_long", filename="long.txt"),
            elements=[
                ExtractedElement(
                    element_id="elem_long",
                    element_type=ElementType.TEXT,
                    content=long_content,
                    order_index=0,
                )
            ],
        )

        chunker = ElementAtomicChunker(max_chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(long_doc)

        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 500
            assert c.source_element_ids == ["elem_long"]
            assert c.document_id == "doc_long"


class TestBoundaryAwareChunker:
    """Tests for BoundaryAwareChunker with semantic boundary enforcement."""

    def test_boundary_aware_grouping(self, multi_element_document: NormalizedDocument) -> None:
        chunker = BoundaryAwareChunker(max_chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(multi_element_document)

        # Check deterministic IDs
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc_storage_test_bg{i:04d}"
            assert len(c.text) <= 500

        # Table isolation invariant: Table element must NEVER be merged with prose
        table_chunks = [c for c in chunks if "elem_table" in c.source_element_ids]
        assert len(table_chunks) == 1
        assert table_chunks[0].source_element_ids == ["elem_table"]
        assert table_chunks[0].text == "Storage nodes configuration table."

        # Heading isolation: Heading elem_h2 starts a fresh chunk
        h2_chunks = [c for c in chunks if "elem_h2" in c.source_element_ids]
        assert len(h2_chunks) >= 1
        # Heading must not contain preceding table content
        assert "Storage nodes" not in h2_chunks[0].text

    def test_compatible_elements_merge_within_limit(self) -> None:
        doc = NormalizedDocument(
            metadata=DocumentMetadata(document_id="doc_compat", filename="compat.txt"),
            elements=[
                ExtractedElement(
                    element_id="e1",
                    element_type=ElementType.TEXT,
                    content="First short paragraph.",
                    order_index=0,
                ),
                ExtractedElement(
                    element_id="e2",
                    element_type=ElementType.TEXT,
                    content="Second short paragraph.",
                    order_index=1,
                ),
            ],
        )

        chunker = BoundaryAwareChunker(max_chunk_size=500)
        chunks = chunker.chunk(doc)

        # Two short compatible paragraphs under 500 chars pack together
        assert len(chunks) == 1
        assert chunks[0].source_element_ids == ["e1", "e2"]
        assert "First short paragraph.\n\nSecond short paragraph." in chunks[0].text


class TestProductionBaselineInvariance:
    """Verify that existing production ElementAwareChunker is strictly unmodified."""

    def test_baseline_chunker_unchanged(self, multi_element_document: NormalizedDocument) -> None:
        chunker = ElementAwareChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk(multi_element_document)

        # Baseline packs greedily to 1000 chars
        assert len(chunks) >= 1
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc_storage_test_c{i:04d}"
            assert c.metadata["chunker"] == "ElementAwareChunker"
            assert c.metadata["chunk_size"] == 1000
            assert c.metadata["chunk_overlap"] == 200

    def test_deterministic_repeated_builds(
        self, multi_element_document: NormalizedDocument
    ) -> None:
        chunkers = [
            ElementAtomicChunker(1000),
            ElementHeadingAttachedChunker(1200),
            BoundaryAwareChunker(500),
            ElementAwareChunker(1000, 200),
        ]

        for chunker in chunkers:
            c1 = chunker.chunk(multi_element_document)
            c2 = chunker.chunk(multi_element_document)
            assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
            assert [c.text for c in c1] == [c.text for c in c2]
            assert [c.source_element_ids for c in c1] == [c.source_element_ids for c in c2]
