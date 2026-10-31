"""Unit tests for M2.1 Step 3: ElementAwareChunker and provenance preservation."""

import pytest

from core.chunking import DocumentChunker, ElementAwareChunker
from core.chunking.chunker import _Fragment
from core.normalize.schema import (
    BoundingBox,
    Chunk,
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)


@pytest.fixture
def base_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc_test_12345678",
        filename="report.txt",
        content_sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        byte_size=1024,
    )


class TestElementAwareChunkerProtocolAndConfig:
    """Verify protocol conformance and constructor parameter validation."""

    def test_implements_document_chunker_protocol(self) -> None:
        chunker = ElementAwareChunker()
        assert isinstance(chunker, DocumentChunker)

    def test_invalid_chunk_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be strictly positive"):
            ElementAwareChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be strictly positive"):
            ElementAwareChunker(chunk_size=-10)

    def test_invalid_chunk_overlap_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            ElementAwareChunker(chunk_size=100, chunk_overlap=-1)

        with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
            ElementAwareChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
            ElementAwareChunker(chunk_size=100, chunk_overlap=150)


class TestElementAwareChunkerBehavior:
    """Verify chunk boundaries, provenance propagation, and token estimation."""

    def test_empty_document_produces_no_chunks(self, base_metadata: DocumentMetadata) -> None:
        doc = NormalizedDocument(metadata=base_metadata, elements=[])
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_whitespace_only_elements_produce_no_chunks(
        self, base_metadata: DocumentMetadata
    ) -> None:
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="   \n\t  ",
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="\n\n",
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_single_element_smaller_than_chunk_limit(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="elem_001",
                element_type=ElementType.TEXT,
                content="This is a short single element.",
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id == f"{base_metadata.document_id}_c0000"
        assert chunk.chunk_index == 0
        assert chunk.document_id == base_metadata.document_id
        assert chunk.text == "This is a short single element."
        assert chunk.source_element_ids == ["elem_001"]
        assert chunk.locations == []

    def test_multiple_elements_fitting_into_one_chunk(
        self, base_metadata: DocumentMetadata
    ) -> None:
        elements = [
            ExtractedElement(
                element_id="elem_h1",
                element_type=ElementType.HEADER,
                content="Overview",
                order_index=0,
            ),
            ExtractedElement(
                element_id="elem_p1",
                element_type=ElementType.TEXT,
                content="First introductory paragraph.",
                order_index=1,
            ),
            ExtractedElement(
                element_id="elem_p2",
                element_type=ElementType.TEXT,
                content="Second follow-up paragraph.",
                order_index=2,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.source_element_ids == ["elem_h1", "elem_p1", "elem_p2"]
        expected_text = "Overview\n\nFirst introductory paragraph.\n\nSecond follow-up paragraph."
        assert chunk.text == expected_text

    def test_multiple_chunks_from_longer_document(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"elem_{i}",
                element_type=ElementType.TEXT,
                content=f"Paragraph block {i} with some descriptive sentences.",
                order_index=i,
            )
            for i in range(10)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # Small chunk size forces splitting across multiple chunks
        chunker = ElementAwareChunker(chunk_size=120, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.chunk_id == f"{base_metadata.document_id}_c{idx:04d}"
            assert chunk.document_id == base_metadata.document_id
            assert len(chunk.text) <= 120

    def test_individual_element_exceeding_limit(self, base_metadata: DocumentMetadata) -> None:
        long_text = "Word " * 100  # 500 characters
        elements = [
            ExtractedElement(
                element_id="elem_oversized",
                element_type=ElementType.TEXT,
                content=long_text,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=40)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk.source_element_ids == ["elem_oversized"]
            assert len(chunk.text) <= 150

    def test_overlap_behavior_preserves_context(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="Short block A.",  # 14 chars
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="Short block B.",  # 14 chars
                order_index=1,
            ),
            ExtractedElement(
                element_id="e2",
                element_type=ElementType.TEXT,
                content="Short block C.",  # 14 chars
                order_index=2,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # chunk_size=35 allows [e0, e1] (14 + 2 + 14 = 30) but not e2 (30 + 2 + 14 = 46 > 35)
        # Single overlap architecture: packing is strictly forward-only, no packing-level overlap.
        chunker = ElementAwareChunker(chunk_size=35, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].source_element_ids == ["e0", "e1"]
        assert chunks[1].source_element_ids == ["e2"]
        assert "Short block B." in chunks[0].text
        assert "Short block C." in chunks[1].text

    def test_correct_sequential_chunk_indices(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Content {i} " * 10,
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        for expected_idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected_idx

    def test_deterministic_ids_across_repeated_runs(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Deterministic paragraph {i}.",
                order_index=i,
            )
            for i in range(4)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=70, chunk_overlap=20)

        run1 = chunker.chunk(doc)
        run2 = chunker.chunk(doc)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2, strict=True):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.chunk_index == c2.chunk_index
            assert c1.source_element_ids == c2.source_element_ids

    def test_unique_ids_for_distinct_chunks(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Paragraph {i}.",
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=30, chunk_overlap=10)
        chunks = chunker.chunk(doc)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_location_propagation_and_deduplication(self, base_metadata: DocumentMetadata) -> None:
        loc1 = PageLocation(
            page_number=1,
            bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.3),
        )
        loc2 = PageLocation(
            page_number=1,
            bbox=BoundingBox(x0=0.1, y0=0.4, x1=0.9, y1=0.6),
        )
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.HEADER,
                content="Header with location",
                order_index=0,
                location=loc1,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="Text with location",
                order_index=1,
                location=loc2,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].locations == [loc1, loc2]

    def test_no_fabricated_locations_when_elements_lack_location(
        self, base_metadata: DocumentMetadata
    ) -> None:
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="No spatial info available",
                order_index=0,
                location=None,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].locations == []

    def test_source_element_contributes_to_multiple_chunks(
        self, base_metadata: DocumentMetadata
    ) -> None:
        # A long element that spans across chunks
        elements = [
            ExtractedElement(
                element_id="shared_elem",
                element_type=ElementType.TEXT,
                content="Sentence that is repeated to make it sufficiently long. " * 10,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert "shared_elem" in chunk.source_element_ids

    def test_estimated_token_counts(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e_short",
                element_type=ElementType.TEXT,
                content="Hi",  # 2 characters
                order_index=0,
            ),
            ExtractedElement(
                element_id="e_longer",
                element_type=ElementType.TEXT,
                content="A" * 100,  # 100 characters
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        # Short chunk must NOT round down to 0
        assert chunks[0].token_count is not None
        assert chunks[0].token_count >= 1

        # Long chunk of 100 characters should approximate chars // 4
        assert chunks[1].token_count is not None
        assert chunks[1].token_count == 25  # (100 + 3) // 4 = 25

    def test_canonical_chunk_schema_validity(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Schema validity test block {i}.",
                order_index=i,
            )
            for i in range(3)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=80, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            # Re-validate with Pydantic v2
            dumped = chunk.model_dump()
            validated = Chunk.model_validate(dumped)
            assert validated == chunk

    # --- Regression tests for M2.1 Step 3 fixes ---

    def test_no_overlap_only_chunk_when_next_unit_exceeds_limit(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Overlap should not create a chunk containing only retained context.

        With chunk_size=100, chunk_overlap=20:
        - Element 0: 50 chars
        - Element 1: 20 chars (fits in overlap)
        - Element 2: 90 chars (cannot fit with element 1: 20 + 2 + 90 = 112 > 100)

        Before fix: produced 3 chunks [e0,e1], [e1], [e2] where middle chunk is overlap-only.
        After fix: should produce 2 chunks [e0,e1], [e2] (element 1 not repeated as solo chunk).
        """
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="A" * 50,
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="B" * 20,
                order_index=1,
            ),
            ExtractedElement(
                element_id="e2",
                element_type=ElementType.TEXT,
                content="C" * 90,
                order_index=2,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        # Should not create an overlap-only chunk for e1
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}: {[c.text for c in chunks]}"

        # First chunk contains e0 and e1
        assert chunks[0].source_element_ids == ["e0", "e1"]
        assert "A" * 50 in chunks[0].text
        assert "B" * 20 in chunks[0].text
        assert len(chunks[0].text) <= 100

        # Second chunk contains e2 (e1 not repeated as overlap-only)
        assert chunks[1].source_element_ids == ["e2"]
        assert "C" * 90 in chunks[1].text
        assert len(chunks[1].text) <= 100

        # All chunks respect max size
        for chunk in chunks:
            assert len(chunk.text) <= 100

        # Sequential indices
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_forward_only_packing_does_not_repeat_whole_elements(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Whole elements are packed forward-only without packing-level overlap duplication.

        With chunk_size=35, chunk_overlap=20:
        - e0: 14 chars, e1: 14 chars, e2: 14 chars
        - First chunk: [e0, e1] = 30 chars (fits)
        - Next chunk starts fresh with e2 = 14 chars (e1 is NOT repeated)
        """
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="A" * 14,
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="B" * 14,
                order_index=1,
            ),
            ExtractedElement(
                element_id="e2",
                element_type=ElementType.TEXT,
                content="C" * 14,
                order_index=2,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=35, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].source_element_ids == ["e0", "e1"]
        assert chunks[1].source_element_ids == ["e2"]
        assert "B" * 14 in chunks[0].text
        assert "C" * 14 in chunks[1].text

    def test_whitespace_preserved_in_oversized_element_split(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Long whitespace runs inside an oversized element must not be lost.

        Element content: "hello" + (900 spaces) + "world" = 910 chars.
        With chunk_size=500, chunk_overlap=100, this element is split.
        The 900-space run should be preserved across pieces.
        """
        long_spaces = " " * 900
        content = f"hello{long_spaces}world"  # 5 + 900 + 5 = 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_ws",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        # Should complete without exception
        assert len(chunks) >= 2

        # All chunks valid
        for chunk in chunks:
            assert len(chunk.text) <= 500
            # Schema validation
            dumped = chunk.model_dump()
            validated = Chunk.model_validate(dumped)
            assert validated == chunk

        # Reconstruct text from chunks (accounting for overlap)
        # Since it's a single element split across chunks, concatenate with overlap awareness
        full_text = "".join(chunk.text for chunk in chunks)
        # The original content should be preserved in order
        assert "hello" in full_text
        assert "world" in full_text
        # Find positions
        hello_pos = full_text.index("hello")
        world_pos = full_text.index("world")
        assert hello_pos < world_pos, "hello must appear before world"

        # The whitespace run should not be completely lost
        # (exact count may vary due to overlap, but substantial whitespace should remain)
        between = full_text[hello_pos + 5 : world_pos]
        assert len(between) > 100, (
            f"Expected substantial whitespace preserved, got {len(between)} chars"
        )

        # Source element ID preserved
        for chunk in chunks:
            assert chunk.source_element_ids == ["elem_ws"]

    def test_oversized_element_with_long_whitespace_run_exceeds_chunk_size(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Test whitespace preservation when element is large enough to force splitting.

        Uses chunk_size=200 to ensure the 910-char element is split into multiple pieces.
        The whitespace skip in overlap logic consumes some whitespace, but substantial
        whitespace should still be preserved in the first chunk.
        """
        long_spaces = " " * 900
        content = f"hello{long_spaces}world"  # 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_ws_large",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        # With whitespace skip in overlap, we get fewer chunks but whitespace is preserved
        assert len(chunks) >= 2

        for chunk in chunks:
            assert len(chunk.text) <= 200
            dumped = chunk.model_dump()
            validated = Chunk.model_validate(dumped)
            assert validated == chunk

        full_text = "".join(chunk.text for chunk in chunks)
        hello_pos = full_text.index("hello")
        world_pos = full_text.index("world")
        assert hello_pos < world_pos

        between = full_text[hello_pos + 5 : world_pos]
        # First chunk holds "hello" + up to 195 spaces (200 - 5)
        # Overlap may duplicate some, but whitespace skip consumes overlap region
        # Expect at least half the chunk_size in preserved whitespace
        assert len(between) > 90, (
            f"Expected substantial whitespace preserved, got {len(between)} chars"
        )

        for chunk in chunks:
            assert chunk.source_element_ids == ["elem_ws_large"]

    # --- Boundary tests ---

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap",
        [
            (100, 0),
            (100, 99),  # chunk_overlap = chunk_size - 1
        ],
    )
    def test_overlap_boundary_configs(
        self, base_metadata: DocumentMetadata, chunk_size: int, chunk_overlap: int
    ) -> None:
        """Test chunker with boundary overlap configurations."""
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content="X" * 30,
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert len(chunk.text) <= chunk_size
            assert chunk.source_element_ids

    def test_element_exactly_chunk_size(self, base_metadata: DocumentMetadata) -> None:
        """Element exactly at chunk_size boundary should fit in one chunk."""
        content = "X" * 100  # exactly chunk_size
        elements = [
            ExtractedElement(
                element_id="e_exact",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == content
        assert len(chunks[0].text) == 100
        assert chunks[0].source_element_ids == ["e_exact"]

    def test_element_one_over_chunk_size(self, base_metadata: DocumentMetadata) -> None:
        """Element one character over chunk_size should be split."""
        content = "X" * 101  # chunk_size + 1
        elements = [
            ExtractedElement(
                element_id="e_over",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk.text) <= 100
            assert chunk.source_element_ids == ["e_over"]

    def test_long_unbroken_string_no_whitespace(self, base_metadata: DocumentMetadata) -> None:
        """Long string with no whitespace should be split deterministically."""
        content = "A" * 1000  # no spaces or newlines
        elements = [
            ExtractedElement(
                element_id="e_unbroken",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk.text) <= 200
            assert chunk.source_element_ids == ["e_unbroken"]

        # Deterministic: re-run produces same chunks
        chunks2 = chunker.chunk(doc)
        assert len(chunks) == len(chunks2)
        for c1, c2 in zip(chunks, chunks2, strict=True):
            assert c1.text == c2.text
            assert c1.chunk_id == c2.chunk_id

    def test_multiple_oversized_elements_in_sequence(self, base_metadata: DocumentMetadata) -> None:
        """Multiple consecutive oversized elements should all be split and packed."""
        elements = [
            ExtractedElement(
                element_id=f"e_big_{i}",
                element_type=ElementType.TEXT,
                content="X" * 500,
                order_index=i,
            )
            for i in range(3)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        # Each 500-char element splits into ~3 pieces (500/(200-50) ≈ 3.3)
        # 3 elements * 3-4 pieces each ≈ 9-12 chunks
        assert len(chunks) >= 9

        for chunk in chunks:
            assert len(chunk.text) <= 200

        # All source element IDs should appear in order across chunks
        seen_ids: list[str] = []
        for chunk in chunks:
            for eid in chunk.source_element_ids:
                if eid not in seen_ids:
                    seen_ids.append(eid)
        assert seen_ids == ["e_big_0", "e_big_1", "e_big_2"]

    def test_unicode_and_crlf_content(self, base_metadata: DocumentMetadata) -> None:
        """Content with Unicode and CRLF line endings should be handled correctly."""
        content = "Hello \u00e9 world\r\nNew line\tTab\u2603"
        elements = [
            ExtractedElement(
                element_id="e_unicode",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == content
        assert len(chunk.text) <= 100
        assert chunk.source_element_ids == ["e_unicode"]


class TestPublicExports:
    """Verify core.chunking public symbols."""

    def test_core_chunking_exports(self) -> None:
        from core.chunking import DocumentChunker, ElementAwareChunker

        assert DocumentChunker is not None
        assert ElementAwareChunker is not None


# ===========================================================================
# M2.1 Step 3 Comprehensive Invariant and Regression Test Suite
# ===========================================================================


def _assert_splitter_invariants(
    fragments: list[_Fragment],
    original_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Assert all mathematical invariants for fragments generated by _split_oversized."""
    assert len(fragments) > 0, "Splitter must produce at least one fragment"
    N = len(original_text)

    # I1: First fragment starts at 0
    assert fragments[0].start == 0, f"Expected start 0, got {fragments[0].start}"

    # I2: Final fragment ends at N
    assert fragments[-1].end == N, f"Expected end {N}, got {fragments[-1].end}"

    # I3: Nonempty valid bounds
    for idx, f in enumerate(fragments):
        assert 0 <= f.start < f.end <= N, f"Invalid interval [{f.start}, {f.end}) for N={N}"
        # I6: Strict size limit
        assert f.end - f.start <= chunk_size, (
            f"Fragment {idx} length {f.end - f.start} exceeds chunk_size {chunk_size}"
        )
        # Source-slice exact fidelity
        assert f.text == original_text[f.start : f.end], (
            f"Fragment text does not match source slice at [{f.start}, {f.end})"
        )

    # Contiguity, monotonic advance, and bounded overlap
    covered_offsets: set[int] = set()
    for idx in range(len(fragments)):
        covered_offsets.update(range(fragments[idx].start, fragments[idx].end))

        if idx < len(fragments) - 1:
            curr = fragments[idx]
            nxt = fragments[idx + 1]
            # I4: Starts strictly increase
            assert nxt.start > curr.start, (
                f"Starts not strictly increasing: {curr.start} >= {nxt.start}"
            )
            # I5: No gaps
            assert nxt.start <= curr.end, f"Gap between fragments: {curr.end} < {nxt.start}"
            # I7: Bounded overlap
            overlap = curr.end - nxt.start
            assert 0 <= overlap <= chunk_overlap, (
                f"Overlap {overlap} outside bounds [0, {chunk_overlap}]"
            )

    # Complete interval coverage
    assert covered_offsets == set(range(N)), "Splitter failed to cover all source character offsets"


class TestSplitterInvariantEdgeCases:
    """Category B: Direct tests of the offset-based splitter and snapping invariants."""

    def test_unbroken_string_invariants(self) -> None:
        text = "A" * 1000
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 200, 50)
        # Unbroken string has no word boundaries; every overlap should be exactly 50
        for i in range(len(fragments) - 1):
            assert fragments[i].end - fragments[i + 1].start == 50

    def test_text_with_spaces_and_newlines_invariants(self) -> None:
        text = (
            "Sentence one with words.\nSentence two has more detail.\nSentence three continues."
        ) * 10
        chunker = ElementAwareChunker(chunk_size=120, chunk_overlap=30)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 120, 30)

    def test_tabs_and_crlf_invariants(self) -> None:
        text = "Item 1\tDescription 1\r\nItem 2\tDescription 2\r\nItem 3\tDescription 3\r\n" * 15
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=40)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 150, 40)

    def test_zero_overlap_invariants(self) -> None:
        text = "A" * 500
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=0)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 0)
        for i in range(len(fragments) - 1):
            assert fragments[i].end == fragments[i + 1].start
        assert "".join(f.text for f in fragments) == text

    def test_max_overlap_invariants(self) -> None:
        text = "B" * 200
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=99)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 99)
        # S = 1, so exactly 101 fragments
        assert len(fragments) == 101
        for i in range(len(fragments) - 1):
            assert fragments[i + 1].start == fragments[i].start + 1
            assert fragments[i].end - fragments[i + 1].start == 99

    def test_boundary_length_c_plus_one(self) -> None:
        text = "X" * 101
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 20)
        assert len(fragments) == 2
        assert fragments[0].start == 0 and fragments[0].end == 100
        assert fragments[1].start == 80 and fragments[1].end == 101

    def test_termination_extreme_configs(self) -> None:
        text = "Word " * 2000  # 10,000 characters
        chunker = ElementAwareChunker(chunk_size=50, chunk_overlap=49)  # S = 1
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 50, 49)

    def test_deterministic_boundaries(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 30
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        run1 = chunker._split_oversized("elem_1", text, None)
        run2 = chunker._split_oversized("elem_1", text, None)
        assert len(run1) == len(run2)
        for f1, f2 in zip(run1, run2, strict=True):
            assert f1.start == f2.start
            assert f1.end == f2.end
            assert f1.text == f2.text


class TestPackingRuleEdgeCases:
    """Category C: Packing behavior, separation, and single-overlap enforcement."""

    def test_multiple_small_elements_separated(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Element {i}",
                order_index=i,
            )
            for i in range(4)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        expected = "Element 0\n\nElement 1\n\nElement 2\n\nElement 3"
        assert chunks[0].text == expected

    def test_same_element_fragments_no_separator(self, base_metadata: DocumentMetadata) -> None:
        # An oversized element split across chunks
        elements = [
            ExtractedElement(
                element_id="e_same",
                element_type=ElementType.TEXT,
                content="Word " * 100,  # 500 chars
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "\n\n" not in chunk.text

    def test_candidate_rejected_when_exceeding_c(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="A" * 60,
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="B" * 60,
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # chunk_size=100 cannot hold e0 (60) + "\n\n" (2) + e1 (60) = 122 > 100
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].text == "A" * 60
        assert chunks[1].text == "B" * 60

    def test_no_packing_level_overlap_whole_elements(self, base_metadata: DocumentMetadata) -> None:
        """Whole elements must NOT be repeated into subsequent chunks."""
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content="Content " * 5,  # 40 chars
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # chunk_size=90 holds e0 + "\n\n" + e1 (40 + 2 + 40 = 82 <= 90)
        chunker = ElementAwareChunker(chunk_size=90, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        # In forward-only packing, each whole element appears in exactly one chunk
        seen_elements: list[str] = []
        for chunk in chunks:
            for eid in chunk.source_element_ids:
                assert eid not in seen_elements, f"Element {eid} repeated across chunks"
                seen_elements.append(eid)
        assert seen_elements == [f"e{i}" for i in range(5)]

    def test_overlapping_fragments_deduplicated_within_chunk(self) -> None:
        """When multiple fragments of the same element pack into one chunk,
        no character is duplicated.
        """
        f1 = _Fragment(
            element_id="e1",
            original_text="0123456789ABCDEF",
            start=0,
            end=10,
            location=None,
        )
        f2 = _Fragment(
            element_id="e1",
            original_text="0123456789ABCDEF",
            start=8,
            end=16,
            location=None,
        )
        assembled = ElementAwareChunker._assemble_chunk_text([f1, f2], separator="\n\n")
        assert assembled == "0123456789ABCDEF"
        assert len(assembled) == 16
        add_len = ElementAwareChunker._measure_additional_length([f1], f2, separator="\n\n")
        assert add_len == 6  # 16 - 10 = 6 new characters

    def test_chunk_ids_and_indices_deterministic(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Deterministic paragraph {i} text content.",
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=80, chunk_overlap=20)
        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)

        assert len(chunks1) == len(chunks2)
        for idx, (c1, c2) in enumerate(zip(chunks1, chunks2, strict=True)):
            assert c1.chunk_index == idx
            assert c1.chunk_id == f"{base_metadata.document_id}_c{idx:04d}"
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text


class TestWhitespaceOmissionEdgeCases:
    """Category D: Whitespace preservation, bounded omission, and metadata audit."""

    def test_empty_and_whitespace_only_elements(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e_empty",
                element_type=ElementType.TEXT,
                content="",
                order_index=0,
            ),
            ExtractedElement(
                element_id="e_ws",
                element_type=ElementType.TEXT,
                content="   \n\t\r   ",
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_long_interior_whitespace_regression_500_100(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Regression: 'hello' + ' '*900 + 'world' with C=500, O=100.

        Guarantees:
        - Both 'hello' and 'world' preserved in order.
        - Exactly 2 chunks emitted.
        - Chunk 0: 'hello' + 495 spaces (len 500).
        - Chunk 1: 105 spaces + 'world' (len 110).
        - Omitted interval [500, 800) length 300 recorded in Chunk 1 metadata.
        - Total characters: 500 (Chunk 0) + 300 (omitted) + 110 (Chunk 1) = 910.
        """
        content = "hello" + (" " * 900) + "world"  # 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_reg_500",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert "hello" in chunks[0].text
        assert "world" in chunks[1].text
        for c in chunks:
            assert len(c.text) <= 500
            assert len(c.text.strip()) > 0

        # Chunk 0 has no omissions
        assert "omitted_whitespace_intervals" not in chunks[0].metadata

        # Chunk 1 records the exact omitted interval [500, 800)
        omitted = chunks[1].metadata.get("omitted_whitespace_intervals")
        assert omitted is not None
        assert len(omitted) == 1
        entry = omitted[0]
        assert entry["element_id"] == "elem_reg_500"
        assert entry["start"] == 500
        assert entry["end"] == 800
        assert entry["length"] == 300

        # Every single source character is accounted for
        assert len(chunks[0].text) + entry["length"] + len(chunks[1].text) == 910

    def test_long_interior_whitespace_regression_200_50(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Regression: 'hello' + ' '*900 + 'world' with C=200, O=50.

        Guarantees:
        - Both 'hello' and 'world' preserved in order.
        - Exactly 2 chunks emitted.
        - Chunk 0 has len 200. Chunk 1 has len 160.
        - Omitted interval [200, 750) length 550 recorded in Chunk 1 metadata.
        - Total characters: 200 + 550 + 160 = 910.
        """
        content = "hello" + (" " * 900) + "world"  # 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_reg_200",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert "hello" in chunks[0].text
        assert "world" in chunks[1].text
        for c in chunks:
            assert len(c.text) <= 200
            assert len(c.text.strip()) > 0

        omitted = chunks[1].metadata.get("omitted_whitespace_intervals")
        assert omitted is not None
        assert len(omitted) == 1
        entry = omitted[0]
        assert entry["element_id"] == "elem_reg_200"
        assert entry["start"] == 200
        assert entry["end"] == 750
        assert entry["length"] == 550

        assert len(chunks[0].text) + entry["length"] + len(chunks[1].text) == 910

    def test_retainable_whitespace_only_fragments_not_omitted(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """When an interior whitespace run fits with adjacent text, it is NOT omitted."""
        # Content: 40 chars 'A' + 30 spaces + 40 chars 'B' = 110 chars
        content = ("A" * 40) + (" " * 30) + ("B" * 40)
        elements = [
            ExtractedElement(
                element_id="e_retain",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == content
        assert "omitted_whitespace_intervals" not in chunks[0].metadata

    def test_no_non_whitespace_character_ever_omitted(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """All non-whitespace characters must be present across emitted chunks."""
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        content = (" " * 400).join(words)
        elements = [
            ExtractedElement(
                element_id="e_words",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        full_output = " ".join(c.text for c in chunks)
        for w in words:
            assert w in full_output, f"Word {w} was lost!"

    def test_no_whitespace_only_chunks_emitted(self, base_metadata: DocumentMetadata) -> None:
        """Every emitted chunk must satisfy text.strip() != ''."""
        content = "start" + (" " * 1500) + "middle" + (" " * 1500) + "end"
        elements = [
            ExtractedElement(
                element_id="e_multi_ws",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=300, chunk_overlap=60)
        chunks = chunker.chunk(doc)

        for c in chunks:
            assert len(c.text.strip()) > 0, "Emitted an invalid whitespace-only chunk!"


class TestGeneralPropertyEdgeCases:
    """Category E: Property tests validating all 12 chunker contract invariants."""

    @pytest.mark.parametrize(
        "c_size,c_overlap",
        [
            (50, 10),
            (100, 0),
            (100, 25),
            (200, 50),
            (500, 150),
        ],
    )
    def test_all_chunks_satisfy_invariants(
        self, base_metadata: DocumentMetadata, c_size: int, c_overlap: int
    ) -> None:
        elements = [
            ExtractedElement(
                element_id=f"elem_{i}",
                element_type=ElementType.TEXT,
                content=(
                    f"Heading {i}\n" + ("Text content sentence with various words. " * (i * 3 + 1))
                ),
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=c_size, chunk_overlap=c_overlap)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            # Invariant 1: 0 < len(chunk.text) <= chunk_size
            assert 0 < len(chunk.text) <= c_size
            # Invariant 2: chunk.text contains non-whitespace
            assert len(chunk.text.strip()) > 0
            # Invariant 3: Pydantic v2 validation passes
            assert Chunk.model_validate(chunk.model_dump()) == chunk
            # Invariant 4: Sequential index
            assert chunk.chunk_index == idx
            # Invariant 5: Deterministic chunk ID
            assert chunk.chunk_id == f"{base_metadata.document_id}_c{idx:04d}"

        # Invariant 6: Element order preserved
        seen_order: list[str] = []
        for chunk in chunks:
            for eid in chunk.source_element_ids:
                if not seen_order or seen_order[-1] != eid:
                    seen_order.append(eid)
        expected_order = [f"elem_{i}" for i in range(5)]
        assert seen_order == expected_order


# ===========================================================================
# M2.1 Step 3 Comprehensive Invariant and Regression Test Suite
# ===========================================================================


def _assert_splitter_invariants(
    fragments: list[_Fragment],
    original_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Assert all mathematical invariants for fragments generated by _split_oversized."""
    assert len(fragments) > 0, "Splitter must produce at least one fragment"
    N = len(original_text)

    # I1: First fragment starts at 0
    assert fragments[0].start == 0, f"Expected start 0, got {fragments[0].start}"

    # I2: Final fragment ends at N
    assert fragments[-1].end == N, f"Expected end {N}, got {fragments[-1].end}"

    # I3: Nonempty valid bounds
    for idx, f in enumerate(fragments):
        assert 0 <= f.start < f.end <= N, f"Invalid interval [{f.start}, {f.end}) for N={N}"
        # I6: Strict size limit
        assert f.end - f.start <= chunk_size, (
            f"Fragment {idx} length {f.end - f.start} exceeds chunk_size {chunk_size}"
        )
        # Source-slice exact fidelity
        assert f.text == original_text[f.start : f.end], (
            f"Fragment text does not match source slice at [{f.start}, {f.end})"
        )

    # Contiguity, monotonic advance, and bounded overlap
    covered_offsets: set[int] = set()
    for idx in range(len(fragments)):
        covered_offsets.update(range(fragments[idx].start, fragments[idx].end))

        if idx < len(fragments) - 1:
            curr = fragments[idx]
            nxt = fragments[idx + 1]
            # I4: Starts strictly increase
            assert nxt.start > curr.start, (
                f"Starts not strictly increasing: {curr.start} >= {nxt.start}"
            )
            # I5: No gaps
            assert nxt.start <= curr.end, f"Gap between fragments: {curr.end} < {nxt.start}"
            # I7: Bounded overlap
            overlap = curr.end - nxt.start
            assert 0 <= overlap <= chunk_overlap, (
                f"Overlap {overlap} outside bounds [0, {chunk_overlap}]"
            )

    # Complete interval coverage
    assert covered_offsets == set(range(N)), "Splitter failed to cover all source character offsets"


class TestSplitterInvariants:
    """Category B: Direct tests of the offset-based splitter and snapping invariants."""

    def test_unbroken_string_invariants(self) -> None:
        text = "A" * 1000
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 200, 50)
        # Unbroken string has no word boundaries; every overlap should be exactly 50
        for i in range(len(fragments) - 1):
            assert fragments[i].end - fragments[i + 1].start == 50

    def test_text_with_spaces_and_newlines_invariants(self) -> None:
        text = (
            "Sentence one with words.\nSentence two has more detail.\nSentence three continues."
        ) * 10
        chunker = ElementAwareChunker(chunk_size=120, chunk_overlap=30)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 120, 30)

    def test_tabs_and_crlf_invariants(self) -> None:
        text = "Item 1\tDescription 1\r\nItem 2\tDescription 2\r\nItem 3\tDescription 3\r\n" * 15
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=40)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 150, 40)

    def test_zero_overlap_invariants(self) -> None:
        text = "A" * 500
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=0)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 0)
        for i in range(len(fragments) - 1):
            assert fragments[i].end == fragments[i + 1].start
        assert "".join(f.text for f in fragments) == text

    def test_max_overlap_invariants(self) -> None:
        text = "B" * 200
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=99)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 99)
        # S = 1, so exactly 101 fragments
        assert len(fragments) == 101
        for i in range(len(fragments) - 1):
            assert fragments[i + 1].start == fragments[i].start + 1
            assert fragments[i].end - fragments[i + 1].start == 99

    def test_boundary_length_c_plus_one(self) -> None:
        text = "X" * 101
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 100, 20)
        assert len(fragments) == 2
        assert fragments[0].start == 0 and fragments[0].end == 100
        assert fragments[1].start == 80 and fragments[1].end == 101

    def test_termination_extreme_configs(self) -> None:
        text = "Word " * 2000  # 10,000 characters
        chunker = ElementAwareChunker(chunk_size=50, chunk_overlap=49)  # S = 1
        fragments = chunker._split_oversized("elem_1", text, None)
        _assert_splitter_invariants(fragments, text, 50, 49)

    def test_deterministic_boundaries(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 30
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        run1 = chunker._split_oversized("elem_1", text, None)
        run2 = chunker._split_oversized("elem_1", text, None)
        assert len(run1) == len(run2)
        for f1, f2 in zip(run1, run2, strict=True):
            assert f1.start == f2.start
            assert f1.end == f2.end
            assert f1.text == f2.text


class TestPackingRules:
    """Category C: Packing behavior, separation, and single-overlap enforcement."""

    def test_multiple_small_elements_separated(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Element {i}",
                order_index=i,
            )
            for i in range(4)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        expected = "Element 0\n\nElement 1\n\nElement 2\n\nElement 3"
        assert chunks[0].text == expected

    def test_same_element_fragments_no_separator(self, base_metadata: DocumentMetadata) -> None:
        # An oversized element split across chunks
        elements = [
            ExtractedElement(
                element_id="e_same",
                element_type=ElementType.TEXT,
                content="Word " * 100,  # 500 chars
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "\n\n" not in chunk.text

    def test_candidate_rejected_when_exceeding_c(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e0",
                element_type=ElementType.TEXT,
                content="A" * 60,
                order_index=0,
            ),
            ExtractedElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                content="B" * 60,
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # chunk_size=100 cannot hold e0 (60) + "\n\n" (2) + e1 (60) = 122 > 100
        chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].text == "A" * 60
        assert chunks[1].text == "B" * 60

    def test_no_packing_level_overlap_whole_elements(self, base_metadata: DocumentMetadata) -> None:
        """Whole elements must NOT be repeated into subsequent chunks."""
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content="Content " * 5,  # 40 chars
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        # chunk_size=90 holds e0 + "\n\n" + e1 (40 + 2 + 40 = 82 <= 90)
        chunker = ElementAwareChunker(chunk_size=90, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        # In forward-only packing, each whole element appears in exactly one chunk
        seen_elements: list[str] = []
        for chunk in chunks:
            for eid in chunk.source_element_ids:
                assert eid not in seen_elements, f"Element {eid} repeated across chunks"
                seen_elements.append(eid)
        assert seen_elements == [f"e{i}" for i in range(5)]

    def test_overlapping_fragments_deduplicated_within_chunk(self) -> None:
        """When multiple fragments of the same element pack into one chunk,
        no character is duplicated.
        """
        f1 = _Fragment(
            element_id="e1",
            original_text="0123456789ABCDEF",
            start=0,
            end=10,
            location=None,
        )
        f2 = _Fragment(
            element_id="e1",
            original_text="0123456789ABCDEF",
            start=8,
            end=16,
            location=None,
        )
        assembled = ElementAwareChunker._assemble_chunk_text([f1, f2], separator="\n\n")
        assert assembled == "0123456789ABCDEF"
        assert len(assembled) == 16
        add_len = ElementAwareChunker._measure_additional_length([f1], f2, separator="\n\n")
        assert add_len == 6  # 16 - 10 = 6 new characters

    def test_chunk_ids_and_indices_deterministic(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id=f"e{i}",
                element_type=ElementType.TEXT,
                content=f"Deterministic paragraph {i} text content.",
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=80, chunk_overlap=20)
        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)

        assert len(chunks1) == len(chunks2)
        for idx, (c1, c2) in enumerate(zip(chunks1, chunks2, strict=True)):
            assert c1.chunk_index == idx
            assert c1.chunk_id == f"{base_metadata.document_id}_c{idx:04d}"
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text


class TestWhitespaceAndOmissionAccounting:
    """Category D: Whitespace preservation, bounded omission, and metadata audit."""

    def test_empty_and_whitespace_only_elements(self, base_metadata: DocumentMetadata) -> None:
        elements = [
            ExtractedElement(
                element_id="e_empty",
                element_type=ElementType.TEXT,
                content="",
                order_index=0,
            ),
            ExtractedElement(
                element_id="e_ws",
                element_type=ElementType.TEXT,
                content="   \n\t\r   ",
                order_index=1,
            ),
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_long_interior_whitespace_regression_500_100(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Regression: 'hello' + ' '*900 + 'world' with C=500, O=100.

        Guarantees:
        - Both 'hello' and 'world' preserved in order.
        - Exactly 2 chunks emitted.
        - Chunk 0: 'hello' + 495 spaces (len 500).
        - Chunk 1: 105 spaces + 'world' (len 110).
        - Omitted interval [500, 800) length 300 recorded in Chunk 1 metadata.
        - Total characters: 500 (Chunk 0) + 300 (omitted) + 110 (Chunk 1) = 910.
        """
        content = "hello" + (" " * 900) + "world"  # 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_reg_500",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert "hello" in chunks[0].text
        assert "world" in chunks[1].text
        for c in chunks:
            assert len(c.text) <= 500
            assert len(c.text.strip()) > 0

        # Chunk 0 has no omissions
        assert "omitted_whitespace_intervals" not in chunks[0].metadata

        # Chunk 1 records the exact omitted interval [500, 800)
        omitted = chunks[1].metadata.get("omitted_whitespace_intervals")
        assert omitted is not None
        assert len(omitted) == 1
        entry = omitted[0]
        assert entry["element_id"] == "elem_reg_500"
        assert entry["start"] == 500
        assert entry["end"] == 800
        assert entry["length"] == 300

        # Every single source character is accounted for
        assert len(chunks[0].text) + entry["length"] + len(chunks[1].text) == 910

    def test_long_interior_whitespace_regression_200_50(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Regression: 'hello' + ' '*900 + 'world' with C=200, O=50.

        Guarantees:
        - Both 'hello' and 'world' preserved in order.
        - Exactly 2 chunks emitted.
        - Chunk 0 has len 200. Chunk 1 has len 160.
        - Omitted interval [200, 750) length 550 recorded in Chunk 1 metadata.
        - Total characters: 200 + 550 + 160 = 910.
        """
        content = "hello" + (" " * 900) + "world"  # 910 chars
        elements = [
            ExtractedElement(
                element_id="elem_reg_200",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert "hello" in chunks[0].text
        assert "world" in chunks[1].text
        for c in chunks:
            assert len(c.text) <= 200
            assert len(c.text.strip()) > 0

        omitted = chunks[1].metadata.get("omitted_whitespace_intervals")
        assert omitted is not None
        assert len(omitted) == 1
        entry = omitted[0]
        assert entry["element_id"] == "elem_reg_200"
        assert entry["start"] == 200
        assert entry["end"] == 750
        assert entry["length"] == 550

        assert len(chunks[0].text) + entry["length"] + len(chunks[1].text) == 910

    def test_retainable_whitespace_only_fragments_not_omitted(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """When an interior whitespace run fits with adjacent text, it is NOT omitted."""
        # Content: 40 chars 'A' + 30 spaces + 40 chars 'B' = 110 chars
        content = ("A" * 40) + (" " * 30) + ("B" * 40)
        elements = [
            ExtractedElement(
                element_id="e_retain",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == content
        assert "omitted_whitespace_intervals" not in chunks[0].metadata

    def test_no_non_whitespace_character_ever_omitted(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """All non-whitespace characters must be present across emitted chunks."""
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        content = (" " * 400).join(words)
        elements = [
            ExtractedElement(
                element_id="e_words",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        full_output = " ".join(c.text for c in chunks)
        for w in words:
            assert w in full_output, f"Word {w} was lost!"

    def test_no_whitespace_only_chunks_emitted(self, base_metadata: DocumentMetadata) -> None:
        """Every emitted chunk must satisfy text.strip() != ''."""
        content = "start" + (" " * 1500) + "middle" + (" " * 1500) + "end"
        elements = [
            ExtractedElement(
                element_id="e_multi_ws",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=300, chunk_overlap=60)
        chunks = chunker.chunk(doc)

        for c in chunks:
            assert len(c.text.strip()) > 0, "Emitted an invalid whitespace-only chunk!"

    def test_overlapping_intervals_not_classified_as_omitted(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Standard splitter overlap must never be reported as omitted whitespace."""
        text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 10
        elements = [
            ExtractedElement(
                element_id="e_no_omit",
                element_type=ElementType.TEXT,
                content=text.strip(),
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=150, chunk_overlap=40)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "omitted_whitespace_intervals" not in chunk.metadata

    def test_whitespace_omissions_tracked_without_spurious_records(
        self, base_metadata: DocumentMetadata
    ) -> None:
        """Verify that when interior whitespace is omitted, the omission interval is exact."""
        content = "first_token" + (" " * 600) + "second_token"
        elements = [
            ExtractedElement(
                element_id="e_precise",
                element_type=ElementType.TEXT,
                content=content,
                order_index=0,
            )
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        omissions = chunks[1].metadata.get("omitted_whitespace_intervals", [])
        assert len(omissions) == 1
        omitted = omissions[0]
        assert omitted["element_id"] == "e_precise"
        # Confirm that the omitted interval covers only spaces
        omitted_text = content[omitted["start"] : omitted["end"]]
        assert len(omitted_text) == omitted["length"]
        assert omitted_text.strip() == ""


class TestGeneralPropertiesAndInvariants:
    """Category E: Property tests validating all 12 chunker contract invariants."""

    @pytest.mark.parametrize(
        "c_size,c_overlap",
        [
            (50, 10),
            (100, 0),
            (100, 25),
            (200, 50),
            (500, 150),
        ],
    )
    def test_all_chunks_satisfy_invariants(
        self, base_metadata: DocumentMetadata, c_size: int, c_overlap: int
    ) -> None:
        elements = [
            ExtractedElement(
                element_id=f"elem_{i}",
                element_type=ElementType.TEXT,
                content=(
                    f"Heading {i}\n" + ("Text content sentence with various words. " * (i * 3 + 1))
                ),
                order_index=i,
            )
            for i in range(5)
        ]
        doc = NormalizedDocument(metadata=base_metadata, elements=elements)
        chunker = ElementAwareChunker(chunk_size=c_size, chunk_overlap=c_overlap)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            # Invariant 1: 0 < len(chunk.text) <= chunk_size
            assert 0 < len(chunk.text) <= c_size
            # Invariant 2: chunk.text contains non-whitespace
            assert len(chunk.text.strip()) > 0
            # Invariant 3: Pydantic v2 validation passes
            assert Chunk.model_validate(chunk.model_dump()) == chunk
            # Invariant 4: Sequential index
            assert chunk.chunk_index == idx
            # Invariant 5: Deterministic chunk ID
            assert chunk.chunk_id == f"{base_metadata.document_id}_c{idx:04d}"

        # Invariant 6: Element order preserved
        seen_order: list[str] = []
        for chunk in chunks:
            for eid in chunk.source_element_ids:
                if not seen_order or seen_order[-1] != eid:
                    seen_order.append(eid)
        expected_order = [f"elem_{i}" for i in range(5)]
        assert seen_order == expected_order
