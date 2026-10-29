"""Unit tests for canonical normalization and chunk schemas."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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

# ---------------------------------------------------------------------------
# DocumentMetadata tests
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    def test_valid_minimal(self):
        meta = DocumentMetadata(document_id="doc-1", filename="test.pdf")
        assert meta.document_id == "doc-1"
        assert meta.filename == "test.pdf"
        assert meta.mime_type is None
        assert meta.content_sha256 is None
        assert meta.byte_size is None
        assert meta.created_at is None
        assert meta.processed_at is None
        assert meta.source_uri is None
        assert meta.custom_metadata == {}

    def test_valid_full(self):
        now = datetime.now(UTC)
        sha = "a" * 64
        meta = DocumentMetadata(
            document_id="doc-123",
            filename="document.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            content_sha256=sha,
            byte_size=1024,
            created_at=now,
            processed_at=now,
            source_uri="file:///data/docs/document.docx",
            custom_metadata={"author": "Alice", "tags": ["report"]},
        )
        assert meta.byte_size == 1024
        assert meta.content_sha256 == sha
        assert meta.created_at == now

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            DocumentMetadata()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            DocumentMetadata(document_id="doc-1")  # type: ignore[call-arg]

    def test_whitespace_only_identifiers(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            DocumentMetadata(document_id="   ", filename="file.pdf")

        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            DocumentMetadata(document_id="doc-1", filename="  \t  ")

    def test_sha256_validation(self):
        valid_upper = "A" * 64
        meta = DocumentMetadata(
            document_id="doc-1", filename="file.pdf", content_sha256=valid_upper
        )
        assert meta.content_sha256 == valid_upper.lower()

        # Too short (63 chars)
        with pytest.raises(ValidationError, match="64-character hexadecimal"):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", content_sha256="a" * 63)

        # Too long (65 chars)
        with pytest.raises(ValidationError, match="64-character hexadecimal"):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", content_sha256="a" * 65)

        # Non-hex characters
        with pytest.raises(ValidationError, match="64-character hexadecimal"):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", content_sha256="z" * 64)

    def test_negative_byte_size(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", byte_size=-1)

    def test_naive_datetime_rejected(self):
        naive_dt = datetime(2026, 1, 1, 12, 0, 0)
        with pytest.raises(ValidationError, match="timezone-aware"):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", created_at=naive_dt)

        with pytest.raises(ValidationError, match="timezone-aware"):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", processed_at=naive_dt)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(document_id="doc-1", filename="file.pdf", unexpected_field="foo")


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_valid_box(self):
        box = BoundingBox(x0=0.1, y0=0.2, x1=0.8, y1=0.9)
        assert box.x0 == 0.1
        assert box.y0 == 0.2
        assert box.x1 == 0.8
        assert box.y1 == 0.9

    def test_point_or_line_box(self):
        # x0 == x1 and y0 == y1 is structurally valid (zero area)
        box = BoundingBox(x0=0.5, y0=0.5, x1=0.5, y1=0.5)
        assert box.x0 == box.x1

    def test_out_of_range_coordinates(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=-0.1, y0=0.0, x1=0.5, y1=0.5)

        with pytest.raises(ValidationError):
            BoundingBox(x0=0.0, y0=0.0, x1=1.05, y1=0.5)

    def test_inverted_coordinates(self):
        with pytest.raises(ValidationError, match="x0 .* cannot exceed x1"):
            BoundingBox(x0=0.8, y0=0.1, x1=0.2, y1=0.9)

        with pytest.raises(ValidationError, match="y0 .* cannot exceed y1"):
            BoundingBox(x0=0.1, y0=0.9, x1=0.8, y1=0.2)

    def test_to_absolute(self):
        box = BoundingBox(x0=0.1, y0=0.2, x1=0.6, y1=0.8)
        abs_coords = box.to_absolute(page_width=1000.0, page_height=2000.0)
        assert abs_coords == (100.0, 400.0, 600.0, 1600.0)

    def test_to_absolute_invalid_dimensions(self):
        box = BoundingBox(x0=0.1, y0=0.2, x1=0.6, y1=0.8)
        with pytest.raises(ValueError, match="strictly positive"):
            box.to_absolute(page_width=0, page_height=500)

        with pytest.raises(ValueError, match="strictly positive"):
            box.to_absolute(page_width=500, page_height=-10)


# ---------------------------------------------------------------------------
# PageLocation tests
# ---------------------------------------------------------------------------


class TestPageLocation:
    def test_valid_minimal(self):
        loc = PageLocation(page_number=1)
        assert loc.page_number == 1
        assert loc.bbox is None
        assert loc.coordinate_system == "normalized_top_left"
        assert loc.page_width is None
        assert loc.page_height is None

    def test_valid_full(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=0.5)
        loc = PageLocation(
            page_number=3,
            bbox=bbox,
            coordinate_system="normalized_top_left",
            page_width=612.0,
            page_height=792.0,
        )
        assert loc.page_number == 3
        assert loc.bbox == bbox
        assert loc.page_width == 612.0

    def test_invalid_page_number(self):
        with pytest.raises(ValidationError):
            PageLocation(page_number=0)

        with pytest.raises(ValidationError):
            PageLocation(page_number=-5)

    def test_invalid_page_dimensions(self):
        with pytest.raises(ValidationError):
            PageLocation(page_number=1, page_width=0)

        with pytest.raises(ValidationError):
            PageLocation(page_number=1, page_height=-100.0)


# ---------------------------------------------------------------------------
# TableData tests
# ---------------------------------------------------------------------------


class TestTableData:
    def test_valid_table(self):
        tbl = TableData(
            rows=[["A1", "B1"], ["A2", "B2"]],
            headers=["Col A", "Col B"],
            caption="Sample Table",
        )
        assert len(tbl.rows) == 2
        assert tbl.headers == ["Col A", "Col B"]
        assert tbl.caption == "Sample Table"


# ---------------------------------------------------------------------------
# ExtractedElement tests
# ---------------------------------------------------------------------------


class TestExtractedElement:
    def test_valid_element(self):
        elem = ExtractedElement(
            element_id="elem-1",
            element_type=ElementType.TEXT,
            content="Introduction section text",
            order_index=0,
            location=PageLocation(page_number=1),
        )
        assert elem.element_id == "elem-1"
        assert elem.element_type == ElementType.TEXT
        assert elem.order_index == 0
        assert elem.location is not None
        assert elem.location.page_number == 1

    def test_table_element(self):
        tbl = TableData(rows=[["1", "2"]])
        elem = ExtractedElement(
            element_id="tbl-1",
            element_type=ElementType.TABLE,
            content="| 1 | 2 |",
            order_index=1,
            table_data=tbl,
        )
        assert elem.element_type == ElementType.TABLE
        assert elem.table_data is not None
        assert elem.table_data.rows == [["1", "2"]]

    def test_whitespace_element_id(self):
        with pytest.raises(ValidationError, match="whitespace-only"):
            ExtractedElement(
                element_id="   ",
                element_type=ElementType.HEADING,
                content="Title",
                order_index=0,
            )

    def test_negative_order_index(self):
        with pytest.raises(ValidationError):
            ExtractedElement(
                element_id="elem-1",
                element_type=ElementType.TEXT,
                content="Text",
                order_index=-1,
            )


# ---------------------------------------------------------------------------
# NormalizedDocument tests
# ---------------------------------------------------------------------------


class TestNormalizedDocument:
    def test_valid_document(self):
        meta = DocumentMetadata(document_id="doc-1", filename="sample.pdf")
        elem1 = ExtractedElement(
            element_id="el-1",
            element_type=ElementType.HEADING,
            content="Header",
            order_index=0,
        )
        elem2 = ExtractedElement(
            element_id="el-2",
            element_type=ElementType.TEXT,
            content="Paragraph",
            order_index=1,
        )
        doc = NormalizedDocument(metadata=meta, elements=[elem1, elem2], page_count=5)
        assert doc.metadata.document_id == "doc-1"
        assert len(doc.elements) == 2
        assert doc.page_count == 5

    def test_duplicate_element_id_rejected(self):
        meta = DocumentMetadata(document_id="doc-1", filename="sample.pdf")
        elem1 = ExtractedElement(
            element_id="dup-id",
            element_type=ElementType.TEXT,
            content="First",
            order_index=0,
        )
        elem2 = ExtractedElement(
            element_id="dup-id",
            element_type=ElementType.TEXT,
            content="Second",
            order_index=1,
        )
        with pytest.raises(ValidationError, match="Duplicate element_id detected"):
            NormalizedDocument(metadata=meta, elements=[elem1, elem2])

    def test_invalid_page_count(self):
        meta = DocumentMetadata(document_id="doc-1", filename="sample.pdf")
        with pytest.raises(ValidationError):
            NormalizedDocument(metadata=meta, elements=[], page_count=0)


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------


class TestChunk:
    def test_valid_chunk(self):
        loc = PageLocation(
            page_number=1,
            bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        )
        chunk = Chunk(
            chunk_id="chunk-001",
            document_id="doc-1",
            text="This is a valid chunk of text.",
            chunk_index=0,
            token_count=7,
            source_element_ids=["el-1", "el-2"],
            locations=[loc],
            metadata={"strategy": "recursive_character"},
        )
        assert chunk.chunk_id == "chunk-001"
        assert chunk.document_id == "doc-1"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 7
        assert len(chunk.source_element_ids) == 2
        assert len(chunk.locations) == 1

    def test_whitespace_only_text_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="   \n\t   ",
                chunk_index=0,
            )

        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="",
                chunk_index=0,
            )

    def test_whitespace_only_ids_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            Chunk(
                chunk_id="   ",
                document_id="d1",
                text="valid text",
                chunk_index=0,
            )

        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            Chunk(
                chunk_id="c1",
                document_id="   ",
                text="valid text",
                chunk_index=0,
            )

    def test_negative_chunk_index(self):
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="valid text",
                chunk_index=-1,
            )

    def test_negative_token_count(self):
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="valid text",
                chunk_index=0,
                token_count=-5,
            )

    def test_optional_token_count_not_fabricated(self):
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="valid text",
            chunk_index=0,
        )
        assert chunk.token_count is None


# ---------------------------------------------------------------------------
# Serialization & Roundtrip tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_document_roundtrip(self):
        meta = DocumentMetadata(
            document_id="doc-99",
            filename="paper.pdf",
            mime_type="application/pdf",
            content_sha256="b" * 64,
            byte_size=2048,
            created_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC),
            processed_at=datetime(2026, 5, 1, 10, 5, 0, tzinfo=UTC),
            source_uri="file:///papers/paper.pdf",
            custom_metadata={"category": "AI"},
        )
        elem = ExtractedElement(
            element_id="e-1",
            element_type=ElementType.HEADING,
            content="Abstract",
            order_index=0,
            location=PageLocation(
                page_number=1,
                bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
                page_width=595.0,
                page_height=842.0,
            ),
        )
        doc = NormalizedDocument(metadata=meta, elements=[elem], page_count=1)

        doc_json = doc.model_dump_json()
        doc_data = json.loads(doc_json)
        reconstructed = NormalizedDocument.model_validate(doc_data)

        assert reconstructed == doc
        assert reconstructed.metadata.created_at == meta.created_at
        assert reconstructed.elements[0].location is not None
        assert reconstructed.elements[0].location.bbox is not None
        assert reconstructed.elements[0].location.bbox.x0 == 0.1

    def test_chunk_roundtrip(self):
        loc = PageLocation(
            page_number=2,
            bbox=BoundingBox(x0=0.05, y0=0.1, x1=0.95, y1=0.5),
        )
        chunk = Chunk(
            chunk_id="chk-42",
            document_id="doc-99",
            text="Chunk content for embedding.",
            chunk_index=3,
            token_count=45,
            source_element_ids=["e-1", "e-2"],
            locations=[loc],
            metadata={"source": "section-1"},
        )
        chunk_json = chunk.model_dump_json()
        reconstructed = Chunk.model_validate_json(chunk_json)
        assert reconstructed == chunk
        assert reconstructed.text == "Chunk content for embedding."
        assert reconstructed.locations[0].bbox.y1 == 0.5  # type: ignore[union-attr]
