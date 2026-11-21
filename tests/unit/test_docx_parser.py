"""Unit tests for DOCXParser verifying document structure, table extraction, and error handling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.errors import CorruptedFileError
from core.normalize.schema import ElementType, NormalizedDocument
from core.parsers.docx_parser import DOCXParser

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


class TestDOCXParser:
    """Test suite for DOCXParser."""

    @pytest.fixture
    def parser(self) -> DOCXParser:
        return DOCXParser()

    def test_parse_valid_docx(self, parser: DOCXParser) -> None:
        sample_path = FIXTURES_DIR / "sample.docx"
        assert sample_path.exists()

        doc = parser.parse(sample_path)

        assert isinstance(doc, NormalizedDocument)
        raw_bytes = sample_path.read_bytes()
        expected_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
        assert doc.metadata.content_sha256 == expected_sha
        assert doc.metadata.document_id == f"doc_{expected_sha[:16]}"
        assert (
            doc.metadata.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert doc.metadata.filename == "sample.docx"
        assert doc.metadata.byte_size == len(raw_bytes)

        # Check element types
        element_types = [elem.element_type for elem in doc.elements]
        assert ElementType.HEADING in element_types
        assert ElementType.TEXT in element_types
        assert ElementType.TABLE in element_types

        # Verify sequential ordering
        for idx, elem in enumerate(doc.elements):
            assert elem.order_index == idx
            assert elem.element_id == f"{doc.metadata.document_id}_e{idx:04d}"

        # Find and verify table element
        table_elems = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        assert len(table_elems) == 1
        tbl = table_elems[0]
        assert tbl.table_data is not None
        assert tbl.table_data.headers == ["Metric", "Score"]
        assert tbl.table_data.rows == [["Accuracy", "0.94"], ["Latency", "42ms"]]
        assert "| Metric | Score |" in tbl.content
        assert "| Accuracy | 0.94 |" in tbl.content

    def test_parse_empty_docx_raises_corrupted(self, parser: DOCXParser) -> None:
        empty_path = FIXTURES_DIR / "empty.docx"
        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse(empty_path)
        assert "empty (0 bytes)" in str(exc_info.value)

    def test_parse_corrupt_docx_raises_corrupted(self, parser: DOCXParser) -> None:
        corrupt_path = FIXTURES_DIR / "corrupt.docx"
        with pytest.raises(CorruptedFileError):
            parser.parse(corrupt_path)

    def test_parse_missing_file_raises_corrupted(self, parser: DOCXParser) -> None:
        missing_path = FIXTURES_DIR / "does_not_exist.docx"
        with pytest.raises(CorruptedFileError):
            parser.parse(missing_path)
