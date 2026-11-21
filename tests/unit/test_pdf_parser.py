"""Unit tests for PDFParser verifying multi-page extraction, provenance, and error handling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.errors import CorruptedFileError, EncryptedDocumentError
from core.normalize.schema import NormalizedDocument
from core.parsers.pdf_text import PDFParser

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


class TestPDFParser:
    """Test suite for PDFParser."""

    @pytest.fixture
    def parser(self) -> PDFParser:
        return PDFParser()

    def test_parse_valid_pdf_multi_page(self, parser: PDFParser) -> None:
        sample_path = FIXTURES_DIR / "sample.pdf"
        assert sample_path.exists(), f"Missing fixture at {sample_path}"

        doc = parser.parse(sample_path)

        assert isinstance(doc, NormalizedDocument)
        raw_bytes = sample_path.read_bytes()
        expected_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
        assert doc.metadata.content_sha256 == expected_sha
        assert doc.metadata.document_id == f"doc_{expected_sha[:16]}"
        assert doc.metadata.mime_type == "application/pdf"
        assert doc.metadata.filename == "sample.pdf"
        assert doc.metadata.byte_size == len(raw_bytes)

        # Page count
        assert doc.page_count == 2
        assert len(doc.elements) >= 2

        # Check page provenance
        page_numbers = {elem.location.page_number for elem in doc.elements if elem.location}
        assert 1 in page_numbers
        assert 2 in page_numbers

        # Verify element IDs are deterministic and sequentially indexed
        for idx, elem in enumerate(doc.elements):
            assert elem.order_index == idx
            assert elem.element_id == f"{doc.metadata.document_id}_e{idx:04d}"

        # Verify text contents
        all_text = " ".join(e.content for e in doc.elements)
        assert "Page 1 content" in all_text
        assert "Page 2 content" in all_text

    def test_parse_empty_file_raises_corrupted(self, parser: PDFParser) -> None:
        empty_path = FIXTURES_DIR / "empty.pdf"
        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse(empty_path)
        assert "empty (0 bytes)" in str(exc_info.value)

    def test_parse_corrupt_file_raises_corrupted(self, parser: PDFParser) -> None:
        corrupt_path = FIXTURES_DIR / "corrupt.pdf"
        with pytest.raises(CorruptedFileError):
            parser.parse(corrupt_path)

    def test_parse_encrypted_pdf_raises_encrypted_document_error(self, parser: PDFParser) -> None:
        encrypted_path = FIXTURES_DIR / "encrypted.pdf"
        assert encrypted_path.exists()
        with pytest.raises(EncryptedDocumentError) as exc_info:
            parser.parse(encrypted_path)
        assert "encrypted" in str(exc_info.value).lower()

    def test_parse_non_existent_file_raises_corrupted(self, parser: PDFParser) -> None:
        missing_path = FIXTURES_DIR / "does_not_exist.pdf"
        with pytest.raises(CorruptedFileError):
            parser.parse(missing_path)
