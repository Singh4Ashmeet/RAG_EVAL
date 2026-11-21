"""Unit tests for ImageParser and ScannedPDFParser with OCR extraction and error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from core.errors import CorruptedFileError, OCRError
from core.normalize.schema import NormalizedDocument
from core.parsers.image_parser import ImageParser
from core.parsers.pdf_scanned import ScannedPDFParser

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


class TestImageParser:
    """Test suite for ImageParser."""

    def test_parse_image_with_custom_ocr_engine(self) -> None:
        sample_png = FIXTURES_DIR / "sample.png"
        assert sample_png.exists()

        def mock_ocr(image: Image.Image) -> str:
            assert isinstance(image, Image.Image)
            return "Extracted OCR text from PNG image"

        parser = ImageParser(ocr_engine=mock_ocr)
        doc = parser.parse(sample_png)

        assert isinstance(doc, NormalizedDocument)
        assert doc.metadata.mime_type == "image/png"
        assert doc.page_count == 1
        assert len(doc.elements) == 1

        elem = doc.elements[0]
        assert elem.content == "Extracted OCR text from PNG image"
        assert elem.location is not None
        assert elem.location.page_number == 1
        assert elem.location.page_width == 120.0
        assert elem.location.page_height == 60.0
        assert elem.metadata.get("source_type") == "image"
        assert elem.metadata.get("extraction_method") == "ocr"

    def test_parse_image_missing_ocr_engine_raises_ocr_error(self) -> None:
        sample_png = FIXTURES_DIR / "sample.png"
        parser = ImageParser(ocr_engine=None)

        with patch("shutil.which", return_value=None):
            with pytest.raises(OCRError) as exc_info:
                parser.parse(sample_png)
            assert "tesseract" in str(exc_info.value).lower()

    def test_parse_empty_image_raises_corrupted(self, tmp_path: Path) -> None:
        empty_img = tmp_path / "empty.png"
        empty_img.write_bytes(b"")
        parser = ImageParser()

        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse(empty_img)
        assert "empty (0 bytes)" in str(exc_info.value)

    def test_parse_corrupt_image_raises_corrupted(self, tmp_path: Path) -> None:
        corrupt_img = tmp_path / "corrupt.png"
        corrupt_img.write_bytes(b"\x89PNG\r\n\x1a\n not a valid png body")
        parser = ImageParser()

        with pytest.raises(CorruptedFileError):
            parser.parse(corrupt_img)


class TestScannedPDFParser:
    """Test suite for ScannedPDFParser."""

    def test_scanned_pdf_parses_native_text_first(self) -> None:
        sample_pdf = FIXTURES_DIR / "sample.pdf"
        parser = ScannedPDFParser()
        doc = parser.parse(sample_pdf)

        assert isinstance(doc, NormalizedDocument)
        assert doc.page_count == 2
        assert len(doc.elements) >= 2

    def test_scanned_pdf_missing_ocr_raises_when_no_text(self, tmp_path: Path) -> None:
        from pypdf import PageObject, PdfWriter

        writer = PdfWriter()
        # Add a blank page with no text
        writer.add_page(PageObject.create_blank_page(width=100, height=100))
        blank_pdf = tmp_path / "blank.pdf"
        with open(blank_pdf, "wb") as f:
            writer.write(f)

        parser = ScannedPDFParser(ocr_engine=None)
        # For a completely blank PDF without images or text, elements is empty
        doc = parser.parse(blank_pdf)
        assert doc.elements == []
