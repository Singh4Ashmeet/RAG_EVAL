"""Concrete parser for scanned / image-based PDF documents."""

from __future__ import annotations

import io
import shutil
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from core.errors import CorruptedFileError, OCRError
from core.normalize.schema import (
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)
from core.parsers.pdf_text import PDFParser


class ScannedPDFParser:
    """Synchronous parser for scanned PDFs combining pypdf with OCR fallback."""

    def __init__(
        self,
        ocr_engine: Callable[[Image.Image], str] | None = None,
    ) -> None:
        """Initialize ScannedPDFParser.

        Args:
            ocr_engine: Optional callable taking a PIL Image and returning extracted OCR text.
        """
        self.ocr_engine = ocr_engine
        self._text_parser = PDFParser()

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse a PDF document, running OCR on pages with images or missing text.

        Args:
            file_path: Filesystem path to the PDF document.

        Returns:
            A NormalizedDocument containing extracted elements.

        Raises:
            CorruptedFileError: If the file cannot be read, is empty, or is invalid PDF.
            EncryptedDocumentError: If the PDF is password-protected.
            OCRError: If OCR extraction is needed but no engine is available, or OCR fails.
        """
        # First try standard text extraction
        native_doc = self._text_parser.parse(file_path)

        # If native document already extracted elements, return it
        if native_doc.elements:
            return native_doc

        # If no elements were extracted, pages may be scanned images.
        try:
            raw_bytes = file_path.read_bytes()
            reader = PdfReader(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise CorruptedFileError(
                f"Failed to load PDF for OCR scanning '{file_path.name}': {exc}"
            ) from exc

        document_id = native_doc.metadata.document_id
        elements: list[ExtractedElement] = []
        order_index = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            page_width: float | None = None
            page_height: float | None = None
            try:
                page_width = float(page.mediabox.width)
                page_height = float(page.mediabox.height)
            except Exception:
                pass

            location = PageLocation(
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )

            # Check for embedded images on this page
            page_images = getattr(page, "images", [])
            page_texts: list[str] = []

            for img_file in page_images:
                try:
                    pil_img = Image.open(io.BytesIO(img_file.data))
                    text = self._run_ocr(pil_img, file_path)
                    if text and text.strip():
                        page_texts.append(text.strip())
                except OCRError:
                    raise
                except Exception:
                    # Ignore image extraction failure for individual images
                    pass

            if not page_texts and page_images:
                # If there were images but OCR wasn't able to extract anything without error
                pass

            for text in page_texts:
                element_id = f"{document_id}_e{order_index:04d}"
                elements.append(
                    ExtractedElement(
                        element_id=element_id,
                        element_type=ElementType.TEXT,
                        content=text,
                        order_index=order_index,
                        location=location,
                        table_data=None,
                        metadata={
                            "page_number": page_number,
                            "extraction_method": "ocr",
                        },
                    )
                )
                order_index += 1

        # Return updated document
        return NormalizedDocument(
            metadata=native_doc.metadata,
            elements=elements,
            page_count=native_doc.page_count,
        )

    def _run_ocr(self, image: Image.Image, file_path: Path) -> str:
        """Execute OCR using the configured engine or system tesseract."""
        if self.ocr_engine is not None:
            try:
                return self.ocr_engine(image)
            except Exception as exc:
                raise OCRError(f"OCR execution failed on '{file_path.name}': {exc}") from exc

        if shutil.which("tesseract") is None:
            raise OCRError(
                f"OCR engine (tesseract) not found on PATH for scanned PDF '{file_path.name}'. "
                "Install Tesseract or provide a custom ocr_engine."
            )

        try:
            import pytesseract

            return str(pytesseract.image_to_string(image))
        except ImportError as exc:
            raise OCRError(
                "pytesseract library is not installed. Install pytesseract or pass an ocr_engine."
            ) from exc
        except Exception as exc:
            raise OCRError(f"OCR execution failed on '{file_path.name}': {exc}") from exc
