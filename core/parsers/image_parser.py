"""Concrete parser for image files using Pillow and OCR engine."""

from __future__ import annotations

import hashlib
import io
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from core.errors import CorruptedFileError, OCRError
from core.normalize.schema import (
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)

# Supported image MIME types
_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


class ImageParser:
    """Synchronous parser for image files extracting text via OCR with provenance."""

    def __init__(
        self,
        ocr_engine: Callable[[Image.Image], str] | None = None,
    ) -> None:
        """Initialize ImageParser.

        Args:
            ocr_engine: Optional callable taking a PIL Image and returning extracted text.
                If None, defaults to pytesseract if tesseract is installed on the system.
        """
        self.ocr_engine = ocr_engine

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse an image file, performing OCR if an engine is available.

        Args:
            file_path: Filesystem path to the image file.

        Returns:
            A NormalizedDocument containing extracted OCR text elements.

        Raises:
            CorruptedFileError: If the file cannot be read, is empty, or cannot be decoded.
            OCRError: If no OCR engine is available or OCR extraction fails.
        """
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise CorruptedFileError(f"Failed to read file '{file_path}': {exc}") from exc

        if not raw_bytes:
            raise CorruptedFileError(f"Image file '{file_path.name}' is empty (0 bytes).")

        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()  # Force verification and decode
            width, height = image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise CorruptedFileError(
                f"File '{file_path.name}' is not a valid or readable image: {exc}"
            ) from exc

        content_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        byte_size = len(raw_bytes)
        document_id = f"doc_{content_sha256[:16]}"
        processed_at = datetime.now(UTC)

        created_at: datetime | None = None
        try:
            st = file_path.stat()
            created_at = datetime.fromtimestamp(st.st_ctime, tz=UTC)
        except OSError:
            pass

        ext = file_path.suffix.lower()
        mime_type = _IMAGE_MIME_TYPES.get(ext, "image/png")

        # Run OCR
        extracted_text = self._run_ocr(image, file_path)

        elements: list[ExtractedElement] = []
        if extracted_text and extracted_text.strip():
            location = PageLocation(
                page_number=1,
                page_width=float(width),
                page_height=float(height),
            )
            element_id = f"{document_id}_e0000"
            elements.append(
                ExtractedElement(
                    element_id=element_id,
                    element_type=ElementType.TEXT,
                    content=extracted_text.strip(),
                    order_index=0,
                    location=location,
                    table_data=None,
                    metadata={"source_type": "image", "extraction_method": "ocr"},
                )
            )

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            mime_type=mime_type,
            content_sha256=content_sha256,
            byte_size=byte_size,
            created_at=created_at,
            processed_at=processed_at,
            source_uri=None,
            custom_metadata={"width": width, "height": height},
        )

        return NormalizedDocument(
            metadata=metadata,
            elements=elements,
            page_count=1,
        )

    def _run_ocr(self, image: Image.Image, file_path: Path) -> str:
        """Execute OCR using the configured engine or system tesseract."""
        if self.ocr_engine is not None:
            try:
                return self.ocr_engine(image)
            except Exception as exc:
                raise OCRError(f"OCR execution failed on '{file_path.name}': {exc}") from exc

        # Check for system tesseract
        if shutil.which("tesseract") is None:
            raise OCRError(
                f"OCR engine (tesseract) not found on PATH to process '{file_path.name}'. "
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
