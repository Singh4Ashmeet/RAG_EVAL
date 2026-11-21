"""Concrete parser for native text PDF documents using pypdf."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.errors import CorruptedFileError, EncryptedDocumentError
from core.normalize.schema import (
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)


class PDFParser:
    """Synchronous parser for text-native PDF files preserving page-level provenance."""

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse a PDF file into a canonical NormalizedDocument.

        Args:
            file_path: Filesystem path to the PDF document.

        Returns:
            A NormalizedDocument containing metadata, elements, and page provenance.

        Raises:
            CorruptedFileError: If the file cannot be read, is empty, or is invalid PDF.
            EncryptedDocumentError: If the PDF is encrypted and cannot be opened without a password.
        """
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise CorruptedFileError(f"Failed to read file '{file_path}': {exc}") from exc

        if not raw_bytes:
            raise CorruptedFileError(f"PDF file '{file_path.name}' is empty (0 bytes).")

        # Validate PDF magic header
        if b"%PDF-" not in raw_bytes[:1024]:
            raise CorruptedFileError(f"File '{file_path.name}' does not have a valid PDF header.")

        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except (PdfReadError, Exception) as exc:
            raise CorruptedFileError(f"Failed to parse PDF file '{file_path.name}': {exc}") from exc

        if reader.is_encrypted:
            # Try decrypting with empty password
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:  # 0 indicates failed decryption
                    raise EncryptedDocumentError(
                        f"PDF document '{file_path.name}' is encrypted and password-protected."
                    )
            except Exception as exc:
                raise EncryptedDocumentError(
                    f"PDF document '{file_path.name}' is encrypted and password-protected: {exc}"
                ) from exc

        content_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        byte_size = len(raw_bytes)
        document_id = f"doc_{content_sha256[:16]}"
        processed_at = datetime.now(UTC)

        # File creation timestamp fallback
        created_at: datetime | None = None
        try:
            st = file_path.stat()
            created_at = datetime.fromtimestamp(st.st_ctime, tz=UTC)
        except OSError:
            pass

        # PDF metadata extraction
        custom_metadata: dict[str, Any] = {}
        try:
            pdf_info = reader.metadata
            if pdf_info:
                if pdf_info.title:
                    custom_metadata["title"] = str(pdf_info.title)
                if pdf_info.author:
                    custom_metadata["author"] = str(pdf_info.author)
                if pdf_info.creator:
                    custom_metadata["creator"] = str(pdf_info.creator)
                if pdf_info.producer:
                    custom_metadata["producer"] = str(pdf_info.producer)
                if pdf_info.creation_date:
                    if isinstance(pdf_info.creation_date, datetime):
                        dt = pdf_info.creation_date
                        created_at = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
                    else:
                        custom_metadata["creation_date_raw"] = str(pdf_info.creation_date)
        except Exception:
            # Metadata extraction is best-effort
            pass

        page_count = len(reader.pages)
        elements: list[ExtractedElement] = []
        order_index = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1

            # Determine page dimensions
            page_width: float | None = None
            page_height: float | None = None
            try:
                box = page.mediabox
                page_width = float(box.width)
                page_height = float(box.height)
            except Exception:
                pass

            location = PageLocation(
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )

            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_elements = self._extract_page_elements(
                text=page_text,
                document_id=document_id,
                start_order_index=order_index,
                location=location,
            )
            elements.extend(page_elements)
            order_index += len(page_elements)

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            mime_type="application/pdf",
            content_sha256=content_sha256,
            byte_size=byte_size,
            created_at=created_at,
            processed_at=processed_at,
            source_uri=None,
            custom_metadata=custom_metadata,
        )

        return NormalizedDocument(
            metadata=metadata,
            elements=elements,
            page_count=page_count,
        )

    def _extract_page_elements(
        self,
        text: str,
        document_id: str,
        start_order_index: int,
        location: PageLocation,
    ) -> list[ExtractedElement]:
        """Extract structured elements from a single page's text."""
        raw_elements: list[tuple[ElementType, str]] = []
        current_para: list[str] = []

        def flush_para() -> None:
            if current_para:
                joined = "\n".join(current_para).strip()
                if joined:
                    raw_elements.append((ElementType.TEXT, joined))
                current_para.clear()

        for line in text.splitlines():
            trimmed = line.strip()
            if not trimmed:
                flush_para()
                continue

            if re.match(r"^#{1,6}\s+", trimmed):
                flush_para()
                raw_elements.append((ElementType.HEADER, trimmed))
            elif re.match(r"^([-*•]|\d+\.)\s+", trimmed):
                flush_para()
                raw_elements.append((ElementType.LIST_ITEM, trimmed))
            else:
                current_para.append(trimmed)

        flush_para()

        extracted: list[ExtractedElement] = []
        for idx, (elem_type, content) in enumerate(raw_elements):
            current_order = start_order_index + idx
            element_id = f"{document_id}_e{current_order:04d}"
            extracted.append(
                ExtractedElement(
                    element_id=element_id,
                    element_type=elem_type,
                    content=content,
                    order_index=current_order,
                    location=location,
                    table_data=None,
                    metadata={"page_number": location.page_number},
                )
            )

        return extracted
