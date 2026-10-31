"""Concrete parser for plain-text (.txt) and Markdown (.md) documents."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from core.errors import CorruptedFileError
from core.normalize.schema import (
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
)


class PlainTextParser:
    """Synchronous parser for UTF-8 plain text (.txt) and Markdown (.md) documents."""

    _MIME_TYPES: dict[str, str] = {
        ".txt": "text/plain",
        ".md": "text/markdown",
    }

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse a text or Markdown file into a canonical NormalizedDocument.

        Args:
            file_path: Filesystem path to the text or Markdown document.

        Returns:
            A NormalizedDocument containing metadata and extracted elements.

        Raises:
            CorruptedFileError: If the file cannot be read or contains invalid UTF-8 bytes.
        """
        # Read raw bytes to ensure exact checksum and size calculation
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise CorruptedFileError(f"Failed to read file '{file_path}': {exc}") from exc

        # Decode strictly as UTF-8
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CorruptedFileError(
                f"Failed to decode '{file_path.name}' as UTF-8: {exc}"
            ) from exc

        content_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        byte_size = len(raw_bytes)
        document_id = f"doc_{content_sha256[:16]}"

        # Timestamps
        processed_at = datetime.now(UTC)
        try:
            st = file_path.stat()
            created_at = datetime.fromtimestamp(st.st_ctime, tz=UTC)
        except OSError:
            created_at = None

        ext = file_path.suffix.lower()
        mime_type = self._MIME_TYPES.get(ext, "text/plain")

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            mime_type=mime_type,
            content_sha256=content_sha256,
            byte_size=byte_size,
            created_at=created_at,
            processed_at=processed_at,
            source_uri=None,
            custom_metadata={},
        )

        elements = self._extract_elements(text, document_id)

        return NormalizedDocument(
            metadata=metadata,
            elements=elements,
            page_count=None,
        )

    def _extract_elements(self, text: str, document_id: str) -> list[ExtractedElement]:
        """Extract ordered content elements from decoded text with minimal Markdown recognition."""
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

            # Header recognition: # Header, ## Header, etc.
            if re.match(r"^#{1,6}\s+", trimmed):
                flush_para()
                raw_elements.append((ElementType.HEADER, trimmed))
            # List item recognition: - item, * item, 1. item
            elif re.match(r"^([-*]|\d+\.)\s+", trimmed):
                flush_para()
                raw_elements.append((ElementType.LIST_ITEM, trimmed))
            else:
                current_para.append(trimmed)

        flush_para()

        extracted: list[ExtractedElement] = []
        for order_index, (elem_type, content) in enumerate(raw_elements):
            element_id = f"{document_id}_e{order_index:04d}"
            extracted.append(
                ExtractedElement(
                    element_id=element_id,
                    element_type=elem_type,
                    content=content,
                    order_index=order_index,
                    location=None,
                    table_data=None,
                    metadata={},
                )
            )

        return extracted
