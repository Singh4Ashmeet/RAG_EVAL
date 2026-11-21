"""Concrete parser for DOCX documents using python-docx."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from core.errors import CorruptedFileError, EncryptedDocumentError
from core.normalize.schema import (
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    TableData,
)
from core.parsers.tables import serialize_table


class DOCXParser:
    """Synchronous parser for Microsoft Word (.docx) documents."""

    _MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse a DOCX file into a canonical NormalizedDocument.

        Args:
            file_path: Filesystem path to the DOCX document.

        Returns:
            A NormalizedDocument containing metadata, text elements, and extracted tables.

        Raises:
            CorruptedFileError: If the file cannot be read, is empty, or is invalid DOCX.
            EncryptedDocumentError: If the document is password-protected or encrypted.
        """
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise CorruptedFileError(f"Failed to read file '{file_path}': {exc}") from exc

        if not raw_bytes:
            raise CorruptedFileError(f"DOCX file '{file_path.name}' is empty (0 bytes).")

        # Validate ZIP magic bytes
        if not raw_bytes.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            raise CorruptedFileError(
                f"File '{file_path.name}' does not have a valid DOCX zip container header."
            )

        # Check for encrypted OLE / password protected packages
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                # Test zip integrity
                bad_file = zf.testzip()
                if bad_file:
                    raise CorruptedFileError(
                        f"Corrupted zip entry '{bad_file}' in DOCX file '{file_path.name}'."
                    )
                namelist = zf.namelist()
                if "EncryptedPackage" in namelist or "encryptedKey" in namelist:
                    raise EncryptedDocumentError(
                        f"DOCX document '{file_path.name}' is encrypted or password-protected."
                    )
        except zipfile.BadZipFile as exc:
            raise CorruptedFileError(
                f"Corrupted or invalid zip container in DOCX file '{file_path.name}': {exc}"
            ) from exc

        try:
            doc = docx.Document(io.BytesIO(raw_bytes))
        except PackageNotFoundError as exc:
            raise CorruptedFileError(
                f"Invalid DOCX document package in '{file_path.name}': {exc}"
            ) from exc
        except Exception as exc:
            if "encrypted" in str(exc).lower() or "password" in str(exc).lower():
                raise EncryptedDocumentError(
                    f"DOCX document '{file_path.name}' is password-protected: {exc}"
                ) from exc
            raise CorruptedFileError(
                f"Failed to load DOCX document '{file_path.name}': {exc}"
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

        # DOCX Core Properties
        custom_metadata: dict[str, Any] = {}
        try:
            props = doc.core_properties
            if props.title:
                custom_metadata["title"] = str(props.title)
            if props.author:
                custom_metadata["author"] = str(props.author)
            if props.comments:
                custom_metadata["comments"] = str(props.comments)
            if props.created:
                dt = props.created
                created_at = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
            if props.modified:
                m_dt = props.modified
                m_aware = m_dt if m_dt.tzinfo is not None else m_dt.replace(tzinfo=UTC)
                custom_metadata["modified_at"] = m_aware.isoformat()
        except Exception:
            pass

        elements: list[ExtractedElement] = []
        order_index = 0

        # Traverse document body in exact XML sequence
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name.lower() if p.style and p.style.name else ""
                if "heading" in style_name or "title" in style_name:
                    elem_type = ElementType.HEADING
                elif "list" in style_name or "bullet" in style_name:
                    elem_type = ElementType.LIST_ITEM
                else:
                    elem_type = ElementType.TEXT

                element_id = f"{document_id}_e{order_index:04d}"
                elements.append(
                    ExtractedElement(
                        element_id=element_id,
                        element_type=elem_type,
                        content=text,
                        order_index=order_index,
                        location=None,
                        table_data=None,
                        metadata={"style": p.style.name if p.style else None},
                    )
                )
                order_index += 1

            elif isinstance(child, CT_Tbl):
                t = Table(child, doc)
                table_element = self._parse_table(
                    table=t,
                    document_id=document_id,
                    order_index=order_index,
                )
                if table_element is not None:
                    elements.append(table_element)
                    order_index += 1

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            mime_type=self._MIME_TYPE,
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
            page_count=None,
        )

    def _parse_table(
        self,
        table: Table,
        document_id: str,
        order_index: int,
    ) -> ExtractedElement | None:
        """Extract structured TableData and serialized markdown from a docx Table."""
        raw_rows: list[list[str]] = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            if any(cell for cell in row_cells):
                raw_rows.append(row_cells)

        if not raw_rows:
            return None

        # If table has >= 2 rows, treat first row as headers
        if len(raw_rows) >= 2:
            headers = raw_rows[0]
            data_rows = raw_rows[1:]
        else:
            headers = None
            data_rows = raw_rows

        table_data = TableData(
            rows=data_rows,
            headers=headers,
            caption=None,
        )
        serialized = serialize_table(table_data)

        element_id = f"{document_id}_e{order_index:04d}"
        return ExtractedElement(
            element_id=element_id,
            element_type=ElementType.TABLE,
            content=serialized,
            order_index=order_index,
            location=None,
            table_data=table_data,
            metadata={"num_rows": len(raw_rows), "num_cols": len(raw_rows[0]) if raw_rows else 0},
        )
