"""Deterministic document-structure-enriched chunk representations."""

from __future__ import annotations

import re
from typing import Any

from core.normalize.schema import Chunk, ElementType, NormalizedDocument


def extract_document_title(document: NormalizedDocument | None) -> str:
    """Extract a clean, deterministic document title from document metadata or first heading."""
    if document is None:
        return "Unknown Document"

    # Check for title/heading in first 3 elements
    for elem in document.elements[:3]:
        text = elem.content.strip()
        if elem.element_type in (ElementType.HEADER, ElementType.HEADING):
            clean = re.sub(r"^#+\s*", "", text).strip()
            if clean:
                return clean
        if elem.metadata.get("style", "").lower() == "title":
            if text:
                return text

    # Fallback to filename
    filename = document.metadata.filename
    clean_name = re.sub(r"\.[a-zA-Z0-9]+$", "", filename).replace("_", " ").replace("-", " ")
    return clean_name.title()


def extract_section_hierarchies(
    document: NormalizedDocument | None,
) -> dict[str, list[str]]:
    """Deterministically map every element_id in a document to its active heading hierarchy.

    Extracts:
    1. Markdown headers (#, ##, ###, etc.)
    2. Document headings from Word/PDF elements (ElementType.HEADING / HEADER)
    3. Structured RFC numbered sections (e.g., '1.  Introduction', '8.1.  Character Encoding')
    4. Plain-text section banners ('Section 1: ...', 'Incident ID: ...')

    Filters out table-of-contents dot-leader lines.
    """
    if document is None or not document.elements:
        return {}

    hierarchy: list[str] = []
    element_sections: dict[str, list[str]] = {}

    rfc_heading_re = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Za-z][^.\n]+(?:\s+[^.\n]+)*)\s*$")

    for elem in document.elements:
        text = elem.content.strip()
        is_heading = False
        level = 1
        heading_text = ""

        if elem.element_type in (ElementType.HEADER, ElementType.HEADING):
            is_heading = True
            m = re.match(r"^(#+)\s*(.*)$", text)
            if m:
                level = len(m.group(1))
                heading_text = m.group(2).strip()
            else:
                style = str(elem.metadata.get("style", "")).lower()
                if "title" in style:
                    level = 1
                elif "1" in style:
                    level = 2
                elif "2" in style:
                    level = 3
                else:
                    level = 2
                heading_text = text
        elif text:
            first_line = text.splitlines()[0].strip()
            m = rfc_heading_re.match(first_line)
            # Ensure it is not a TOC line with dot leaders
            if m and "...." not in text and ". ." not in text:
                is_heading = True
                nums = m.group(1).split(".")
                level = len([n for n in nums if n])
                heading_text = f"{m.group(1)}. {m.group(2).strip()}"
            elif first_line.startswith("Section ") and ":" in first_line and len(first_line) < 100:
                is_heading = True
                level = 1
                heading_text = first_line
            elif text.startswith("Incident ID:") and len(first_line) < 100:
                is_heading = True
                level = 1
                heading_text = first_line

        if is_heading and heading_text:
            while len(hierarchy) >= level:
                hierarchy.pop()
            hierarchy.append(heading_text)

        element_sections[elem.element_id] = list(hierarchy)

    return element_sections


def extract_table_and_element_metadata(
    document: NormalizedDocument | None,
) -> tuple[dict[str, ElementType], dict[str, str]]:
    """Extract element types and serialized table headers/captions for each element."""
    elem_types: dict[str, ElementType] = {}
    table_contexts: dict[str, str] = {}

    if document is None or not document.elements:
        return elem_types, table_contexts

    for elem in document.elements:
        elem_types[elem.element_id] = elem.element_type
        if elem.table_data is not None:
            parts = []
            if elem.table_data.caption:
                parts.append(f"Caption: {elem.table_data.caption}")
            if elem.table_data.headers:
                headers_str = " | ".join(str(h) for h in elem.table_data.headers)
                parts.append(f"Columns: {headers_str}")
            if parts:
                table_contexts[elem.element_id] = "; ".join(parts)

    return elem_types, table_contexts


class HeadingEnrichedRepresentation:
    """Representation B: Enriches chunk text with Document Title and Section Hierarchy."""

    def __init__(self, separator: str = " > ") -> None:
        self.separator = separator
        self._doc_context_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "heading_enriched"

    def _get_document_context(
        self, document: NormalizedDocument | None
    ) -> tuple[str, dict[str, list[str]]]:
        if document is None:
            return "Unknown Document", {}

        doc_id = document.metadata.document_id
        if doc_id not in self._doc_context_cache:
            title = extract_document_title(document)
            sections = extract_section_hierarchies(document)
            self._doc_context_cache[doc_id] = (title, sections)

        return self._doc_context_cache[doc_id]

    def transform(
        self,
        chunk: Chunk,
        document: NormalizedDocument | None = None,
    ) -> str:
        """Enrich chunk text with deterministic document title and section hierarchy."""
        doc_title, sections_map = self._get_document_context(document)

        # Determine section path from contributing source elements
        section_path_list: list[str] = []
        for elem_id in chunk.source_element_ids:
            if elem_id in sections_map and sections_map[elem_id]:
                section_path_list = sections_map[elem_id]
                break

        section_str = (
            self.separator.join(section_path_list) if section_path_list else "General Content"
        )

        return f"[DOCUMENT]\n{doc_title}\n\n[SECTION]\n{section_str}\n\n[CONTENT]\n{chunk.text}"


class FullStructuralRepresentation:
    """Representation C: Enriches chunk with Document Title, Section Hierarchy, and Structure."""

    def __init__(self, separator: str = " > ") -> None:
        self.separator = separator
        self._doc_context_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "full_structural"

    def _get_document_context(
        self, document: NormalizedDocument | None
    ) -> tuple[str, dict[str, list[str]], dict[str, ElementType], dict[str, str]]:
        if document is None:
            return "Unknown Document", {}, {}, {}

        doc_id = document.metadata.document_id
        if doc_id not in self._doc_context_cache:
            title = extract_document_title(document)
            sections = extract_section_hierarchies(document)
            elem_types, tables = extract_table_and_element_metadata(document)
            self._doc_context_cache[doc_id] = (title, sections, elem_types, tables)

        return self._doc_context_cache[doc_id]

    def transform(
        self,
        chunk: Chunk,
        document: NormalizedDocument | None = None,
    ) -> str:
        """Enrich chunk with document title, section hierarchy, element types, and table schema."""
        doc_title, sections_map, elem_types, tables = self._get_document_context(document)

        # Section path
        section_path_list: list[str] = []
        for elem_id in chunk.source_element_ids:
            if elem_id in sections_map and sections_map[elem_id]:
                section_path_list = sections_map[elem_id]
                break

        section_str = (
            self.separator.join(section_path_list) if section_path_list else "General Content"
        )

        # Structural metadata
        types_in_chunk = sorted(
            {elem_types[eid].value for eid in chunk.source_element_ids if eid in elem_types}
        )
        type_str = ", ".join(types_in_chunk) if types_in_chunk else "text"

        struct_parts = [f"Type: {type_str}"]

        # Page numbers
        pages = sorted({loc.page_number for loc in chunk.locations if loc.page_number})
        if pages:
            pages_str = ", ".join(str(p) for p in pages)
            struct_parts.append(f"Page: {pages_str}")

        # Table schema if any contributing element has table data
        for elem_id in chunk.source_element_ids:
            if elem_id in tables:
                struct_parts.append(f"Table Data: [{tables[elem_id]}]")
                break

        struct_str = " | ".join(struct_parts)

        return (
            f"[DOCUMENT]\n"
            f"{doc_title}\n\n"
            f"[SECTION]\n"
            f"{section_str}\n\n"
            f"[CONTENT]\n"
            f"{chunk.text}\n\n"
            f"[STRUCTURE]\n"
            f"{struct_str}"
        )
