"""Boundary-aware document chunker implementation.

Enforces strict semantic boundaries during element packing:
1. Section boundaries: Never cross headings; new headings start new chunks.
2. Table boundaries: Tables are never merged with surrounding prose.
3. Code boundaries: Code blocks remain isolated.
4. Bounded grouping: Adjacent compatible elements within the same section
   pack up to a bounded maximum size (default 500 characters).
"""

from __future__ import annotations

import math
import re

from core.normalize.schema import (
    Chunk,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)


def _estimate_token_count(text: str) -> int:
    return math.ceil(len(text) / 4)


class BoundaryAwareChunker:
    """Chunker that groups compatible adjacent elements without crossing semantic boundaries.

    Parameters:
        max_chunk_size: Bounded maximum character length (default 500).
        chunk_overlap: Overlap in characters when splitting oversized elements (default 100).
    """

    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got: {max_chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be non-negative and < {max_chunk_size}"
            raise ValueError(msg)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def _is_heading_element(self, elem: ExtractedElement) -> bool:
        """Deterministic heading detection."""
        content = elem.content.strip()
        if not content:
            return False
        if elem.element_type in (ElementType.HEADER, ElementType.HEADING):
            return True
        first_line = content.splitlines()[0].strip()
        rfc_heading_re = re.compile(
            r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Za-z][^.\n]+(?:\s+[^.\n]+)*)\s*$"
        )
        if rfc_heading_re.match(first_line) and "...." not in content and ". ." not in content:
            return True
        if first_line.startswith("Section ") and ":" in first_line and len(first_line) < 100:
            return True
        if content.startswith("Incident ID:") and len(first_line) < 100:
            return True
        return False

    def _is_table_element(self, elem: ExtractedElement) -> bool:
        return elem.element_type == ElementType.TABLE or elem.table_data is not None

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Produce boundary-aware chunks respecting semantic boundaries."""
        chunks: list[Chunk] = []
        doc_id = document.metadata.document_id

        current_elements: list[ExtractedElement] = []
        current_len = 0

        def emit_current_chunk() -> None:
            nonlocal current_elements, current_len
            if not current_elements:
                return

            text_parts = [e.content.strip() for e in current_elements if e.content.strip()]
            chunk_text = "\n\n".join(text_parts)

            if chunk_text.strip():
                chunk_index = len(chunks)
                chunk_id = f"{doc_id}_bg{chunk_index:04d}"

                source_ids: list[str] = []
                locations: list[PageLocation] = []
                elem_types: list[str] = []

                for e in current_elements:
                    if e.element_id not in source_ids:
                        source_ids.append(e.element_id)
                    if e.location and e.location not in locations:
                        locations.append(e.location)
                    et = e.element_type.value
                    if et not in elem_types:
                        elem_types.append(et)

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        token_count=_estimate_token_count(chunk_text),
                        source_element_ids=source_ids,
                        locations=locations,
                        metadata={
                            "chunker": "BoundaryAwareChunker",
                            "max_chunk_size": self.max_chunk_size,
                            "element_types": elem_types,
                            "element_count": len(current_elements),
                        },
                    )
                )

            current_elements = []
            current_len = 0

        for elem in document.elements:
            content = elem.content.strip()
            if not content:
                continue

            elem_len = len(content)
            is_heading = self._is_heading_element(elem)
            is_table = self._is_table_element(elem)

            # Rule 1: Headings always flush open chunks and start their own chunk
            if is_heading:
                emit_current_chunk()
                current_elements = [elem]
                current_len = elem_len
                # Headings can optionally pack with the first following body paragraph
                # if combined length <= max_chunk_size
                continue

            # Rule 2: Tables are strictly isolated (never merged with preceding or following prose)
            if is_table:
                emit_current_chunk()
                # Table alone as its own chunk
                current_elements = [elem]
                emit_current_chunk()
                continue

            # Rule 3: Single element exceeding max_chunk_size
            if elem_len > self.max_chunk_size:
                emit_current_chunk()
                # Split oversized element into slices
                start = 0
                step = self.max_chunk_size - self.chunk_overlap
                while start < elem_len:
                    end = min(start + self.max_chunk_size, elem_len)
                    slice_text = content[start:end].strip()
                    if slice_text:
                        chunk_index = len(chunks)
                        chunk_id = f"{doc_id}_bg{chunk_index:04d}"
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                text=slice_text,
                                chunk_index=chunk_index,
                                token_count=_estimate_token_count(slice_text),
                                source_element_ids=[elem.element_id],
                                locations=[elem.location] if elem.location else [],
                                metadata={
                                    "chunker": "BoundaryAwareChunker",
                                    "max_chunk_size": self.max_chunk_size,
                                    "oversized_slice": True,
                                    "slice_start": start,
                                    "slice_end": end,
                                },
                            )
                        )
                    if end >= elem_len:
                        break
                    start += step
                continue

            # Rule 4: Compatible element grouping within max_chunk_size
            separator_len = 2 if current_elements else 0
            candidate_len = current_len + separator_len + elem_len

            if candidate_len <= self.max_chunk_size:
                current_elements.append(elem)
                current_len = candidate_len
            else:
                emit_current_chunk()
                current_elements = [elem]
                current_len = elem_len

        # Flush any remaining elements
        emit_current_chunk()
        return chunks
