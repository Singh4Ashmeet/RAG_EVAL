"""Element-centric document chunker implementations.

Segment NormalizedDocuments into fine-grained, semantically atomic retrieval units:
1. ElementAtomicChunker: Exactly one semantic element per retrieval unit.
2. ElementHeadingAttachedChunker: Attaches immediate parent heading context to each atomic unit.
"""

from __future__ import annotations

import math
import re

from core.normalize.schema import Chunk, ElementType, NormalizedDocument


def _estimate_token_count(text: str) -> int:
    """Deterministic token count estimation matching production chunker (char / 4)."""
    return math.ceil(len(text) / 4)


class ElementAtomicChunker:
    """Chunker that turns each semantic ExtractedElement into an atomic retrieval unit.

    Does not merge adjacent elements together, preserving semantic boundaries.
    Oversized elements exceeding max_chunk_size are split cleanly.
    """

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got: {max_chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be non-negative and < {max_chunk_size}"
            raise ValueError(msg)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Produce atomic chunks, one per non-empty extracted element."""
        chunks: list[Chunk] = []
        doc_id = document.metadata.document_id

        for elem in document.elements:
            content = elem.content.strip()
            if not content:
                continue

            locations = [elem.location] if elem.location else []

            if len(content) <= self.max_chunk_size:
                chunk_index = len(chunks)
                chunk_id = f"{doc_id}_ea{chunk_index:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=content,
                        chunk_index=chunk_index,
                        token_count=_estimate_token_count(content),
                        source_element_ids=[elem.element_id],
                        locations=locations,
                        metadata={
                            "chunker": "ElementAtomicChunker",
                            "element_type": elem.element_type.value,
                            "original_element_id": elem.element_id,
                        },
                    )
                )
            else:
                # Oversized element: split into sequential slices with overlap
                start = 0
                step = self.max_chunk_size - self.chunk_overlap
                while start < len(content):
                    end = min(start + self.max_chunk_size, len(content))
                    slice_text = content[start:end].strip()
                    if slice_text:
                        chunk_index = len(chunks)
                        chunk_id = f"{doc_id}_ea{chunk_index:04d}"
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                text=slice_text,
                                chunk_index=chunk_index,
                                token_count=_estimate_token_count(slice_text),
                                source_element_ids=[elem.element_id],
                                locations=locations,
                                metadata={
                                    "chunker": "ElementAtomicChunker",
                                    "element_type": elem.element_type.value,
                                    "original_element_id": elem.element_id,
                                    "slice_start": start,
                                    "slice_end": end,
                                },
                            )
                        )
                    if end >= len(content):
                        break
                    start += step

        return chunks


class ElementHeadingAttachedChunker:
    """Atomic element chunker with deterministic parent heading association.

    For each atomic element, attaches the immediate parent heading / section title
    to provide local hierarchy without cross-element content pollution.
    """

    def __init__(self, max_chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Produce atomic chunks with active heading context prepended."""
        chunks: list[Chunk] = []
        doc_id = document.metadata.document_id

        active_heading: str | None = None
        active_heading_elem_id: str | None = None

        rfc_heading_re = re.compile(
            r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Za-z][^.\n]+(?:\s+[^.\n]+)*)\s*$"
        )

        for elem in document.elements:
            content = elem.content.strip()
            if not content:
                continue

            # Detect if this element is a heading
            is_heading = False
            first_line = content.splitlines()[0].strip()

            if elem.element_type in (ElementType.HEADER, ElementType.HEADING):
                is_heading = True
            elif first_line:
                m = rfc_heading_re.match(first_line)
                if m and "...." not in content and ". ." not in content:
                    is_heading = True
                elif (
                    first_line.startswith("Section ")
                    and ":" in first_line
                    and len(first_line) < 100
                ):
                    is_heading = True
                elif content.startswith("Incident ID:") and len(first_line) < 100:
                    is_heading = True

            if is_heading:
                active_heading = content
                active_heading_elem_id = elem.element_id
                # Headings are also emitted as atomic chunks
                chunk_index = len(chunks)
                chunk_id = f"{doc_id}_eh{chunk_index:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=content,
                        chunk_index=chunk_index,
                        token_count=_estimate_token_count(content),
                        source_element_ids=[elem.element_id],
                        locations=[elem.location] if elem.location else [],
                        metadata={
                            "chunker": "ElementHeadingAttachedChunker",
                            "element_type": elem.element_type.value,
                            "is_heading": True,
                        },
                    )
                )
                continue

            # Non-heading body element: attach active heading if available
            if active_heading:
                attached_text = f"{active_heading}\n\n{content}"
                source_ids = (
                    [active_heading_elem_id, elem.element_id]
                    if active_heading_elem_id
                    else [elem.element_id]
                )
            else:
                attached_text = content
                source_ids = [elem.element_id]

            # Deduplicate source_ids in order
            seen_ids: set[str] = set()
            clean_source_ids: list[str] = []
            for x in source_ids:
                if x and x not in seen_ids:
                    seen_ids.add(x)
                    clean_source_ids.append(x)

            locations = [elem.location] if elem.location else []

            if len(attached_text) <= self.max_chunk_size:
                chunk_index = len(chunks)
                chunk_id = f"{doc_id}_eh{chunk_index:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=attached_text,
                        chunk_index=chunk_index,
                        token_count=_estimate_token_count(attached_text),
                        source_element_ids=clean_source_ids,
                        locations=locations,
                        metadata={
                            "chunker": "ElementHeadingAttachedChunker",
                            "element_type": elem.element_type.value,
                            "parent_heading": active_heading,
                        },
                    )
                )
            else:
                # Oversized: split body while keeping heading attached to each slice
                body_max = self.max_chunk_size - (len(active_heading) + 2 if active_heading else 0)
                body_max = max(body_max, 200)
                step = body_max - self.chunk_overlap
                start = 0
                while start < len(content):
                    end = min(start + body_max, len(content))
                    body_slice = content[start:end].strip()
                    if body_slice:
                        slice_text = (
                            f"{active_heading}\n\n{body_slice}" if active_heading else body_slice
                        )
                        chunk_index = len(chunks)
                        chunk_id = f"{doc_id}_eh{chunk_index:04d}"
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                text=slice_text,
                                chunk_index=chunk_index,
                                token_count=_estimate_token_count(slice_text),
                                source_element_ids=clean_source_ids,
                                locations=locations,
                                metadata={
                                    "chunker": "ElementHeadingAttachedChunker",
                                    "element_type": elem.element_type.value,
                                    "parent_heading": active_heading,
                                    "slice_start": start,
                                    "slice_end": end,
                                },
                            )
                        )
                    if end >= len(content):
                        break
                    start += step

        return chunks
