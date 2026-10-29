"""Base chunker interfaces and protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.normalize.schema import Chunk, NormalizedDocument


@runtime_checkable
class DocumentChunker(Protocol):
    """Protocol for chunking algorithms segmenting NormalizedDocuments into Chunks."""

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Segment a NormalizedDocument into discrete Chunks.

        Returns chunks in the order defined by the implementation.

        Args:
            document: Canonical NormalizedDocument to segment.

        Returns:
            A list of Chunks linked to the parent document and source elements.
        """
        ...
