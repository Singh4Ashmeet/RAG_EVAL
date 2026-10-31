"""Storage contract for persistent chunk embeddings."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.normalize.schema import Chunk


@runtime_checkable
class VectorStore(Protocol):
    """Store canonical chunks with embeddings and search them by cosine similarity."""

    def replace_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        model_id: str,
        dimension: int,
    ) -> None:
        """Atomically replace all indexed chunks belonging to one document."""
        ...

    def search(
        self,
        embedding: list[float],
        *,
        model_id: str,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Return matching chunks ordered by descending cosine similarity."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Delete indexed chunks for a document and return the number removed."""
        ...
