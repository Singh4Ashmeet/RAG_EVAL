"""Base embedder interfaces and protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models generating dense vector representations."""

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality (e.g. 384, 768, 1536)."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a list of document texts.

        Args:
            texts: List of string passages to embed.

        Returns:
            List of embedding vectors, one float vector per input text.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a search query string.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...
