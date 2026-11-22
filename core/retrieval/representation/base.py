"""Abstract base and protocol definitions for chunk representations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.normalize.schema import Chunk, NormalizedDocument


@runtime_checkable
class ChunkRepresentation(Protocol):
    """Protocol for transforming a canonical Chunk into a retrieval representation.

    Implementations must be deterministic, preserve the original chunk text,
    and derive any structural context strictly from the NormalizedDocument metadata
    without inventing content or calling generative models.
    """

    @property
    def name(self) -> str:
        """Stable identifier for the representation variant."""
        ...

    def transform(
        self,
        chunk: Chunk,
        document: NormalizedDocument | None = None,
    ) -> str:
        """Transform a canonical Chunk into an enriched representation string.

        Args:
            chunk: Canonical Chunk instance.
            document: Optional parent NormalizedDocument containing element metadata.

        Returns:
            A deterministic string representation for indexing/embedding.
        """
        ...
