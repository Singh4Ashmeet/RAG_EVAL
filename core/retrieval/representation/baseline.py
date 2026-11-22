"""Baseline representation returning raw chunk text without enrichment."""

from __future__ import annotations

from core.normalize.schema import Chunk, NormalizedDocument


class BaselineRepresentation:
    """Production baseline representation that uses raw chunk text unmodified."""

    @property
    def name(self) -> str:
        return "baseline"

    def transform(
        self,
        chunk: Chunk,
        document: NormalizedDocument | None = None,
    ) -> str:
        """Return the unmodified text of the canonical chunk."""
        return chunk.text
