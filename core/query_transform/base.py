"""Base interfaces for query transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class QuerySet:
    """A set of queries derived from a single original query."""

    original: str
    transformed: tuple[str, ...] = field(default_factory=tuple)
    strategy: str = "identity"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.original or not self.original.strip():
            raise ValueError("Original query cannot be empty")
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in self.transformed:
            if q and q.strip() and q not in seen:
                seen.add(q)
                unique.append(q.strip())
        object.__setattr__(self, "transformed", tuple(unique))

    def all_queries(self) -> tuple[str, ...]:
        """Return all queries including the original."""
        if not self.transformed:
            return (self.original,)
        # Ensure original is first
        if self.transformed and self.transformed[0] == self.original:
            return self.transformed
        return (self.original,) + self.transformed


@runtime_checkable
class QueryTransformer(Protocol):
    """Protocol for transforming a user query into a set of retrieval queries."""

    def transform(self, query: str) -> QuerySet:
        """Transform a user query into a set of retrieval queries.

        Args:
            query: The original user query.

        Returns:
            A QuerySet containing the original and any transformed queries.
        """
        ...

    @property
    def name(self) -> str:
        """Unique identifier for this transformer."""
        ...


class TransformerError(Exception):
    """Raised when query transformation fails."""

    pass
