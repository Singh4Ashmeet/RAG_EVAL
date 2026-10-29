"""Base retrieval and reranking interfaces and protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.retrieval.schema import SearchResult


@runtime_checkable
class Retriever(Protocol):
    """Protocol for candidate retrieval components (dense, sparse, or hybrid)."""

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve candidate chunks matching the search query.

        Args:
            query: Non-empty search query string (after trimming whitespace).
            top_k: Maximum number of candidate results to return (must be >= 1).

        Returns:
            List of SearchResult candidates ordered by score/rank, up to top_k items.
            Implementations are responsible for honoring the requested top_k limit.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for candidate reranking components (e.g. cross-encoders)."""

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        """Rerank pre-retrieved candidate chunks against the search query.

        Args:
            query: Non-empty search query string (after trimming whitespace).
            candidates: Pre-retrieved candidate SearchResults to rescore and reorder.
            top_k: Maximum number of reranked results to return (must be >= 1).

        Returns:
            List of rescored and reordered SearchResult instances, up to top_k items.
            Implementations are responsible for honoring the requested top_k limit.
        """
        ...
