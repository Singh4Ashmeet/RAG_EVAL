"""Identity query transformer - returns the original query unchanged."""

from __future__ import annotations

from core.query_transform.base import QuerySet


class IdentityTransformer:
    """Returns the original query unchanged - the baseline transformation."""

    @property
    def name(self) -> str:
        return "identity"

    def transform(self, query: str) -> QuerySet:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return QuerySet(
            original=query.strip(),
            transformed=(),
            strategy="identity",
            metadata={},
        )
