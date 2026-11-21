"""HyDE retriever that uses hypothetical passages for dense retrieval."""

from __future__ import annotations

import logging
from typing import Any

from core.hyde.transformer import HyDETransformer
from core.retrieval.base import Retriever
from core.retrieval.schema import RetrievalStrategy, SearchResult

logger = logging.getLogger(__name__)


class HyDERetriever(Retriever):
    """
    HyDE retriever that generates hypothetical passages and uses them for dense retrieval.

    This retriever generates a hypothetical passage from the query and uses it
    as the retrieval representation instead of (or in addition to) the original query.
    """

    def __init__(
        self,
        dense_retriever,
        hyde_transformer: HyDETransformer,
        *,
        use_original_query: bool = True,
        top_k: int = 10,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.hyde_transformer = hyde_transformer
        self.use_original_query = use_original_query
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve using HyDE-generated hypothetical passage."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        try:
            # Generate HyDE passage
            hyde_passage = self.hyde_transformer.transform(query)

            # Retrieve using the HyDE passage
            hyde_results = self.dense_retriever.retrieve(hyde_passage, top_k=top_k)

            # Mark results as from HyDE
            for result in hyde_results:
                result.strategy = RetrievalStrategy.HYDE

            # Optionally also retrieve with original query and fuse
            # (This is handled by the HybridRetriever for fusion)
            return hyde_results[:top_k]

        except Exception as exc:
            raise ValueError(f"HyDE retrieval failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return retriever statistics."""
        stats = {
            "use_original_query": self.use_original_query,
            "top_k": self.top_k,
            "hyde_transformer": {
                "model_name": self.hyde_transformer.config.model_name,
                "temperature": self.hyde_transformer.config.temperature,
            },
        }
        if hasattr(self.dense_retriever, "get_stats"):
            stats["dense_retriever"] = self.dense_retriever.get_stats()
        return stats
