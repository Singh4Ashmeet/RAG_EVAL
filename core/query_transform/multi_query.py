"""Multi-query retrieval with rank-based fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.normalize.schema import Chunk
from core.query_transform.base import QueryTransformer
from core.retrieval.base import Retriever
from core.retrieval.schema import RetrievalStrategy, SearchResult


class MultiQueryRetriever:
    """Retrieves using multiple queries and fuses results via RRF."""

    def __init__(
        self,
        base_retriever: Retriever,
        query_transformer: QueryTransformer,
        *,
        fusion_k: int = 60,
        max_queries: int = 3,
    ) -> None:
        self.base_retriever = base_retriever
        self.query_transformer = query_transformer
        self.fusion_k = fusion_k
        self.max_queries = max_queries

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve using multiple transformed queries and fuse via RRF."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Get query set from transformer
        query_set = self.query_transformer.transform(query)

        # Limit number of queries to process
        all_queries = query_set.all_queries()[: self.max_queries]

        # Retrieve for each query
        all_results: list[list[SearchResult]] = []
        for q in all_queries:
            results = self.base_retriever.retrieve(q, top_k=top_k * 2)  # Get more for fusion
            all_results.append(results)

        # Fuse results using RRF across all queries
        if len(all_results) == 1:
            return all_results[0][:top_k]

        # RRF fusion across query results
        fused = _multi_query_rrf_fusion(all_results, k=self.fusion_k, top_k=top_k)
        return fused


def _multi_query_rrf_fusion(
    query_results: Sequence[Sequence[SearchResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Fuse results from multiple queries using Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_i(d))) for each query ranking i where d appears.
    """
    # Build rank maps for each query's results
    rank_maps: list[dict[str, int]] = []
    chunk_map: dict[str, Chunk] = {}

    for results in query_results:
        ranks: dict[str, int] = {}
        for rank, result in enumerate(results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in ranks:
                ranks[chunk_id] = rank + 1  # 1-indexed
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result.chunk
        rank_maps.append(ranks)

    # Compute RRF scores
    all_chunk_ids: set[str] = set()
    for ranks in rank_maps:
        all_chunk_ids.update(ranks.keys())

    rrf_scores: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        for ranks in rank_maps:
            if chunk_id in ranks:
                score += 1.0 / (k + ranks[chunk_id])
        if score > 0:
            rrf_scores.append((chunk_id, score))

    # Sort by RRF score descending
    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # Build results
    results = []
    for rank, (chunk_id, score) in enumerate(rrf_scores[:top_k]):
        results.append(
            SearchResult(
                chunk=chunk_map[chunk_id],
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.HYBRID,
            )
        )

    return results


def create_multi_query_retriever(
    base_retriever: Retriever,
    transformer_name: str = "identity",
    **transformer_kwargs: Any,
) -> MultiQueryRetriever:
    """Factory to create a multi-query retriever with a specific transformer."""
    from core.query_transform import (
        IdentityTransformer,
        NormalizeTransformer,
        TechnicalTermExtractor,
    )

    transformers: dict[str, QueryTransformer] = {
        "identity": IdentityTransformer(),
        "normalize": NormalizeTransformer(),
        "technical_terms": TechnicalTermExtractor(),
    }

    if transformer_name not in transformers:
        raise ValueError(f"Unknown transformer: {transformer_name}")

    transformer = transformers[transformer_name]
    return MultiQueryRetriever(base_retriever, transformer, **transformer_kwargs)
