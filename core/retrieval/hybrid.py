"""Hybrid retrieval combining dense, sparse (BM25), learned sparse (SPLADE),
and HyDE retrieval with RRF."""

from __future__ import annotations

from typing import Any, Literal

from core.errors import RetrievalError
from core.hyde.retriever import HyDERetriever
from core.retrieval.base import Retriever
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.schema import RetrievalStrategy, SearchResult
from core.sparse.retriever import SparseRetriever

FusionMethod = Literal["rrf"]


def _rrf_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF(d) = sum(1 / (k + rank_i(d))) for each ranking i where d appears.
    """
    # Build rank maps (1-indexed ranks) for each result list
    rank_maps: list[dict[str, int]] = []
    for results in results_list:
        ranks = {r.chunk.chunk_id: r.rank + 1 for r in results}
        rank_maps.append(ranks)

    # All unique chunk IDs
    all_chunk_ids: set[str] = set()
    for ranks in rank_maps:
        all_chunk_ids.update(ranks.keys())

    # Get chunk objects (from first result list that has it)
    chunk_map = {}
    for results in results_list:
        for r in results:
            if r.chunk.chunk_id not in chunk_map:
                chunk_map[r.chunk.chunk_id] = r.chunk

    # Compute RRF scores
    rrf_scores = []
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


class HybridRetriever(Retriever):
    """Hybrid retriever combining dense, BM25, SPLADE, and HyDE retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        *,
        splade_retriever: SparseRetriever | None = None,
        hyde_retriever: HyDERetriever | None = None,
        fusion_method: FusionMethod = "rrf",
        rrf_k: int = 100,
        candidate_k: int = 20,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.splade_retriever = splade_retriever
        self.hyde_retriever = hyde_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        if fusion_method == "rrf":
            if rrf_k < 1:
                raise RetrievalError("RRF parameter k must be >= 1.")
        else:
            raise RetrievalError(f"Unknown fusion method: {fusion_method}")

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve using hybrid fusion of dense, BM25, SPLADE, and HyDE results."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")

        try:
            # Retrieve larger candidate pools from all retrievers
            results_list: list[list[SearchResult]] = []

            dense_results = self.dense_retriever.retrieve(query, top_k=self.candidate_k)
            results_list.append(dense_results)

            bm25_results = self.bm25_retriever.retrieve(query, top_k=self.candidate_k)
            results_list.append(bm25_results)

            if self.splade_retriever is not None:
                splade_results = self.splade_retriever.retrieve(query, top_k=self.candidate_k)
                results_list.append(splade_results)

            if self.hyde_retriever is not None:
                hyde_results = self.hyde_retriever.retrieve(query, top_k=self.candidate_k)
                results_list.append(hyde_results)

            # Fuse using RRF
            fused = _rrf_fusion(
                results_list,
                k=self.rrf_k,
                top_k=top_k,
            )

            return fused

        except Exception as exc:
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return hybrid retriever statistics."""
        dense_stats = (
            self.dense_retriever.get_stats() if hasattr(self.dense_retriever, "get_stats") else {}
        )
        bm25_stats = (
            self.bm25_retriever.get_stats() if hasattr(self.bm25_retriever, "get_stats") else {}
        )
        splade_stats = (
            self.splade_retriever.get_stats()
            if self.splade_retriever is not None and hasattr(self.splade_retriever, "get_stats")
            else {}
        )
        hyde_stats: dict[str, Any] = (
            self.hyde_retriever.get_stats()
            if self.hyde_retriever is not None and hasattr(self.hyde_retriever, "get_stats")
            else {}
        )
        return {
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k if self.fusion_method == "rrf" else None,
            "candidate_k": self.candidate_k,
            "dense_stats": dense_stats,
            "bm25_stats": bm25_stats,
            "splade_stats": splade_stats,
            "hyde_stats": hyde_stats,
        }
