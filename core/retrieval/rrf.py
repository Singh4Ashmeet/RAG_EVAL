"""Dependency-light Reciprocal Rank Fusion shared by retrieval paths."""

from __future__ import annotations

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult


def _rrf_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 10,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Combine result lists using Reciprocal Rank Fusion (RRF).

    This function intentionally imports only retrieval schemas. In particular,
    it must remain independent of optional SPLADE/HyDE implementations so the
    hosted dense-plus-BM25 Ask path does not load their ML dependencies.
    """
    if weights is not None and len(weights) != len(results_list):
        raise RetrievalError("weights must have the same length as results_list.")

    effective_weights = weights if weights is not None else [1.0] * len(results_list)

    # Build rank maps (1-indexed ranks) for each result list.
    rank_maps: list[dict[str, int]] = []
    for results in results_list:
        ranks = {r.chunk.chunk_id: r.rank + 1 for r in results}
        rank_maps.append(ranks)

    # All unique chunk IDs (sorted for deterministic iteration).
    all_chunk_ids: list[str] = sorted({cid for ranks in rank_maps for cid in ranks})

    # Get chunk objects from the first result list that contains them.
    chunk_map: dict[str, Chunk] = {}
    for results in results_list:
        for result in results:
            if result.chunk.chunk_id not in chunk_map:
                chunk_map[result.chunk.chunk_id] = result.chunk

    rrf_scores: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        for index, ranks in enumerate(rank_maps):
            if chunk_id in ranks:
                score += effective_weights[index] / (k + ranks[chunk_id])
        if score > 0:
            rrf_scores.append((chunk_id, score))

    # Sort by RRF score descending, breaking ties deterministically by chunk ID.
    rrf_scores.sort(key=lambda item: (-item[1], item[0]))

    return [
        SearchResult(
            chunk=chunk_map[chunk_id],
            score=score,
            rank=rank,
            strategy=RetrievalStrategy.HYBRID,
        )
        for rank, (chunk_id, score) in enumerate(rrf_scores[:top_k])
    ]
