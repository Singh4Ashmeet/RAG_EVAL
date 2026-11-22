"""Structural metadata fusion strategies for ranking optimization."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from core.retrieval.schema import RetrievalStrategy, SearchResult
from core.retrieval.structural.scorer import StructuralScorer


class StructuralFusionStrategy(StrEnum):
    """Available fusion mechanisms for structural metadata scoring."""

    NONE = "none"
    ADDITIVE = "additive"
    NORMALIZED_ADDITIVE = "normalized_additive"
    RANK_AWARE_BOOST = "rank_aware_boost"
    TIE_BREAK = "tie_break"
    FILTER = "filter"


def fuse_structural(
    candidates: list[SearchResult],
    query: str,
    scorer: StructuralScorer,
    strategy: StructuralFusionStrategy | str = StructuralFusionStrategy.NONE,
    alpha: float = 0.10,
    margin_threshold: float = 0.001,
    top_k: int = 5,
    custom_weights: dict[str, float] | None = None,
) -> tuple[list[SearchResult], dict[str, Any]]:
    """Apply structural fusion to an ordered list of RRF candidates.

    Args:
        candidates: Initial search candidates (e.g. top-20 from Dense + BM25 RRF).
        query: User query text.
        scorer: StructuralScorer instance.
        strategy: Fusion mechanism to apply.
        alpha: Weight for structural score in additive strategies.
        margin_threshold: Score delta threshold for Strategy B (rank-aware boost).
        top_k: Number of final results to return.
        custom_weights: Optional signal weights override.

    Returns:
        tuple of (fused_results, diagnostic_info).
    """
    if isinstance(strategy, str):
        try:
            strategy = StructuralFusionStrategy(strategy.lower())
        except ValueError:
            strategy = StructuralFusionStrategy.NONE

    if not candidates or strategy == StructuralFusionStrategy.NONE or alpha == 0.0:
        # Return exact baseline behavior unchanged
        baseline_slice = candidates[:top_k]
        return [
            SearchResult(
                chunk=r.chunk,
                score=r.score,
                rank=i,
                strategy=r.strategy,
            )
            for i, r in enumerate(baseline_slice)
        ], {"strategy": "none", "modifications": 0}

    # Score each candidate chunk
    scored_candidates: list[tuple[SearchResult, float, dict[str, Any]]] = []
    max_rrf = max((r.score for r in candidates), default=1.0)
    if max_rrf <= 0.0:
        max_rrf = 1.0

    for r in candidates:
        s_res = scorer.score(query, r.chunk)
        scored_candidates.append((r, s_res.total_score, s_res.as_dict()))

    diagnostics: dict[str, Any] = {
        "strategy": strategy.value,
        "alpha": alpha,
        "margin_threshold": margin_threshold,
        "candidate_count": len(candidates),
        "scores": {r.chunk.chunk_id: d for r, _, d in scored_candidates},
    }

    # Apply specific fusion strategy
    if strategy == StructuralFusionStrategy.ADDITIVE:
        # Strategy A: Post-RRF additive scoring (exact formula from Section 7)
        # final_score = rrf_score + alpha * structural_score
        re_ranked = []
        for r, s_score, _ in scored_candidates:
            final_s = r.score + alpha * s_score
            re_ranked.append((final_s, r.score, s_score, r))
        # Sort descending by final_s, tie-break by chunk_id
        re_ranked.sort(key=lambda x: (-x[0], x[3].chunk.chunk_id))
        final_candidates = [item[3] for item in re_ranked]

    elif strategy == StructuralFusionStrategy.NORMALIZED_ADDITIVE:
        # Strategy A variant: normalized RRF in [0, 1] + alpha * structural_score
        re_ranked = []
        for r, s_score, _ in scored_candidates:
            norm_rrf = r.score / max_rrf
            final_s = norm_rrf + alpha * s_score
            re_ranked.append((final_s, r.score, s_score, r))
        re_ranked.sort(key=lambda x: (-x[0], x[3].chunk.chunk_id))
        final_candidates = [item[3] for item in re_ranked]

    elif strategy == StructuralFusionStrategy.RANK_AWARE_BOOST:
        # Strategy B: Only re-order within candidate clusters whose RRF scores
        # are within margin_threshold of each other.
        # This prevents structural metadata from overpowering strong lexical/semantic matches.
        clusters: list[list[tuple[SearchResult, float]]] = []
        current_cluster: list[tuple[SearchResult, float]] = []

        for r, s_score, _ in scored_candidates:
            if not current_cluster:
                current_cluster.append((r, s_score))
            else:
                prev_r, _ = current_cluster[-1]
                if abs(prev_r.score - r.score) <= margin_threshold:
                    current_cluster.append((r, s_score))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(r, s_score)]
        if current_cluster:
            clusters.append(current_cluster)

        re_ranked_list: list[SearchResult] = []
        for cluster in clusters:
            if len(cluster) == 1:
                re_ranked_list.append(cluster[0][0])
            else:
                # Within cluster, sort by structural score descending
                sorted_cluster = sorted(
                    cluster,
                    key=lambda x: (-x[1], -x[0].score, x[0].chunk.chunk_id),
                )
                re_ranked_list.extend([item[0] for item in sorted_cluster])
        final_candidates = re_ranked_list

    elif strategy == StructuralFusionStrategy.TIE_BREAK:
        # Strategy C: Structural tie-break
        # Only re-order when RRF scores are effectively tied (delta <= 1e-6)
        clusters_c: list[list[tuple[SearchResult, float]]] = []
        curr_c: list[tuple[SearchResult, float]] = []

        for r, s_score, _ in scored_candidates:
            if not curr_c:
                curr_c.append((r, s_score))
            else:
                prev_r, _ = curr_c[-1]
                if abs(prev_r.score - r.score) <= 1e-6:
                    curr_c.append((r, s_score))
                else:
                    clusters_c.append(curr_c)
                    curr_c = [(r, s_score)]
        if curr_c:
            clusters_c.append(curr_c)

        re_ranked_c: list[SearchResult] = []
        for cluster in clusters_c:
            if len(cluster) == 1:
                re_ranked_c.append(cluster[0][0])
            else:
                sorted_c = sorted(
                    cluster,
                    key=lambda x: (-x[1], -x[0].score, x[0].chunk.chunk_id),
                )
                re_ranked_c.extend([item[0] for item in sorted_c])
        final_candidates = re_ranked_c

    elif strategy == StructuralFusionStrategy.FILTER:
        # Strategy D: Structural candidate filtering
        # Filter out candidates with s_score == 0.0 ONLY IF at least top_k candidates remain
        non_zero = [r for r, s_score, _ in scored_candidates if s_score > 0.0]
        if len(non_zero) >= top_k:
            final_candidates = non_zero
        else:
            final_candidates = [r for r, _, _ in scored_candidates]

    else:
        final_candidates = [r for r, _, _ in scored_candidates]

    # Build final top_k results
    out_results: list[SearchResult] = []
    modifications = 0
    orig_top_k_ids = [r.chunk.chunk_id for r in candidates[:top_k]]

    for rank, item in enumerate(final_candidates[:top_k]):
        if rank < len(orig_top_k_ids) and item.chunk.chunk_id != orig_top_k_ids[rank]:
            modifications += 1
        out_results.append(
            SearchResult(
                chunk=item.chunk,
                score=item.score,
                rank=rank,
                strategy=RetrievalStrategy.HYBRID,
            )
        )

    diagnostics["modifications"] = modifications
    return out_results, diagnostics
