"""Deterministic chunk-level ranking metrics with explicit @k semantics."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of known relevant chunks found in the first k ranks."""
    relevant = _validate_relevant(relevant_ids)
    hits = sum(_relevance_at_k(retrieved_ids, relevant, k))
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Relevant unique results in the first k ranks divided by k.

    The denominator remains k if the retriever returns fewer than k results,
    making missing result slots count as non-relevant.
    """
    relevant = _validate_relevant(relevant_ids)
    return sum(_relevance_at_k(retrieved_ids, relevant, k)) / k


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent in top k."""
    relevant = _validate_relevant(relevant_ids)
    for rank, is_relevant in enumerate(_relevance_at_k(retrieved_ids, relevant, k), start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Average precision through rank k, normalized by min(relevant count, k)."""
    relevant = _validate_relevant(relevant_ids)
    relevance = _relevance_at_k(retrieved_ids, relevant, k)
    hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Return one if at least one relevant chunk appears in the first k ranks."""
    relevant = _validate_relevant(relevant_ids)
    return float(any(_relevance_at_k(retrieved_ids, relevant, k)))


def _validate_relevant(relevant_ids: Sequence[str]) -> set[str]:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("At least one relevant chunk ID is required to compute retrieval metrics.")
    return relevant


def _relevance_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> list[bool]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    seen: set[str] = set()
    relevance: list[bool] = []
    for chunk_id in retrieved_ids[:k]:
        is_new = chunk_id not in seen
        relevance.append(is_new and chunk_id in relevant_ids)
        seen.add(chunk_id)
    return relevance
