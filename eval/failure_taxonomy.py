"""Deterministic failure taxonomy classification for evaluation query results."""

from __future__ import annotations

from enum import StrEnum

from eval.datasets.schema import Answerability, EvaluationCase
from eval.metrics.abstention import is_abstention


class FailureType(StrEnum):
    """Semantic category of an evaluation failure."""

    NONE = "none"
    RETRIEVAL_FAILURE = "retrieval_failure"
    RANKING_FAILURE = "ranking_failure"
    GENERATION_FAILURE = "generation_failure"
    ABSTENTION_FAILURE = "abstention_failure"
    ANNOTATION_FAILURE = "annotation_failure"
    SYSTEM_FAILURE = "system_failure"


def classify_query_failure(
    case: EvaluationCase,
    errors: list[str],
    retrieval_metrics: dict[str, float],
    answer_metrics: dict[str, float],
    generated_answer: str | None,
    max_cutoff: int = 5,
    low_token_f1_threshold: float = 0.3,
) -> FailureType:
    """Deterministically classify an evaluation case result into a failure category.

    Args:
        case: The golden evaluation case.
        errors: Errors encountered during retrieval or generation.
        retrieval_metrics: Computed retrieval metrics (recall@k, hit_rate@k, etc.).
        answer_metrics: Computed generation metrics (token_f1, exact_match).
        generated_answer: The generated answer string, if generation was run.
        max_cutoff: The maximum cutoff rank used in retrieval.
        low_token_f1_threshold: Token F1 threshold below which answer is deemed poor.

    Returns:
        The failure category (FailureType.NONE if successful).
    """
    # 1. System / infrastructure failures
    system_errors = [e for e in errors if not e.startswith("answer metrics unavailable:")]
    if system_errors:
        return FailureType.SYSTEM_FAILURE

    # 2. Unanswerable query evaluation
    is_unanswerable = case.answerability == Answerability.UNANSWERABLE or (case.answerable is False)
    if is_unanswerable:
        if generated_answer is not None:
            abstained = is_abstention(generated_answer)
            if not abstained:
                return FailureType.ABSTENTION_FAILURE
        return FailureType.NONE

    # 3. Annotation sanity check (answerable query missing ground truth chunks or reference answer)
    if not case.relevant_chunk_ids or not case.reference_answer:
        return FailureType.ANNOTATION_FAILURE

    # 4. Retrieval failure check (relevant evidence not in top-K)
    if case.relevant_chunk_ids:
        max_recall = retrieval_metrics.get(f"recall@{max_cutoff}", 0.0)
        max_hit = retrieval_metrics.get(f"hit_rate@{max_cutoff}", 0.0)
        if max_recall == 0.0 and max_hit == 0.0:
            return FailureType.RETRIEVAL_FAILURE

        # 5. Ranking failure check (retrieved within top-K, but missed rank 1)
        hit_1 = retrieval_metrics.get("hit_rate@1", 0.0)
        if (max_hit == 1.0 or max_recall > 0.0) and hit_1 == 0.0:
            return FailureType.RANKING_FAILURE

    # 6. Generation failure check (evidence retrieved, but generated answer is poor)
    token_f1 = answer_metrics.get("token_f1")
    if token_f1 is not None and token_f1 < low_token_f1_threshold:
        return FailureType.GENERATION_FAILURE

    return FailureType.NONE
