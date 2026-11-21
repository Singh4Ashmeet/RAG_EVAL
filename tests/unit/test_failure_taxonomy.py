"""Tests for evaluation failure taxonomy and failure classification."""

from __future__ import annotations

from eval.datasets.schema import (
    AnnotationProvenance,
    Answerability,
    Difficulty,
    EvaluationCase,
    QuestionCategory,
)
from eval.failure_taxonomy import FailureType, classify_query_failure


def _make_case(
    query_id: str = "q1",
    answerable: bool = True,
    relevant_chunks: list[str] | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        query_id=query_id,
        query="Test query?",
        relevant_chunk_ids=relevant_chunks if relevant_chunks is not None else ["c1"],
        relevant_element_ids=["e1"],
        reference_answer="Test reference answer." if answerable else None,
        status="verified",
        answerability=Answerability.ANSWERABLE if answerable else Answerability.UNANSWERABLE,
        answerable=answerable,
        category=QuestionCategory.DIRECT_FACTUAL if answerable else QuestionCategory.UNANSWERABLE,
        difficulty=Difficulty.EASY,
        provenance=AnnotationProvenance(
            annotator="test",
            verified_at="2026-11-16T00:00:00Z",
            source_documents=["doc1.txt"],
        ),
        track="track_a",
    )


def test_failure_type_enum_values() -> None:
    assert FailureType.NONE == "none"
    assert FailureType.RETRIEVAL_FAILURE == "retrieval_failure"
    assert FailureType.RANKING_FAILURE == "ranking_failure"
    assert FailureType.GENERATION_FAILURE == "generation_failure"
    assert FailureType.ABSTENTION_FAILURE == "abstention_failure"
    assert FailureType.ANNOTATION_FAILURE == "annotation_failure"
    assert FailureType.SYSTEM_FAILURE == "system_failure"


def test_classify_system_failure() -> None:
    case = _make_case()
    res = classify_query_failure(
        case=case,
        errors=["ConnectionRefusedError: Ollama daemon not reachable"],
        retrieval_metrics={},
        answer_metrics={},
        generated_answer=None,
    )
    assert res == FailureType.SYSTEM_FAILURE


def test_classify_abstention_failure() -> None:
    case = _make_case(answerable=False, relevant_chunks=[])
    # Model gave an answer to an unanswerable query -> ABSTENTION_FAILURE
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={},
        answer_metrics={},
        generated_answer="The framework uses Kafka 0.8 adapter with 10MB payload.",
    )
    assert res == FailureType.ABSTENTION_FAILURE

    # Model correctly abstained -> NONE
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={},
        answer_metrics={},
        generated_answer="I cannot answer this based on the provided documents.",
    )
    assert res == FailureType.NONE


def test_classify_annotation_failure() -> None:
    # Query is marked answerable but has no ground truth relevant chunks
    case = _make_case(answerable=True, relevant_chunks=[])
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"recall@5": 0.0, "hit_rate@5": 0.0},
        answer_metrics={"token_f1": 0.8},
        generated_answer="Valid answer",
    )
    assert res == FailureType.ANNOTATION_FAILURE


def test_classify_retrieval_failure() -> None:
    case = _make_case(answerable=True, relevant_chunks=["c1"])
    # Relevant chunk was not retrieved at all in top-k
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"recall@5": 0.0, "hit_rate@5": 0.0},
        answer_metrics={"token_f1": 0.0},
        generated_answer="Irrelevant answer",
    )
    assert res == FailureType.RETRIEVAL_FAILURE


def test_classify_ranking_failure() -> None:
    case = _make_case(answerable=True, relevant_chunks=["c1"])
    # Hit at rank 5, but hit_rate@1 is 0.0, and generated answer is poor (low F1)
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"hit_rate@1": 0.0, "hit_rate@5": 1.0, "recall@5": 1.0},
        answer_metrics={"token_f1": 0.1},
        generated_answer="Partially relevant but mostly wrong",
    )
    assert res == FailureType.RANKING_FAILURE


def test_classify_generation_failure() -> None:
    case = _make_case(answerable=True, relevant_chunks=["c1"])
    # Perfect retrieval at rank 1, but generation F1 is below threshold
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"hit_rate@1": 1.0, "hit_rate@5": 1.0, "recall@5": 1.0},
        answer_metrics={"token_f1": 0.15},
        generated_answer="Completely hallucinated explanation",
    )
    assert res == FailureType.GENERATION_FAILURE

    # Or model abstained on an answerable query
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"hit_rate@1": 1.0, "hit_rate@5": 1.0, "recall@5": 1.0},
        answer_metrics={"token_f1": 0.0},
        generated_answer="I cannot answer this question based on the text.",
    )
    assert res == FailureType.GENERATION_FAILURE


def test_classify_success_case() -> None:
    case = _make_case(answerable=True, relevant_chunks=["c1"])
    # Perfect retrieval at rank 1, strong F1 score
    res = classify_query_failure(
        case=case,
        errors=[],
        retrieval_metrics={"hit_rate@1": 1.0, "hit_rate@5": 1.0, "recall@5": 1.0},
        answer_metrics={"token_f1": 0.85},
        generated_answer="Test reference answer.",
    )
    assert res == FailureType.NONE
