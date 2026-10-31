"""Known-answer tests for retrieval ranking and answer text metrics."""

from __future__ import annotations

import pytest

from eval.metrics import (
    average_precision_at_k,
    exact_match,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    token_f1,
)


def test_retrieval_metrics_match_hand_calculated_rankings() -> None:
    retrieved = ["irrelevant", "relevant-a", "relevant-b"]
    relevant = ["relevant-a", "relevant-b"]

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert average_precision_at_k(retrieved, relevant, 2) == pytest.approx(0.25)
    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0

    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert average_precision_at_k(retrieved, relevant, 3) == pytest.approx(7 / 12)


def test_precision_denominator_remains_k_when_fewer_results_are_returned() -> None:
    assert precision_at_k(["relevant"], ["relevant"], 5) == pytest.approx(0.2)


def test_duplicate_results_do_not_inflate_retrieval_metrics() -> None:
    assert precision_at_k(["a", "a", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "a"], ["a", "b"], 2) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_retrieval_metrics_require_positive_integer_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["a"], ["a"], k)


def test_retrieval_metrics_require_relevant_ground_truth() -> None:
    with pytest.raises(ValueError, match="At least one relevant"):
        hit_rate_at_k(["a"], [], 1)


def test_exact_match_normalizes_unicode_case_punctuation_and_citations() -> None:
    assert exact_match("Café is blue! [S1]", "CAFÉ is blue.") == 1.0
    assert exact_match("blue sky", "sky blue") == 0.0


def test_token_f1_uses_multiset_overlap() -> None:
    assert token_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)
    assert token_f1("entirely unrelated", "expected phrase") == 0.0
    assert token_f1("", "expected phrase") == 0.0
