"""Tests for abstention and hallucination resistance metrics."""

from __future__ import annotations

import pytest

from eval.metrics.abstention import (
    abstention_accuracy,
    false_answer_rate,
    is_abstention,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I do not have enough information to answer this question.", True),
        ("Based on the provided context, the information is not available.", True),
        ("The documents do not contain information about GraphQL.", True),
        ("I cannot answer this based on the provided text.", True),
        ("Insufficient information to answer.", True),
        ("There is no mention of Cassandra in the documentation.", True),
        ("Unknown based on the provided documents.", True),
        ("I am unable to find any information regarding this topic.", True),
        ("JSON provides objects and arrays as structured types.", False),
        ("Tier-1 Gateway has an uptime target of 99.99%.", False),
        ("The media type for JSON is application/json.", False),
        ("", True),
        ("   ", True),
        (None, True),
    ],
)
def test_is_abstention_detection(text: str | None, expected: bool) -> None:
    assert is_abstention(text) is expected


def test_abstention_accuracy_and_false_answer_rate_lists() -> None:
    # 4 unanswerable queries: 3 correctly abstained, 1 hallucinated
    is_abstained = [True, True, True, False]
    is_unanswerable = [True, True, True, True]

    assert abstention_accuracy(is_abstained, is_unanswerable) == pytest.approx(0.75)
    assert false_answer_rate(is_abstained, is_unanswerable) == pytest.approx(0.25)


def test_abstention_accuracy_and_false_answer_rate_single() -> None:
    # Unanswerable query that abstained
    assert abstention_accuracy(True, True) == 1.0
    assert false_answer_rate(True, True) == 0.0

    # Unanswerable query that hallucinated
    assert abstention_accuracy(False, True) == 0.0
    assert false_answer_rate(False, True) == 1.0

    # Answerable query that answered
    assert abstention_accuracy(False, False) == 1.0
    assert false_answer_rate(False, False) == 0.0


def test_abstention_metrics_mixed_cohort() -> None:
    # 2 unanswerable: 1 abstained, 1 answered (hallucination)
    # 2 answerable: 2 answered (non-abstention)
    is_abstained = [True, False, False, False]
    is_unanswerable = [True, True, False, False]

    assert abstention_accuracy(is_abstained, is_unanswerable) == pytest.approx(0.5)
    assert false_answer_rate(is_abstained, is_unanswerable) == pytest.approx(0.5)


def test_abstention_metrics_empty_cohort() -> None:
    assert abstention_accuracy([], []) == pytest.approx(0.0)
    assert false_answer_rate([], []) == pytest.approx(0.0)

    # All answerable (no unanswerable cases)
    assert abstention_accuracy([False, False], [False, False]) == pytest.approx(1.0)
    assert false_answer_rate([False, False], [False, False]) == pytest.approx(0.0)


def test_abstention_metrics_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        abstention_accuracy([True], [True, False])
    with pytest.raises(ValueError, match="same length"):
        false_answer_rate([True], [True, False])
