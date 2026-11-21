"""Deterministic abstention and hallucination metrics for unanswerable queries."""

from __future__ import annotations

import re
import unicodedata

# Common refusal / abstention patterns emitted by grounded RAG systems
_ABSTENTION_PATTERNS = [
    r"cannot (answer|be answered|determine|be determined|find|provide)",
    r"(could not|couldn't) find",
    r"unable to (answer|determine|find|provide)",
    r"(do not|don't) have (enough )?information",
    r"not have enough information",
    r"not enough information",
    r"insufficient (information|evidence|context)",
    r"(does|do) not (contain|mention|provide|have|include)",
    r"not (mentioned|provided|available|found)",
    r"no information (is )?(available|provided|found)",
    r"no mention of",
    r"not supported by the (provided )?(source|context|data|text|documents?)",
    r"i (do not|don't) know",
    r"unanswerable",
    r"unknown",
]

_COMPILED_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


def is_abstention(text: str | None) -> bool:
    """Determine whether an answer string represents an explicit refusal or abstention.

    Returns True if the text is empty, None, or contains standard grounded refusal phrases.
    """
    if text is None:
        return True
    cleaned = unicodedata.normalize("NFKC", text).strip()
    if not cleaned:
        return True
    return bool(_COMPILED_ABSTENTION_RE.search(cleaned))


def abstention_accuracy(
    is_abstained: bool | list[bool],
    is_unanswerable: bool | list[bool],
) -> float:
    """Return accuracy of abstention decisions relative to ground-truth answerability.

    Can accept individual booleans or lists of booleans.
    For unanswerable queries: 1.0 if abstained, 0.0 otherwise.
    For answerable queries: 1.0 if not abstained, 0.0 otherwise.
    """
    if isinstance(is_abstained, list) and isinstance(is_unanswerable, list):
        if len(is_abstained) != len(is_unanswerable):
            raise ValueError("is_abstained and is_unanswerable must have the same length")
        if not is_abstained:
            return 0.0
        # Calculate accuracy on unanswerable cohort if unanswerable queries exist
        unans_pairs = [(a, u) for a, u in zip(is_abstained, is_unanswerable, strict=True) if u]
        if not unans_pairs:
            # If no unanswerable queries, check overall match
            correct = sum(1 for a, u in zip(is_abstained, is_unanswerable, strict=True) if a == u)
            return float(correct / len(is_abstained))
        correct = sum(1 for a, _ in unans_pairs if a)
        return float(correct / len(unans_pairs))

    # Single boolean branch
    abstained_bool = bool(is_abstained)
    unans_bool = bool(is_unanswerable)
    if unans_bool:
        return 1.0 if abstained_bool else 0.0
    return 1.0 if not abstained_bool else 0.0


def false_answer_rate(
    is_abstained: bool | list[bool],
    is_unanswerable: bool | list[bool],
) -> float:
    """Return the proportion of unanswerable queries where an answer was attempted.

    Can accept individual booleans or lists of booleans.
    Defined as 1.0 - abstention_accuracy on the unanswerable cohort.
    """
    if isinstance(is_abstained, list) and isinstance(is_unanswerable, list):
        if len(is_abstained) != len(is_unanswerable):
            raise ValueError("is_abstained and is_unanswerable must have the same length")
        if not is_abstained:
            return 0.0
        unans_pairs = [(a, u) for a, u in zip(is_abstained, is_unanswerable, strict=True) if u]
        if not unans_pairs:
            return 0.0
        false_answers = sum(1 for a, _ in unans_pairs if not a)
        return float(false_answers / len(unans_pairs))

    # Single boolean branch
    abstained_bool = bool(is_abstained)
    unans_bool = bool(is_unanswerable)
    if unans_bool:
        return 0.0 if abstained_bool else 1.0
    return 0.0
