"""Deterministic lexical answer metrics; neither metric verifies factual support."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def exact_match(prediction: str, reference: str) -> float:
    """Case- and punctuation-insensitive exact token sequence match."""
    return float(_tokens(prediction) == _tokens(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Multiset token overlap F1 after Unicode normalization and case folding."""
    predicted_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0

    predicted_counts = Counter(predicted_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((predicted_counts & reference_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _tokens(text: str) -> list[str]:
    without_citations = re.sub(r"\[S\d+\]", " ", text)
    normalized = unicodedata.normalize("NFKC", without_citations).casefold()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)
