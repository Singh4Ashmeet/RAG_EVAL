"""Parse model citation labels and resolve them to retrieved chunks."""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.generate.schema import AnswerCitation
from core.retrieval.schema import SearchResult

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def resolve_citations(
    answer: str,
    sources: Sequence[SearchResult],
) -> tuple[list[AnswerCitation], list[str]]:
    """Return unique source references and citation labels that do not resolve.

    This validates citation references, not factual support. A valid label only
    proves that the model referenced a chunk that was present in its context.
    """
    labels = list(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
    citations: list[AnswerCitation] = []
    unresolved: list[str] = []
    for label in labels:
        if not re.fullmatch(r"S[1-9]\d*", label):
            unresolved.append(label)
            continue
        suffix = label[1:]
        if not suffix.isdecimal():
            unresolved.append(label)
            continue
        source_index = int(suffix) - 1
        if not 0 <= source_index < len(sources):
            unresolved.append(label)
            continue
        chunk = sources[source_index].chunk
        citations.append(
            AnswerCitation(
                label=label,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
            )
        )
    return citations, unresolved
