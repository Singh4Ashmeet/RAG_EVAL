"""BM25 lexical retrieval implementation over the same chunk corpus as dense retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.base import Retriever
from core.retrieval.schema import RetrievalStrategy, SearchResult

# Technical token patterns that must remain searchable as single units
_TECHNICAL_PATTERNS = [
    r"RFC\s*\d+",  # RFC 8259, RFC8259
    r"UTF-\d+",  # UTF-8, UTF-16
    r"JSON",  # JSON
    r"ABNF",  # ABNF
    r"MIME",  # MIME
    r"HTTP",  # HTTP, HTTPS
    r"TLS",  # TLS
    r"SHA-\d+",  # SHA-256
    r"ECMA-\d+",  # ECMA-404
    r"ISO-\d+",  # ISO-8601
    r"BOM",  # BOM
    r"URI",  # URI, URL
    r"API",  # API
    r"CPU",  # CPU
    r"GPU",  # GPU
    r"RAM",  # RAM
    r"SSD",  # SSD
    r"NVMe",  # NVMe
    r"SQL",  # SQL
    r"HTML",  # HTML
    r"XML",  # XML
    r"YAML",  # YAML
    r"TOML",  # TOML
    r"Docker",  # Docker
    r"Kubernetes",  # Kubernetes
    r"Prometheus",  # Prometheus
    r"OpenTelemetry",  # OpenTelemetry
    r"W3C",  # W3C
    r"IETF",  # IETF
    r"IANA",  # IANA
    r"BCP\s*\d+",  # BCP 178
    r"STD\s*\d+",  # STD 90
    r"OTLP",  # OTLP
    r"gRPC",  # gRPC
]

# Compile patterns for technical term preservation
_TECHNICAL_RE = re.compile("|".join(f"({p})" for p in _TECHNICAL_PATTERNS), re.IGNORECASE)

# Standard word tokenization (preserve technical terms)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*|[A-Za-z]+|[0-9]+|[^\s\w]")


def _tokenize(text: str) -> list[str]:
    """
    Deterministic tokenization that preserves technical terms.

    Technical terms (RFC 8259, UTF-8, JSON, etc.) are kept as single tokens.
    Other text is split on word boundaries.
    """
    if not text or not text.strip():
        return []

    # First, protect technical terms by replacing them with placeholders
    protected: list[str] = []
    placeholder_map: dict[str, str] = {}

    def protect_match(match: re.Match) -> str:
        term = match.group(0)
        placeholder = f"__TECH_{len(protected)}__"
        protected.append(term)
        placeholder_map[placeholder] = term
        return placeholder

    protected_text = _TECHNICAL_RE.sub(protect_match, text)

    # Tokenize the remaining text
    tokens = _WORD_RE.findall(protected_text)

    # Restore technical terms
    result = []
    for token in tokens:
        if token in placeholder_map:
            result.append(placeholder_map[token].lower())
        else:
            # Normalize: lowercase, keep alphanumeric and dots
            normalized = token.lower()
            if normalized:
                result.append(normalized)

    return result


class BM25Retriever(Retriever):
    """BM25 lexical retriever over a fixed chunk corpus."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer=None,
    ) -> None:
        if not chunks:
            raise RetrievalError("BM25Retriever requires at least one chunk.")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer or _tokenize

        # Build tokenized corpus
        self._tokenized_chunks = [self._tokenize(chunk.text) for chunk in self.chunks]

        # Compute average document length
        total_len = sum(len(tokens) for tokens in self._tokenized_chunks)
        self._avgdl = total_len / len(self._tokenized_chunks)

        # Build BM25 index
        self._bm25 = BM25Okapi(self._tokenized_chunks, k1=k1, b=b)

        # Map chunk index to chunk
        self._chunk_by_index = {i: chunk for i, chunk in enumerate(self.chunks)}

    def _tokenize(self, text: str) -> list[str]:
        return self._tokenizer(text)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve top matching chunks using BM25 scores."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")

        try:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # Get BM25 scores for all documents
            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices by score (descending)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            # Build SearchResults
            results = []
            for rank, idx in enumerate(top_indices):
                chunk = self._chunk_by_index[idx]
                score = float(scores[idx])
                # BM25 scores can be negative; normalize to [0, 1] range for consistency
                normalized_score = 1.0 / (1.0 + math.exp(-score))  # Sigmoid normalization
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=normalized_score,
                        rank=rank,
                        strategy=RetrievalStrategy.SPARSE,
                    )
                )

            return results

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"BM25 retrieval failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return BM25 index statistics."""
        return {
            "num_chunks": len(self.chunks),
            "avg_chunk_length": self._avgdl,
            "k1": self.k1,
            "b": self.b,
            "vocab_size": len(self._bm25.idf) if hasattr(self._bm25, "idf") else 0,
        }
