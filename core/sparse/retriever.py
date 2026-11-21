"""Sparse retriever using SPLADE sparse vectors."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.normalize.schema import Chunk
from core.retrieval.base import Retriever
from core.retrieval.schema import RetrievalStrategy, SearchResult
from core.sparse.encoder import SparseEncoder, SparseVector

logger = logging.getLogger(__name__)


@dataclass
class SparseIndexEntry:
    """Entry in the sparse index."""

    chunk: Chunk
    vector: SparseVector
    doc_id: str


class SparseRetriever(Retriever):
    """Sparse retriever using SPLADE sparse vectors."""

    def __init__(
        self,
        encoder: SparseEncoder,
        chunks: Sequence[Chunk],
        *,
        top_k: int = 10,
    ) -> None:
        self.encoder = encoder
        self.top_k = top_k
        self.chunks = list(chunks)

        logger.info(f"Building sparse index for {len(self.chunks)} chunks...")
        start = time.perf_counter()

        self._index: list[SparseIndexEntry] = []
        self._build_index()

        build_time = time.perf_counter() - start
        logger.info(f"Sparse index built in {build_time:.2f}s")

    def _build_index(self) -> None:
        """Build the sparse index from chunks."""
        texts = [chunk.text for chunk in self.chunks]
        vectors = self.encoder.encode_documents(texts)

        for chunk, vector in zip(self.chunks, vectors, strict=False):
            self._index.append(
                SparseIndexEntry(
                    chunk=chunk,
                    vector=vector,
                    doc_id=chunk.document_id,
                )
            )

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve top matching chunks using sparse dot product."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        try:
            query_vector = self.encoder.encode_query(query)

            # Score all chunks
            scored: list[tuple[float, SparseIndexEntry]] = []
            for entry in self._index:
                score = query_vector.dot(entry.vector)
                if score > 0:
                    scored.append((score, entry))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Build results
            results = []
            for rank, (score, entry) in enumerate(scored[:top_k]):
                results.append(
                    SearchResult(
                        chunk=entry.chunk,
                        score=score,
                        rank=rank,
                        strategy=RetrievalStrategy.SPARSE,
                    )
                )

            return results

        except Exception as exc:
            raise ValueError(f"Sparse retrieval failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return retriever statistics."""
        total_nnz = sum(len(entry.vector) for entry in self._index)
        avg_nnz = total_nnz / len(self._index) if self._index else 0

        return {
            "num_chunks": len(self._index),
            "total_nnz": total_nnz,
            "avg_nnz_per_chunk": avg_nnz,
            "max_nnz": max((len(e.vector) for e in self._index), default=0),
            "top_k": self.top_k,
            "encoder_stats": self.encoder.get_stats(),
        }
