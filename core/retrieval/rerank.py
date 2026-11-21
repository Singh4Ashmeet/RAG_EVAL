"""Cross-encoder reranking for hybrid retrieval results."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.errors import RetrievalError
from core.retrieval.base import Reranker
from core.retrieval.schema import RetrievalStrategy, SearchResult

logger = logging.getLogger(__name__)


try:
    from sentence_transformers import CrossEncoder

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    CrossEncoder = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


# Lightweight cross-encoder models suitable for CPU
CROSS_ENCODER_MODELS = {
    "mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~22M params, fast on CPU
    "tiny": "cross-encoder/ms-marco-TinyBERT-L-2-v2",  # ~17M params, very fast on CPU
    "base": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # ~33M params, slower on CPU
}


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "mini",
        *,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RetrievalError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )

        if model_name in CROSS_ENCODER_MODELS:
            model_id = CROSS_ENCODER_MODELS[model_name]
        else:
            model_id = model_name

        self.model_name = model_name
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size

        # Determine device
        if device is not None:
            self.device = device
        else:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading cross-encoder model: {model_id} on {self.device}")
        start = time.perf_counter()

        self._model = CrossEncoder(
            model_id,
            max_length=max_length,
            device=self.device,
            cache_folder=str(cache_dir) if cache_dir else None,
        )

        load_time = time.perf_counter() - start
        logger.info(f"Cross-encoder loaded in {load_time:.2f}s on {self.device}")

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank candidates using cross-encoder scores."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")
        if not candidates:
            return []

        try:
            # Prepare query-chunk pairs
            pairs = [(query, candidate.chunk.text) for candidate in candidates]

            # Get cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} candidates with cross-encoder")
            start = time.perf_counter()

            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            rerank_time = time.perf_counter() - start
            logger.debug(f"Cross-encoder reranked {len(pairs)} candidates in {rerank_time:.3f}s")

            # Combine with candidates and sort by score descending
            scored_candidates = list(zip(candidates, scores, strict=True))
            scored_candidates.sort(key=lambda x: float(x[1]), reverse=True)

            # Build reranked results
            results = []
            for rank, (candidate, score) in enumerate(scored_candidates[:top_k]):
                results.append(
                    SearchResult(
                        chunk=candidate.chunk,
                        score=float(score),
                        rank=rank,
                        strategy=RetrievalStrategy.HYBRID,
                    )
                )

            return results

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Cross-encoder reranking failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return reranker statistics."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
        }


def create_reranker(
    model_name: str = "mini",
    *,
    max_length: int = 512,
    batch_size: int = 32,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> CrossEncoderReranker:
    """Factory function to create a cross-encoder reranker."""
    return CrossEncoderReranker(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )


class RerankingRetriever:
    """Wrapper that adds cross-encoder reranking to any retriever."""

    def __init__(
        self,
        base_retriever,
        reranker: CrossEncoderReranker,
        *,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.final_k = final_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve candidates then rerank."""
        if top_k is None:
            top_k = self.final_k
        if top_k < 1:
            top_k = 1

        # Retrieve larger candidate pool
        candidates = self.base_retriever.retrieve(query, top_k=self.candidate_k)

        # Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return reranked

    def get_stats(self) -> dict[str, Any]:
        """Return combined statistics."""
        base_stats = (
            self.base_retriever.get_stats() if hasattr(self.base_retriever, "get_stats") else {}
        )
        reranker_stats = self.reranker.get_stats() if hasattr(self.reranker, "get_stats") else {}
        return {
            "base_retriever": base_stats,
            "reranker": reranker_stats,
            "candidate_k": self.candidate_k,
            "final_k": self.final_k,
        }
