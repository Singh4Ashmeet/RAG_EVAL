"""SPLADE sparse encoder for learned sparse retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation with token IDs and weights."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        # Ensure indices are sorted
        pairs = sorted(zip(self.indices, self.values, strict=False))
        object.__setattr__(self, "indices", tuple(p[0] for p in pairs))
        object.__setattr__(self, "values", tuple(p[1] for p in pairs))

    def dot(self, other: SparseVector) -> float:
        """Compute dot product with another sparse vector."""
        i = j = 0
        result = 0.0
        while i < len(self.indices) and j < len(other.indices):
            if self.indices[i] == other.indices[j]:
                result += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.indices[i] < other.indices[j]:
                i += 1
            else:
                j += 1
        return result

    def __len__(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=False))

    @classmethod
    def from_dict(cls, d: dict[int, float]) -> SparseVector:
        if not d:
            return cls(indices=(), values=())
        items = sorted(d.items())
        return cls(indices=tuple(k for k, _ in items), values=tuple(v for _, v in items))


class SparseEncoder:
    """SPLADE sparse encoder using SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        *,
        max_length: int = 512,
        device: str | None = None,
        cache_dir: str | None = None,
        sparse_threshold: float = 0.0,
        top_k_tokens: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.sparse_threshold = sparse_threshold
        self.top_k_tokens = top_k_tokens
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading SPLADE model: {model_name} on {self.device}")
        start = time.perf_counter()

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_dir,
            trust_remote_code=True,
        )

        # Set max sequence length
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = max_length

        load_time = time.perf_counter() - start
        logger.info(f"SPLADE model loaded in {load_time:.2f}s on {self.device}")

    def encode_query(self, query: str) -> SparseVector:
        """Encode a single query into a sparse vector."""
        vectors = self.encode_queries([query])
        return vectors[0]

    def encode_queries(self, queries: list[str]) -> list[SparseVector]:
        """Encode multiple queries into sparse vectors."""
        if not queries:
            return []

        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
        )

        return [self._postprocess_embedding(emb) for emb in embeddings]

    def encode_document(self, text: str) -> SparseVector:
        """Encode a single document into a sparse vector."""
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode multiple documents into sparse vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
        )

        return [self._postprocess_embedding(emb) for emb in embeddings]

    def _postprocess_embedding(self, embedding: torch.Tensor) -> SparseVector:
        """Convert dense embedding to sparse vector with thresholding."""
        # Move to CPU for processing
        vec = embedding.detach().cpu()

        # Apply ReLU (SPLADE uses ReLU activation)
        vec = torch.relu(vec)

        # Apply threshold
        if self.sparse_threshold > 0:
            vec = torch.where(vec > self.sparse_threshold, vec, torch.zeros_like(vec))

        # Optionally keep only top-k tokens
        if self.top_k_tokens is not None and self.top_k_tokens > 0:
            k = min(self.top_k_tokens, vec.size(0))
            topk_values, topk_indices = torch.topk(vec, k)
            # Create sparse vector from top-k
            indices = topk_indices.tolist()
            values = topk_values.tolist()
        else:
            # Get all non-zero indices
            non_zero = vec.nonzero(as_tuple=True)[0]
            indices = non_zero.tolist()
            values = vec[non_zero].tolist()

        return SparseVector(indices=tuple(indices), values=tuple(values))

    def get_stats(self) -> dict[str, Any]:
        """Return encoder statistics."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "device": self.device,
            "sparse_threshold": self.sparse_threshold,
            "top_k_tokens": self.top_k_tokens,
            "batch_size": self.batch_size,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
        }
