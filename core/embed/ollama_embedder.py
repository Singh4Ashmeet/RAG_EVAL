"""Local embedding provider backed by Ollama's embedding API."""

from __future__ import annotations

import math
from typing import Any

import ollama

from core.errors import EmbedderError


class OllamaEmbedder:
    """Generate document and query embeddings with a configured local Ollama model.

    The model must support embeddings and must already be available to Ollama. This
    class never downloads models. The output dimension is learned from the first
    response and checked on every subsequent call.
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        client: Any | None = None,
        *,
        timeout: float | None = 120.0,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise EmbedderError("An Ollama embedding model must be configured.")
        self.model = cleaned_model
        self.model_id = f"ollama:{cleaned_model}"
        if client is not None:
            self._client = client
        elif timeout is not None:
            self._client = ollama.Client(host=host, timeout=timeout)
        else:
            self._client = ollama.Client(host=host)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the observed embedding dimension after the first successful call."""
        if self._dimension is None:
            raise EmbedderError(
                "Embedding dimension is unknown until Ollama returns the first vector."
            )
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of non-empty document passages."""
        if not isinstance(texts, list):
            raise EmbedderError("Document texts must be provided as a list of strings.")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbedderError("Document texts must be non-empty strings.")
        response = self._embed(texts)
        raw_vectors = getattr(response, "embeddings", None)
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
            raise EmbedderError(
                "Ollama returned an unexpected number of document embeddings "
                f"(expected {len(texts)})."
            )
        return [self._validate_vector(vector) for vector in raw_vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed one non-empty search query."""
        if not isinstance(text, str) or not text.strip():
            raise EmbedderError("Query text must be a non-empty string.")
        response = self._embed(text)
        raw_vectors = getattr(response, "embeddings", None)
        if not isinstance(raw_vectors, list) or len(raw_vectors) != 1:
            raise EmbedderError("Ollama returned an unexpected query embedding response.")
        return self._validate_vector(raw_vectors[0])

    def _embed(self, input_data: str | list[str]) -> Any:
        try:
            return self._client.embed(model=self.model, input=input_data)
        except Exception as exc:
            raise EmbedderError(
                f"Ollama embedding request failed for model '{self.model}': {exc}"
            ) from exc

    def _validate_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise EmbedderError("Ollama returned an empty or invalid embedding vector.")

        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbedderError("Embedding values must be finite numbers.")
            try:
                numeric = float(value)
            except (OverflowError, ValueError) as exc:
                raise EmbedderError("Embedding values must be finite numbers.") from exc
            if not math.isfinite(numeric):
                raise EmbedderError("Embedding values must be finite numbers.")
            values.append(numeric)

        if self._dimension is None:
            self._dimension = len(values)
        elif len(values) != self._dimension:
            raise EmbedderError(
                f"Ollama embedding dimension changed from {self._dimension} to {len(values)}."
            )
        return values
