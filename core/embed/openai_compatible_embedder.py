"""OpenAI-compatible HTTP embedding provider."""

from __future__ import annotations

import math
from typing import Any

import httpx

from core.errors import EmbedderError


class OpenAICompatibleEmbedder:
    """Generate embeddings from an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        client: httpx.Client | None = None,
        *,
        timeout: float | None = 120.0,
    ) -> None:
        cleaned_model = model.strip()
        cleaned_base_url = base_url.rstrip("/")
        if not cleaned_model:
            raise EmbedderError("An OpenAI-compatible embedding model must be configured.")
        if not cleaned_base_url:
            raise EmbedderError("An OpenAI-compatible embedding base URL must be configured.")
        if not api_key or not api_key.strip():
            raise EmbedderError("An OpenAI-compatible embedding API key must be configured.")

        self.model = cleaned_model
        self.model_id = f"openai_compatible:{cleaned_model}"
        self.base_url = cleaned_base_url
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbedderError(
                "Embedding dimension is unknown until the provider returns the first vector."
            )
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise EmbedderError("Document texts must be provided as a list of strings.")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbedderError("Document texts must be non-empty strings.")
        return self._embed(texts)

    def embed_query(self, text: str, *, timeout: float | None = None) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbedderError("Query text must be a non-empty string.")
        return self._embed([text], timeout=timeout)[0]

    def _embed(self, texts: list[str], *, timeout: float | None = None) -> list[list[float]]:
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers,
                json={"model": self.model, "input": texts, "encoding_format": "float"},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise EmbedderError(
                "OpenAI-compatible embedding request failed "
                f"with HTTP {exc.response.status_code} for model '{self.model}'."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbedderError(
                f"OpenAI-compatible embedding request failed for model '{self.model}'."
            ) from exc

        raw_items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list) or len(raw_items) != len(texts):
            raise EmbedderError(
                "OpenAI-compatible provider returned an unexpected number of embeddings "
                f"(expected {len(texts)})."
            )

        vectors: list[list[float]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict) or item.get("index") != index:
                raise EmbedderError("OpenAI-compatible provider returned embeddings out of order.")
            vectors.append(self._validate_vector(item.get("embedding")))
        return vectors

    def _validate_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise EmbedderError(
                "OpenAI-compatible provider returned an empty or invalid embedding vector."
            )

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
                f"Embedding dimension changed from {self._dimension} to {len(values)}."
            )
        return values
