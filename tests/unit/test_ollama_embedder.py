"""Tests for Ollama embedding response validation without a live Ollama service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.embed import OllamaEmbedder
from core.errors import EmbedderError


class FakeOllamaClient:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self.embeddings = embeddings or []
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def embed(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(embeddings=self.embeddings)


def test_batch_and_query_embedding_record_model_and_dimension() -> None:
    client = FakeOllamaClient([[1.0, 0.0], [0.0, 1.0]])
    embedder = OllamaEmbedder("local-embed", client=client)

    vectors = embedder.embed_documents(["first", "second"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.dimension == 2
    assert embedder.model_id == "ollama:local-embed"
    assert client.calls[0] == {"model": "local-embed", "input": ["first", "second"]}

    client.embeddings = [[0.5, 0.5]]
    assert embedder.embed_query("query") == [0.5, 0.5]
    assert client.calls[1] == {"model": "local-embed", "input": "query"}


def test_empty_batch_does_not_call_provider() -> None:
    client = FakeOllamaClient()
    embedder = OllamaEmbedder("local-embed", client=client)
    assert embedder.embed_documents([]) == []
    assert client.calls == []


@pytest.mark.parametrize("model", ["", "   "])
def test_model_must_be_configured(model: str) -> None:
    with pytest.raises(EmbedderError, match="must be configured"):
        OllamaEmbedder(model, client=FakeOllamaClient())


def test_rejects_blank_inputs() -> None:
    embedder = OllamaEmbedder("local-embed", client=FakeOllamaClient())
    with pytest.raises(EmbedderError, match="non-empty"):
        embedder.embed_documents(["valid", " "])
    with pytest.raises(EmbedderError, match="non-empty"):
        embedder.embed_query("\t")


def test_rejects_wrong_vector_count_and_non_finite_values() -> None:
    client = FakeOllamaClient([[1.0, 2.0]])
    embedder = OllamaEmbedder("local-embed", client=client)
    with pytest.raises(EmbedderError, match="number of document embeddings"):
        embedder.embed_documents(["one", "two"])

    client.embeddings = [[float("nan")]]
    with pytest.raises(EmbedderError, match="finite"):
        embedder.embed_query("query")

    client.embeddings = [[10**10000]]
    with pytest.raises(EmbedderError, match="finite"):
        embedder.embed_query("query")


def test_rejects_dimension_drift_between_calls() -> None:
    client = FakeOllamaClient([[1.0, 0.0]])
    embedder = OllamaEmbedder("local-embed", client=client)
    assert embedder.embed_query("first") == [1.0, 0.0]
    client.embeddings = [[1.0, 0.0, 0.0]]
    with pytest.raises(EmbedderError, match="dimension changed"):
        embedder.embed_query("second")


def test_provider_failure_is_reported_with_model_identity() -> None:
    client = FakeOllamaClient()
    client.error = ConnectionError("server unavailable")
    embedder = OllamaEmbedder("local-embed", client=client)
    with pytest.raises(EmbedderError, match="local-embed.*server unavailable"):
        embedder.embed_query("query")
