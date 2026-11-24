"""Tests for the OpenAI-compatible embedding provider without network access."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from core.embed import OpenAICompatibleEmbedder
from core.errors import EmbedderError


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embeddings_use_standard_endpoint_auth_and_ordered_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1, 0]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                ]
            },
        )

    embedder = OpenAICompatibleEmbedder(
        "@cf/baai/bge-small-en-v1.5",
        "https://api.example.test/ai/v1/",
        "test-token",
        client=_client(handler),
    )

    assert embedder.embed_documents(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.dimension == 2
    assert embedder.model_id == "openai_compatible:@cf/baai/bge-small-en-v1.5"
    assert seen["url"] == "https://api.example.test/ai/v1/embeddings"
    assert seen["authorization"] == "Bearer test-token"
    assert seen["body"] == {
        "model": "@cf/baai/bge-small-en-v1.5",
        "input": ["first", "second"],
        "encoding_format": "float",
    }


def test_empty_batch_and_invalid_inputs_do_not_send_requests() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    embedder = OpenAICompatibleEmbedder(
        "model", "https://api.example.test/v1", "key", _client(handler)
    )
    assert embedder.embed_documents([]) == []
    with pytest.raises(EmbedderError, match="non-empty"):
        embedder.embed_documents(["valid", " "])
    with pytest.raises(EmbedderError, match="non-empty"):
        embedder.embed_query("\t")
    assert calls == 0


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"data": [{"index": 0, "embedding": [1.0]}]}, "number of embeddings"),
        (
            {"data": [{"index": 1, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
            "out of order",
        ),
    ],
)
def test_rejects_invalid_provider_responses(payload, error) -> None:
    embedder = OpenAICompatibleEmbedder(
        "model",
        "https://api.example.test/v1",
        "key",
        _client(lambda request: httpx.Response(200, json=payload)),
    )
    with pytest.raises(EmbedderError, match=error):
        embedder.embed_documents(["first", "second"])


def test_rejects_non_finite_vectors() -> None:
    embedder = OpenAICompatibleEmbedder(
        "model",
        "https://api.example.test/v1",
        "key",
        _client(lambda request: httpx.Response(200)),
    )
    with pytest.raises(EmbedderError, match="finite"):
        embedder._validate_vector([math.nan])


def test_rejects_dimension_drift_and_does_not_leak_response_body() -> None:
    responses = iter(
        [
            {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]},
        ]
    )
    embedder = OpenAICompatibleEmbedder(
        "model",
        "https://api.example.test/v1",
        "key",
        _client(lambda request: httpx.Response(200, json=next(responses))),
    )
    assert embedder.embed_query("first") == [1.0, 0.0]
    with pytest.raises(EmbedderError, match="dimension changed"):
        embedder.embed_query("second")


def test_http_failure_is_sanitized() -> None:
    embedder = OpenAICompatibleEmbedder(
        "model",
        "https://api.example.test/v1",
        "secret-token",
        _client(lambda request: httpx.Response(401, text="provider response detail")),
    )
    with pytest.raises(EmbedderError, match="request failed") as error:
        embedder.embed_query("query")
    assert "secret-token" not in str(error.value)
    assert "provider response detail" not in str(error.value)
