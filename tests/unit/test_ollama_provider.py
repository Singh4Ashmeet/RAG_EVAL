"""Unit tests for the Ollama chat adapter using a fake SDK client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.errors import LLMError
from core.llm import GenerationSettings, LLMProvider, OllamaLLMProvider


class FakeOllamaChatClient:
    def __init__(self, content: str = "A response.") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def test_ollama_provider_sends_non_streaming_chat_and_generation_options() -> None:
    client = FakeOllamaChatClient("  Answer [S1].  ")
    provider = OllamaLLMProvider("llama3.2:3b", client=client)
    settings = GenerationSettings(temperature=0.3, max_tokens=96)

    response = provider.generate("System policy", "Question and context", settings=settings)

    assert response == "Answer [S1]."
    assert provider.model_id == "ollama:llama3.2:3b"
    assert isinstance(provider, LLMProvider)
    assert client.calls == [
        {
            "model": "llama3.2:3b",
            "messages": [
                {"role": "system", "content": "System policy"},
                {"role": "user", "content": "Question and context"},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 96},
        }
    ]


@pytest.mark.parametrize("model", ["", " "])
def test_model_must_be_configured(model: str) -> None:
    with pytest.raises(LLMError, match="must be configured"):
        OllamaLLMProvider(model, client=FakeOllamaChatClient())


@pytest.mark.parametrize("system,user", [("", "question"), ("system", "  ")])
def test_prompts_must_be_non_empty(system: str, user: str) -> None:
    provider = OllamaLLMProvider("llama3.2:3b", client=FakeOllamaChatClient())
    with pytest.raises(LLMError, match="prompts must be non-empty"):
        provider.generate(system, user, settings=GenerationSettings())


def test_empty_provider_output_is_an_error() -> None:
    provider = OllamaLLMProvider("llama3.2:3b", client=FakeOllamaChatClient(" \n"))
    with pytest.raises(LLMError, match="empty response"):
        provider.generate("system", "user", settings=GenerationSettings())


def test_provider_failure_is_reported_with_model_identity() -> None:
    client = FakeOllamaChatClient()
    client.error = ConnectionError("server unavailable")
    provider = OllamaLLMProvider("llama3.2:3b", client=client)
    with pytest.raises(LLMError, match="llama3.2:3b.*server unavailable"):
        provider.generate("system", "user", settings=GenerationSettings())


def test_generation_settings_validate_supported_ranges() -> None:
    with pytest.raises(ValueError):
        GenerationSettings(temperature=2.1)
    with pytest.raises(ValueError):
        GenerationSettings(max_tokens=0)
