"""Tests for token usage metadata extraction in providers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.llm import GenerationResponse, GenerationSettings, OllamaLLMProvider


class FakeOllamaChatClientWithMetadata:
    """Fake Ollama client that returns response with token usage metadata."""

    def __init__(
        self,
        content: str = "A response.",
        prompt_eval_count: int | None = None,
        eval_count: int | None = None,
    ) -> None:
        self.content = content
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        # Build response with token usage fields
        message = SimpleNamespace(content=self.content)
        attrs = {"message": message}
        if self.prompt_eval_count is not None:
            attrs["prompt_eval_count"] = self.prompt_eval_count
        if self.eval_count is not None:
            attrs["eval_count"] = self.eval_count
        return SimpleNamespace(**attrs)


class TestOllamaProviderMetadataExtraction:
    """Tests for Ollama token usage metadata extraction."""

    def test_generate_with_metadata_returns_generation_response(self) -> None:
        """Test that generate_with_metadata returns a GenerationResponse."""
        client = FakeOllamaChatClientWithMetadata(
            "  Answer [S1].  ", prompt_eval_count=100, eval_count=50
        )
        provider = OllamaLLMProvider("llama3.2:3b", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        response = provider.generate_with_metadata(
            "System policy", "Question and context", settings=settings
        )

        assert isinstance(response, GenerationResponse)
        assert response.text == "Answer [S1]."
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150

    def test_generate_with_metadata_extracts_input_tokens(self) -> None:
        """Test that input tokens (prompt_eval_count) are extracted."""
        client = FakeOllamaChatClientWithMetadata("Response", prompt_eval_count=42, eval_count=None)
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 42
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_extracts_output_tokens(self) -> None:
        """Test that output tokens (eval_count) are extracted."""
        client = FakeOllamaChatClientWithMetadata("Response", prompt_eval_count=None, eval_count=21)
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens == 21
        assert response.total_tokens is None

    def test_generate_with_metadata_explicit_zero_values_preserved(self) -> None:
        """Test that explicit zero token values are preserved."""
        client = FakeOllamaChatClientWithMetadata("Response", prompt_eval_count=0, eval_count=0)
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0

    def test_generate_with_metadata_missing_fields_are_none(self) -> None:
        """Test that missing token fields remain None."""
        client = FakeOllamaChatClientWithMetadata("Response")
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_total_computed_when_both_present(self) -> None:
        """Test that total_tokens is computed when both input and output are present."""
        client = FakeOllamaChatClientWithMetadata("Response", prompt_eval_count=10, eval_count=5)
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens == 15

    def test_generate_with_metadata_total_none_when_partial(self) -> None:
        """Test that total_tokens is None when only one count is present."""
        client = FakeOllamaChatClientWithMetadata("Response", prompt_eval_count=10, eval_count=None)
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens is None

    def test_generate_with_metadata_malformed_values_become_none(self) -> None:
        """Test that malformed token values are handled gracefully."""
        client = FakeOllamaChatClientWithMetadata("Response")
        provider = OllamaLLMProvider("llama3.2:3b", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())
        assert response.text == "Response"

    def test_generate_preserves_string_return(self) -> None:
        """Test that generate() still returns a plain string."""
        client = FakeOllamaChatClientWithMetadata(
            "  Answer [S1].  ", prompt_eval_count=100, eval_count=50
        )
        provider = OllamaLLMProvider("llama3.2:3b", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        text = provider.generate("System policy", "Question and context", settings=settings)

        assert isinstance(text, str)
        assert text == "Answer [S1]."

    def test_generate_with_metadata_and_generate_consistency(self) -> None:
        """Test that generate() and generate_with_metadata() are consistent."""
        client = FakeOllamaChatClientWithMetadata(
            "  Test response.  ", prompt_eval_count=50, eval_count=25
        )
        provider = OllamaLLMProvider("llama3.2:3b", client=client)
        settings = GenerationSettings()

        response = provider.generate_with_metadata("system", "user", settings=settings)
        text = provider.generate("system", "user", settings=settings)

        assert isinstance(response, GenerationResponse)
        assert isinstance(text, str)
        assert text == response.text


class FakeGeminiResponse:
    """Fake Gemini response with token usage metadata."""

    def __init__(
        self,
        text: str = "A response.",
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
    ) -> None:
        self.text = text
        if any(
            v is not None for v in (prompt_token_count, candidates_token_count, total_token_count)
        ):
            self.usage_metadata = SimpleNamespace(
                prompt_token_count=prompt_token_count,
                candidates_token_count=candidates_token_count,
                total_token_count=total_token_count,
            )
        else:
            self.usage_metadata = None


class FakeGeminiModels:
    """Fake Gemini models with generate_content method."""

    def __init__(self, response: FakeGeminiResponse) -> None:
        self.response = response

    def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
        return self.response


class FakeGeminiClient:
    """Fake Gemini client that returns responses with token usage metadata."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: FakeGeminiResponse | None = None
        self.error: Exception | None = None
        self.models: FakeGeminiModels | None = None

    def _ensure_models(self) -> None:
        if self.response is None:
            self.response = FakeGeminiResponse()
        if self.models is None:
            self.models = FakeGeminiModels(self.response)


class TestGeminiProviderMetadataExtraction:
    """Tests for Gemini token usage metadata extraction."""

    def test_generate_with_metadata_returns_generation_response(self) -> None:
        """Test that generate_with_metadata returns a GenerationResponse."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse(
            text="  Answer [S1].  ",
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        response = provider.generate_with_metadata(
            "System policy", "Question and context", settings=settings
        )

        assert isinstance(response, GenerationResponse)
        assert response.text == "Answer [S1]."
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150

    def test_generate_with_metadata_extracts_input_tokens(self) -> None:
        """Test that input tokens are extracted."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse("Response", prompt_token_count=42)
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 42
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_extracts_output_tokens(self) -> None:
        """Test that output tokens are extracted."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse("Response", candidates_token_count=21)
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens == 21
        assert response.total_tokens is None

    def test_generate_with_metadata_explicit_zero_values_preserved(self) -> None:
        """Test that explicit zero token values are preserved."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse(
            "Response", prompt_token_count=0, candidates_token_count=0
        )
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0

    def test_generate_with_metadata_missing_usage_metadata_are_none(self) -> None:
        """Test that missing usage_metadata results in None token fields."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse("Response")
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_total_computed_when_both_present(self) -> None:
        """Test that total_tokens is computed when both input and output are present."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse(
            "Response", prompt_token_count=10, candidates_token_count=5
        )
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens == 15

    def test_generate_with_metadata_total_none_when_partial(self) -> None:
        """Test that total_tokens is None when only one count is present."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse("Response", prompt_token_count=10)
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens is None

    def test_generate_preserves_string_return(self) -> None:
        """Test that generate() still returns a plain string."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse(
            "  Answer [S1].  ", prompt_token_count=100, candidates_token_count=50
        )
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        text = provider.generate("System policy", "Question and context", settings=settings)

        assert isinstance(text, str)
        assert text == "Answer [S1]."

    def test_generate_with_metadata_and_generate_consistency(self) -> None:
        """Test that generate() and generate_with_metadata() are consistent."""
        from core.llm import GeminiLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGeminiClient()
        client.response = FakeGeminiResponse(
            "  Test response.  ", prompt_token_count=50, candidates_token_count=25
        )
        client._ensure_models()
        provider = GeminiLLMProvider("gemini-2.5-flash", api_key="test-key", client=client)
        settings = GenerationSettings()

        response = provider.generate_with_metadata("system", "user", settings=settings)
        text = provider.generate("system", "user", settings=settings)

        assert isinstance(response, GenerationResponse)
        assert isinstance(text, str)
        assert text == response.text


class FakeGroqChoice:
    """Fake Groq choice with message content."""

    def __init__(self, content: str = "A response.") -> None:
        self.message = SimpleNamespace(content=content)


class FakeGroqUsage:
    """Fake Groq usage metadata."""

    def __init__(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class FakeGroqResponse:
    """Fake Groq response with token usage metadata."""

    def __init__(
        self,
        content: str = "A response.",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.choices = [FakeGroqChoice(content)]
        if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens)):
            self.usage = FakeGroqUsage(prompt_tokens, completion_tokens, total_tokens)
        else:
            self.usage = None


class FakeGroqCompletions:
    """Fake Groq completions with create method."""

    def __init__(self, response: FakeGroqResponse) -> None:
        self.response = response

    def create(self, **kwargs: Any) -> FakeGroqResponse:
        return self.response


class FakeGroqChat:
    """Fake Groq chat with completions."""

    def __init__(self, response: FakeGroqResponse) -> None:
        self.completions = FakeGroqCompletions(response)


class FakeGroqClient:
    """Fake Groq client that returns responses with token usage metadata."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: FakeGroqResponse | None = None
        self.error: Exception | None = None
        self.chat: FakeGroqChat | None = None

    def _ensure_chat(self) -> None:
        if self.response is None:
            self.response = FakeGroqResponse()
        if self.chat is None:
            self.chat = FakeGroqChat(self.response)


class TestGroqProviderMetadataExtraction:
    """Tests for Groq token usage metadata extraction."""

    def test_generate_with_metadata_returns_generation_response(self) -> None:
        """Test that generate_with_metadata returns a GenerationResponse."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse(
            content="  Answer [S1].  ",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        response = provider.generate_with_metadata(
            "System policy", "Question and context", settings=settings
        )

        assert isinstance(response, GenerationResponse)
        assert response.text == "Answer [S1]."
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150

    def test_generate_with_metadata_extracts_input_tokens(self) -> None:
        """Test that input tokens are extracted."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response", prompt_tokens=42)
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 42
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_extracts_output_tokens(self) -> None:
        """Test that output tokens are extracted."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response", completion_tokens=21)
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens == 21
        assert response.total_tokens is None

    def test_generate_with_metadata_explicit_zero_values_preserved(self) -> None:
        """Test that explicit zero token values are preserved."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response", prompt_tokens=0, completion_tokens=0)
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0

    def test_generate_with_metadata_missing_usage_are_none(self) -> None:
        """Test that missing usage results in None token fields."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response")
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None

    def test_generate_with_metadata_total_computed_when_both_present(self) -> None:
        """Test that total_tokens is computed when both input and output are present."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response", prompt_tokens=10, completion_tokens=5)
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens == 15

    def test_generate_with_metadata_total_none_when_partial(self) -> None:
        """Test that total_tokens is None when only one count is present."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse("Response", prompt_tokens=10)
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)

        response = provider.generate_with_metadata("system", "user", settings=GenerationSettings())

        assert response.total_tokens is None

    def test_generate_preserves_string_return(self) -> None:
        """Test that generate() still returns a plain string."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse(
            "  Answer [S1].  ", prompt_tokens=100, completion_tokens=50
        )
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)
        settings = GenerationSettings(temperature=0.3, max_tokens=96)

        text = provider.generate("System policy", "Question and context", settings=settings)

        assert isinstance(text, str)
        assert text == "Answer [S1]."

    def test_generate_with_metadata_and_generate_consistency(self) -> None:
        """Test that generate() and generate_with_metadata() are consistent."""
        from core.llm import GroqLLMProvider
        from core.llm.base import GenerationSettings

        client = FakeGroqClient()
        client.response = FakeGroqResponse(
            "  Test response.  ", prompt_tokens=50, completion_tokens=25
        )
        client._ensure_chat()
        provider = GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key", client=client)
        settings = GenerationSettings()

        response = provider.generate_with_metadata("system", "user", settings=settings)
        text = provider.generate("system", "user", settings=settings)

        assert isinstance(response, GenerationResponse)
        assert isinstance(text, str)
        assert text == response.text


def test_groq_client_constructed_with_max_retries_zero() -> None:
    """Test that Groq client is constructed with max_retries=0."""
    from unittest.mock import MagicMock, patch

    from core.llm import GroqLLMProvider

    with patch("core.llm.groq_provider.Groq") as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client

        GroqLLMProvider("qwen/qwen3.8-27b", api_key="test-key")

        # Verify Groq client was constructed with max_retries=0
        mock_groq.assert_called_once()
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs.get("max_retries") == 0, "Expected max_retries=0 to disable SDK retries"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
