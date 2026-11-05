"""Unit tests for provider fallback mechanism."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.errors import LLMError
from core.llm.base import GenerationSettings
from core.llm.gemini_provider import (
    ProviderAuthenticationError as GeminiAuthError,
)
from core.llm.gemini_provider import (
    ProviderRateLimitError as GeminiRateLimitError,
)
from core.llm.gemini_provider import (
    ProviderUnavailableError as GeminiUnavailableError,
)
from core.llm.router import (
    FallbackProvider,
    ProviderFallbackExhaustedError,
    _is_recoverable_error,
    _parse_fallback_providers,
    clear_provider_cache,
    get_chat_provider,
)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str, response: str = "OK", error: Exception | None = None) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._response = response
        self._error = error
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._response


def test_parse_fallback_providers() -> None:
    """Test parsing comma-separated fallback provider string."""
    assert _parse_fallback_providers("gemini,groq,ollama") == ["gemini", "groq", "ollama"]
    assert _parse_fallback_providers(" gemini , groq , ollama ") == ["gemini", "groq", "ollama"]
    assert _parse_fallback_providers("") == []
    assert _parse_fallback_providers("  ") == []
    assert _parse_fallback_providers("ollama") == ["ollama"]


def test_is_recoverable_error_recoverable() -> None:
    """Test that recoverable errors are correctly identified."""
    import httpx

    # ProviderUnavailableError
    assert _is_recoverable_error(GeminiUnavailableError("Service unavailable")) is True

    # ProviderRateLimitError
    assert _is_recoverable_error(GeminiRateLimitError("Rate limited", retry_after=10)) is True

    # httpx timeout
    assert _is_recoverable_error(httpx.TimeoutException("Timeout")) is True

    # httpx connection error
    assert _is_recoverable_error(httpx.ConnectError("Connection failed")) is True

    # 5xx HTTP error
    response = SimpleNamespace(status_code=503, text="Service unavailable")
    assert (
        _is_recoverable_error(httpx.HTTPStatusError("503", request=None, response=response)) is True
    )

    # 429 rate limit
    response = SimpleNamespace(status_code=429, text="Rate limited")
    assert (
        _is_recoverable_error(httpx.HTTPStatusError("429", request=None, response=response)) is True
    )


def test_is_recoverable_error_non_recoverable() -> None:
    """Test that non-recoverable errors are correctly identified."""
    import httpx

    # Authentication error
    assert _is_recoverable_error(GeminiAuthError("Invalid API key")) is False

    # 401 HTTP error
    response = SimpleNamespace(status_code=401, text="Unauthorized")
    assert (
        _is_recoverable_error(httpx.HTTPStatusError("401", request=None, response=response))
        is False
    )

    # 400 Bad Request (malformed)
    response = SimpleNamespace(status_code=400, text="Bad request")
    assert (
        _is_recoverable_error(httpx.HTTPStatusError("400", request=None, response=response))
        is False
    )

    # 404 Not Found (invalid model)
    response = SimpleNamespace(status_code=404, text="Model not found")
    assert (
        _is_recoverable_error(httpx.HTTPStatusError("404", request=None, response=response))
        is False
    )

    # Keywords in error message
    assert _is_recoverable_error(LLMError("Authentication failed")) is False
    assert _is_recoverable_error(LLMError("Invalid API key provided")) is False
    assert _is_recoverable_error(LLMError("Model not found")) is False
    assert _is_recoverable_error(LLMError("Invalid request")) is False


def test_fallback_provider_primary_succeeds() -> None:
    """Test FallbackProvider when primary succeeds."""
    primary = MockProvider("primary", "Primary response")
    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[MockProvider("fallback", "Fallback response")],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    result = fallback.generate("sys", "user", settings=GenerationSettings())
    assert result == "Primary response"
    assert primary.call_count == 1


def test_fallback_provider_primary_fails_fallback_succeeds() -> None:
    """Test FallbackProvider when primary fails and fallback succeeds."""
    primary = MockProvider("primary", error=GeminiUnavailableError("Primary down"))
    fallback_provider = MockProvider("fallback", "Fallback response")
    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback_provider],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    result = fallback.generate("sys", "user", settings=GenerationSettings())
    assert result == "Fallback response"
    assert primary.call_count == 1
    assert fallback_provider.call_count == 1


def test_fallback_provider_all_fail() -> None:
    """Test FallbackProvider when all providers fail."""
    primary = MockProvider("primary", error=GeminiUnavailableError("Primary down"))
    fallback1 = MockProvider("fallback1", error=GeminiUnavailableError("Fallback1 down"))
    fallback2 = MockProvider("fallback2", error=GeminiUnavailableError("Fallback2 down"))

    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback1, fallback2],
        primary_name="primary",
        fallback_names=["fallback1", "fallback2"],
    )

    with pytest.raises(ProviderFallbackExhaustedError) as exc_info:
        fallback.generate("sys", "user", settings=GenerationSettings())

    assert "All providers failed" in str(exc_info.value)
    assert exc_info.value.failed_providers == [
        "primary: ProviderUnavailableError",
        "fallback1: ProviderUnavailableError",
        "fallback2: ProviderUnavailableError",
    ]
    assert primary.call_count == 1
    assert fallback1.call_count == 1
    assert fallback2.call_count == 1


def test_fallback_provider_non_recoverable_error_no_fallback() -> None:
    """Test that non-recoverable errors don't trigger fallback."""
    primary = MockProvider("primary", error=GeminiAuthError("Invalid key"))
    fallback_provider = MockProvider("fallback", "Fallback response")

    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback_provider],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    with pytest.raises(GeminiAuthError):
        fallback.generate("sys", "user", settings=GenerationSettings())

    assert primary.call_count == 1
    assert fallback_provider.call_count == 0


def test_fallback_provider_preserves_original_exception() -> None:
    """Test that the original exception is preserved in fallback exhaustion."""
    primary = MockProvider("primary", error=GeminiUnavailableError("Primary down"))
    fallback1 = MockProvider("fallback1", error=GeminiUnavailableError("Fallback1 down"))

    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback1],
        primary_name="primary",
        fallback_names=["fallback1"],
    )

    with pytest.raises(ProviderFallbackExhaustedError) as exc_info:
        fallback.generate("sys", "user", settings=GenerationSettings())

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, GeminiUnavailableError)


def test_fallback_provider_model_id() -> None:
    """Test that FallbackProvider exposes primary model_id."""
    primary = MockProvider("primary", "response")
    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[MockProvider("fallback", "response")],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    assert fallback.model_id == "mock:primary"


def test_fallback_provider_duplicate_primary_not_in_fallback() -> None:
    """Test that primary provider is not duplicated in fallback chain."""
    # This is tested at the router level, but we can verify the FallbackProvider
    # doesn't call the same provider twice if passed duplicates
    primary = MockProvider("primary", "response")
    fallback_provider = MockProvider("primary", "fallback response")  # Same name

    fallback = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback_provider],
        primary_name="primary",
        fallback_names=["primary"],
    )

    result = fallback.generate("sys", "user", settings=GenerationSettings())
    assert result == "response"
    assert primary.call_count == 1
    assert fallback_provider.call_count == 0


def test_clear_provider_cache() -> None:
    """Test that clear_provider_cache works."""
    # This tests the actual router function
    clear_provider_cache()
    # Should not raise
    clear_provider_cache()


def test_get_chat_provider_caching() -> None:
    """Test that get_chat_provider caches instances."""
    import os

    os.environ["CHAT_PROVIDER"] = "ollama"

    # Reload settings
    from core.config import clear_settings_cache

    clear_settings_cache()

    provider1 = get_chat_provider("ollama")
    provider2 = get_chat_provider("ollama")

    assert provider1 is provider2  # Same cached instance


def test_provider_fallback_exhausted_error_stores_failed() -> None:
    """Test that ProviderFallbackExhaustedError stores failed providers list."""
    failed = ["primary: ProviderUnavailableError", "fallback1: ProviderUnavailableError"]
    exc = ProviderFallbackExhaustedError("All failed", failed)
    assert exc.failed_providers == failed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
