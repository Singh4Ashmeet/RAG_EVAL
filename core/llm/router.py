"""Provider registry for LLM chat and embedding providers.

Supports fallback mechanisms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.errors import LLMError
from core.llm.base import GenerationSettings, LLMProvider

if TYPE_CHECKING:
    from core.config import Settings

logger = logging.getLogger(__name__)


class ProviderNotConfiguredError(Exception):
    """Raised when a provider is selected but not properly configured."""

    pass


class ProviderNotSupportedError(Exception):
    """Raised when an unsupported provider is requested."""

    pass


class ProviderFallbackExhaustedError(LLMError):
    """Raised when all fallback providers have been exhausted."""

    def __init__(self, message: str, failed_providers: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed_providers = failed_providers or []


# Cache for provider instances
_chat_provider_cache: dict[str, LLMProvider] = {}
_embedding_provider_cache: dict[str, LLMProvider] = {}


class FallbackProvider(LLMProvider):
    """Wraps primary and fallback LLMProviders to handle automatic failover."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_providers: list[LLMProvider],
        primary_name: str,
        fallback_names: list[str],
    ) -> None:
        self._primary_provider = primary_provider
        self._fallback_providers = fallback_providers
        self._primary_name = primary_name
        self._fallback_names = fallback_names

    @property
    def model_id(self) -> str:
        return getattr(self._primary_provider, "model_id", f"unknown:{self._primary_name}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        failed_providers: list[str] = []
        last_error: Exception | None = None

        # Build list of (name, provider) preserving order and deduplicating
        providers_to_try: list[tuple[str, LLMProvider]] = [
            (self._primary_name, self._primary_provider)
        ]
        seen_names = {self._primary_name}
        for name, provider in zip(self._fallback_names, self._fallback_providers, strict=False):
            if name not in seen_names:
                seen_names.add(name)
                providers_to_try.append((name, provider))

        for idx, (name, provider) in enumerate(providers_to_try):
            try:
                if idx > 0:
                    logger.warning("Attempting fallback to provider %s (fallback %d)", name, idx)
                return provider.generate(system_prompt, user_prompt, settings=settings)
            except Exception as exc:
                last_error = exc
                failed_providers.append(f"{name}: {type(exc).__name__}")

                # Non-recoverable error stops fallback immediately
                if not _is_recoverable_error(exc):
                    logger.error(
                        "Provider %s failed with non-recoverable error %s: %s; aborting fallback",
                        name,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                logger.warning(
                    "Provider %s failed with recoverable error %s: %s; considering fallback",
                    name,
                    type(exc).__name__,
                    exc,
                )

        # All providers failed
        logger.error("All %d providers failed: %s", len(providers_to_try), failed_providers)
        msg = (
            f"All providers failed ({len(providers_to_try)} attempted). "
            f"Attempted: {failed_providers}"
        )
        raise ProviderFallbackExhaustedError(
            msg,
            failed_providers=failed_providers,
        ) from last_error


def _is_recoverable_error(exc: Exception) -> bool:
    """Determine if an exception represents a recoverable/transient error."""
    type_name = type(exc).__name__
    if "Auth" in type_name or "Permission" in type_name:
        return False

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            if status in (400, 401, 403, 404):
                return False
            if status == 429 or status >= 500:
                return True

    exc_str = str(exc).lower()
    non_recoverable_keywords = [
        "authentication",
        "invalid api key",
        "api key",
        "unauthorized",
        "forbidden",
        "bad request",
        "model not found",
        "invalid request",
        "invalid argument",
    ]
    for kw in non_recoverable_keywords:
        if kw in exc_str:
            return False

    if (
        "Unavailable" in type_name
        or "RateLimit" in type_name
        or "Timeout" in type_name
        or "Connect" in type_name
    ):
        return True

    import httpx

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError, ConnectionError)):
        return True

    recoverable_keywords = [
        "rate limit",
        "too many requests",
        "service unavailable",
        "timed out",
        "timeout",
        "connection",
        "overloaded",
    ]
    for kw in recoverable_keywords:
        if kw in exc_str:
            return True

    return False


def _parse_fallback_providers(value: str) -> list[str]:
    """Parse comma-separated list of fallback provider names."""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _create_raw_provider(provider_name: str, settings: Settings) -> LLMProvider:
    """Instantiate a base chat provider."""
    if provider_name == "ollama":
        from core.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(
            model=settings.chat_model,
            host=settings.chat_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "gemini":
        from core.llm.gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(
            model=settings.gemini_chat_model,
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "groq":
        from core.llm.groq_provider import GroqLLMProvider

        return GroqLLMProvider(
            model=settings.groq_chat_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.rq_job_timeout,
        )
    else:
        raise ProviderNotSupportedError(
            f"Unsupported chat provider: {provider_name}. Supported providers: ollama, gemini, groq"
        )


def get_chat_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Get a chat provider instance by name, wrapped with fallback if enabled.

    Args:
        provider_name: Provider name ("ollama", "gemini", "groq").
                       If None, uses CHAT_PROVIDER from settings.

    Returns:
        LLMProvider instance
    """
    from core.config import get_settings

    settings = get_settings()
    provider_name = provider_name or settings.chat_provider

    cache_key = f"chat:{provider_name}:{settings.fallback_enabled}:{settings.fallback_providers}"
    if cache_key in _chat_provider_cache:
        return _chat_provider_cache[cache_key]

    raw_primary = _create_raw_provider(provider_name, settings)

    if settings.fallback_enabled:
        fallback_names = [
            f for f in _parse_fallback_providers(settings.fallback_providers) if f != provider_name
        ]
        wrapped_fallbacks = [_create_raw_provider(f_name, settings) for f_name in fallback_names]
        provider = FallbackProvider(
            primary_provider=raw_primary,
            fallback_providers=wrapped_fallbacks,
            primary_name=provider_name,
            fallback_names=fallback_names,
        )
    else:
        provider = raw_primary

    _chat_provider_cache[cache_key] = provider
    _chat_provider_cache[provider_name] = provider
    return provider


def get_embedding_provider(provider_name: str | None = None):
    """
    Get an embedding provider instance by name.

    Args:
        provider_name: Provider name ("ollama"). If None, uses EMBEDDING_PROVIDER from settings.

    Returns:
        LLMProvider instance (wraps embedding functionality)
    """
    from core.config import get_settings

    settings = get_settings()

    provider_name = provider_name or settings.embedding_provider

    cache_key = f"embedding:{provider_name}"
    if cache_key in _embedding_provider_cache:
        return _embedding_provider_cache[cache_key]

    if provider_name == "ollama":
        from core.embed.ollama_embedder import OllamaEmbedder

        provider = OllamaEmbedder(
            model=settings.embedding_model,
            host=settings.embedding_base_url,
            timeout=settings.rq_job_timeout,
        )
    else:
        raise ProviderNotSupportedError(
            f"Unsupported embedding provider: {provider_name}. Supported providers: ollama"
        )

    _embedding_provider_cache[provider_name] = provider
    return provider


def clear_provider_cache() -> None:
    """Clear provider caches (for testing)."""
    _chat_provider_cache.clear()
    _embedding_provider_cache.clear()
