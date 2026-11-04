"""Provider registry for LLM chat and embedding providers."""

from __future__ import annotations

from core.llm.base import LLMProvider


class ProviderNotConfiguredError(Exception):
    """Raised when a provider is selected but not properly configured."""

    pass


class ProviderNotSupportedError(Exception):
    """Raised when an unsupported provider is requested."""

    pass


# Cache for provider instances
_chat_provider_cache: dict[str, LLMProvider] = {}
_embedding_provider_cache: dict[str, LLMProvider] = {}


def get_chat_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Get a chat provider instance by name.

    Args:
        provider_name: Provider name ("ollama", "gemini", "groq").
                       If None, uses CHAT_PROVIDER from settings.

    Returns:
        LLMProvider instance

    Raises:
        ProviderNotSupportedError: If provider name is unknown.
        ProviderNotConfiguredError: If provider is not properly configured.
    """
    from core.config import get_settings

    settings = get_settings()

    provider_name = provider_name or settings.chat_provider

    cache_key = f"chat:{provider_name}"
    if cache_key in _chat_provider_cache:
        return _chat_provider_cache[cache_key]

    if provider_name == "ollama":
        from core.llm.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider(
            model=settings.chat_model,
            host=settings.chat_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "gemini":
        from core.llm.gemini_provider import GeminiLLMProvider

        provider = GeminiLLMProvider(
            model=settings.gemini_chat_model,
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "groq":
        from core.llm.groq_provider import GroqLLMProvider

        provider = GroqLLMProvider(
            model=settings.groq_chat_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.rq_job_timeout,
        )
    else:
        raise ProviderNotSupportedError(
            f"Unsupported chat provider: {provider_name}. Supported providers: ollama, gemini, groq"
        )

    _chat_provider_cache[provider_name] = provider
    return provider


def get_embedding_provider(provider_name: str | None = None):
    """
    Get an embedding provider instance by name.

    Args:
        provider_name: Provider name ("ollama"). If None, uses EMBEDDING_PROVIDER from settings.

    Returns:
        LLMProvider instance (wraps embedding functionality)

    Raises:
        ProviderNotSupportedError: If provider name is unknown.
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


# Cache for provider instances
_chat_provider_cache: dict[str, LLMProvider] = {}
_embedding_provider_cache: dict[str, LLMProvider] = {}
