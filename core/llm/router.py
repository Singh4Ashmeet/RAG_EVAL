"""Provider registry for LLM chat and embedding providers.

Supports fallback and retry mechanisms.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
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


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    enabled: bool = True
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    timeout_seconds: float = 30.0


class RetryProvider(LLMProvider):
    """Wraps an LLMProvider with exponential backoff retry logic."""

    def __init__(
        self,
        provider: LLMProvider,
        policy: RetryPolicy,
        provider_name: str,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._provider_name = provider_name

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", f"unknown:{self._provider_name}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        start_time = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(self._policy.max_attempts):
            # Check deadline before attempt (if not initial attempt)
            if attempt > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= self._policy.timeout_seconds:
                    logger.warning(
                        "Provider %s: Deadline exceeded after %d attempts, %.2fs elapsed",
                        self._provider_name,
                        attempt,
                        elapsed,
                    )
                    break

            try:
                return self._provider.generate(
                    system_prompt,
                    user_prompt,
                    settings=settings,
                )
            except Exception as exc:
                last_error = exc

                # If retries disabled or error is non-recoverable, fail immediately
                if not self._policy.enabled or not _is_recoverable_error(exc):
                    raise

                # Do not sleep after final attempt
                if attempt == self._policy.max_attempts - 1:
                    break

                # Calculate backoff delay
                retry_after = _extract_retry_after(exc)
                delay = _calculate_backoff(
                    attempt,
                    self._policy.base_delay_seconds,
                    self._policy.exponential_base,
                    self._policy.max_delay_seconds,
                    self._policy.jitter_enabled,
                    retry_after=retry_after,
                )

                # Check deadline with planned sleep
                elapsed = time.monotonic() - start_time
                remaining = self._policy.timeout_seconds - elapsed
                if remaining <= 0:
                    break
                actual_delay = min(delay, remaining)

                logger.warning(
                    "Provider %s: Attempt %d/%d failed (%.2fs), retrying in %.2fs: %s",
                    self._provider_name,
                    attempt + 1,
                    self._policy.max_attempts,
                    elapsed,
                    actual_delay,
                    exc,
                )
                time.sleep(actual_delay)

        # All retry attempts exhausted
        logger.error(
            "Provider %s: All %d retry attempts exhausted, last error: %s",
            self._provider_name,
            self._policy.max_attempts,
            last_error,
        )
        msg = (
            f"Provider {self._provider_name}: All {self._policy.max_attempts} "
            f"retry attempts exhausted. Last error: {last_error}"
        )
        raise ProviderFallbackExhaustedError(
            msg,
            failed_providers=[f"{self._provider_name}: {type(last_error).__name__}"],
        ) from last_error


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


def _calculate_backoff(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter_enabled: bool,
    retry_after: float | None = None,
) -> float:
    """Calculate backoff delay with exponential backoff and optional jitter."""
    if retry_after is not None:
        delay = min(retry_after, max_delay)
    else:
        delay = min(base_delay * (exponential_base**attempt), max_delay)

    if jitter_enabled:
        delay += random.uniform(0, delay * 0.1)

    return delay


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value from an exception if present."""
    response = getattr(exc, "response", None)
    if response is None:
        retry_after_attr = getattr(exc, "retry_after", None)
        if retry_after_attr is not None:
            try:
                return float(retry_after_attr)
            except (ValueError, TypeError):
                return None
        return None

    headers = getattr(response, "headers", None)
    if not headers or (not isinstance(headers, dict) and not hasattr(headers, "get")):
        return None

    retry_after_val = None
    if hasattr(headers, "get"):
        retry_after_val = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after_val is None:
        return None

    try:
        val = float(retry_after_val)
        if val >= 0:
            return val
    except (ValueError, TypeError):
        pass

    try:
        from datetime import UTC, datetime
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(retry_after_val)
        now = datetime.now(UTC)
        diff = (dt - now).total_seconds()
        return max(0.0, diff)
    except Exception:
        pass

    return None


def _is_recoverable_error(exc: Exception) -> bool:
    """Determine if an exception represents a recoverable/transient error."""
    if isinstance(exc, ProviderFallbackExhaustedError):
        if exc.__cause__ is not None:
            return _is_recoverable_error(exc.__cause__)
        return True

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


def _get_retry_policy(settings: Settings) -> RetryPolicy:
    """Extract retry policy from settings."""
    return RetryPolicy(
        enabled=settings.retry_enabled,
        max_attempts=settings.retry_max_attempts,
        base_delay_seconds=settings.retry_base_delay_seconds,
        max_delay_seconds=settings.retry_max_delay_seconds,
        exponential_base=settings.retry_exponential_base,
        jitter_enabled=settings.retry_jitter_enabled,
        timeout_seconds=settings.retry_timeout_seconds,
    )


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


def _wrap_provider_pipeline(
    provider_name: str, raw_provider: LLMProvider, settings: Settings
) -> LLMProvider:
    """Wrap raw provider with retry layer."""
    retry_policy = _get_retry_policy(settings)
    return RetryProvider(raw_provider, retry_policy, provider_name)


def get_chat_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Get a chat provider instance by name, wrapped with resilience features.

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
    wrapped_primary = _wrap_provider_pipeline(provider_name, raw_primary, settings)

    if settings.fallback_enabled:
        fallback_names = [
            f for f in _parse_fallback_providers(settings.fallback_providers) if f != provider_name
        ]
        wrapped_fallbacks = [
            _wrap_provider_pipeline(f_name, _create_raw_provider(f_name, settings), settings)
            for f_name in fallback_names
        ]
        provider = FallbackProvider(
            primary_provider=wrapped_primary,
            fallback_providers=wrapped_fallbacks,
            primary_name=provider_name,
            fallback_names=fallback_names,
        )
    else:
        provider = wrapped_primary

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
