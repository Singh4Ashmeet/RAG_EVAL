"""Gemini LLM provider using Google Generative AI SDK."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from google import genai
from google.genai import types

from core.errors import LLMError, ProviderDeadlineExceededError
from core.llm.base import GenerationResponse, GenerationSettings
from core.llm.deadline import GenerationDeadline

logger = logging.getLogger(__name__)


class ProviderAuthenticationError(LLMError):
    """Raised when provider authentication fails."""

    pass


class ProviderRateLimitError(LLMError):
    """Raised when provider rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailableError(LLMError):
    """Raised when provider is temporarily unavailable."""

    pass


class ProviderResponseError(LLMError):
    """Raised when provider returns an invalid/unexpected response."""

    pass


class GeminiLLMProvider:
    """Gemini chat completion provider using Google Generative AI SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        *,
        timeout: float | None = 60.0,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ValueError("Gemini model must be specified")
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key must be provided")

        self.model = cleaned_model
        self.model_id = f"gemini:{cleaned_model}"
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout

        # Initialize the GenAI client (allow injection for testing)
        if client is not None:
            self._client = client
            self._async_client = async_client if async_client is not None else client
        else:
            client_config = {"api_key": self._api_key}
            if self._base_url:
                # For Vertex AI or custom endpoints
                client_config = {
                    "api_key": self._api_key,
                    "http_options": {"base_url": self._base_url},
                }
            else:
                client_config = {"api_key": self._api_key}
            self._client = genai.Client(**client_config)
            self._async_client = genai.Client(**client_config)

    def _handle_error(self, exc: Exception) -> LLMError:
        """Convert provider exceptions to normalized errors."""
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                return ProviderAuthenticationError(
                    "Gemini authentication failed: invalid or missing API key"
                )
            elif status == 429:
                # Try to extract Retry-After header
                retry_after = None
                if "retry-after" in exc.response.headers:
                    try:
                        retry_after = float(exc.response.headers["retry-after"])
                    except ValueError:
                        pass  # Could be HTTP-date format
                return ProviderRateLimitError(
                    f"Gemini rate limit exceeded: {exc.response.text}",
                    retry_after=retry_after,
                )
            elif status >= 500:
                return ProviderUnavailableError(f"Gemini service unavailable: {exc.response.text}")
            else:
                return ProviderResponseError(
                    f"Gemini request failed ({status}): {exc.response.text}"
                )
        elif isinstance(exc, httpx.TimeoutException):
            return ProviderUnavailableError("Gemini request timed out")
        elif isinstance(exc, httpx.RequestError):
            return ProviderUnavailableError(f"Gemini connection failed: {exc}")
        elif "API key" in str(exc).lower() or "API_KEY" in str(exc).lower():
            return ProviderAuthenticationError("Gemini authentication failed: invalid API key")
        elif "quota" in str(exc).lower() or "rate limit" in str(exc).lower():
            return ProviderRateLimitError(f"Gemini rate limit: {exc}")
        elif "timeout" in str(exc).lower():
            return ProviderUnavailableError("Gemini request timed out")
        else:
            return LLMError(f"Gemini error: {exc}")

    def _extract_token_usage(self, response: Any) -> tuple[int | None, int | None, int | None]:
        """Extract token usage from Gemini response.

        Returns (input_tokens, output_tokens, total_tokens).
        """
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return None, None, None

        def _normalize_token_count(value: Any) -> int | None:
            if value is None:
                return None
            try:
                ival = int(value)
                if ival < 0:
                    return None
                return ival
            except (ValueError, TypeError):
                return None

        input_tokens = _normalize_token_count(getattr(usage_metadata, "prompt_token_count", None))
        output_tokens = _normalize_token_count(
            getattr(usage_metadata, "candidates_token_count", None)
        )
        total_tokens = _normalize_token_count(getattr(usage_metadata, "total_token_count", None))

        # If total_tokens is missing but we have both input and output, compute it
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return input_tokens, output_tokens, total_tokens

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,
    ) -> GenerationResponse:
        """Generate a non-streaming chat response with token usage metadata."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        # Combine system and user prompts for Gemini
        # Gemini expects a single conversation with user/assistant roles
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Build config with timeout if provided
        config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
        )
        if timeout is not None:
            # Gemini timeout is in milliseconds
            timeout_ms = int(timeout * 1000)
            config.http_options = types.HttpOptions(timeout=timeout_ms)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config,
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        # Extract text from response
        text = response.text
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response.")

        input_tokens, output_tokens, total_tokens = self._extract_token_usage(response)

        return GenerationResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Generate a non-streaming chat response from system and user prompts."""
        response = self.generate_with_metadata(system_prompt, user_prompt, settings=settings)
        return response.text

    async def generate_with_metadata_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        """Async version of generate_with_metadata with deadline propagation."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        # Handle deadline
        if deadline is not None:
            if deadline.expired:
                raise ProviderDeadlineExceededError(
                    "generate_with_metadata_async: deadline exceeded"
                )
            remaining = deadline.remaining_seconds()
            if remaining <= 0:
                raise ProviderDeadlineExceededError(
                    "generate_with_metadata_async: deadline exceeded"
                )
            # Use the remaining deadline as the timeout for this request
            request_timeout = remaining
        else:
            # No deadline provided, use explicit timeout if given
            request_timeout = timeout

        if request_timeout is not None and request_timeout <= 0:
            raise ProviderDeadlineExceededError(
                "generate_with_metadata_async: timeout must be positive"
            )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Build config with timeout if provided
        config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
        )
        if request_timeout is not None:
            # Gemini timeout is in milliseconds
            timeout_ms = int(request_timeout * 1000)
            config.http_options = types.HttpOptions(timeout=timeout_ms)

        try:
            response = await self._async_client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config,
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        # Check deadline again after response
        if deadline is not None and deadline.expired:
            raise ProviderDeadlineExceededError(
                "generate_with_metadata_async: deadline exceeded after response"
            )

        # Extract text from response
        text = response.text
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response.")

        input_tokens, output_tokens, total_tokens = self._extract_token_usage(response)

        return GenerationResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> str:
        """Async version of generate for async contexts."""
        response = await self.generate_with_metadata_async(
            system_prompt, user_prompt, settings=settings, timeout=timeout, deadline=deadline
        )
        return response.text

    async def aclose(self) -> None:
        """Release async client resources. Idempotent."""
        if hasattr(self._async_client, "aclose"):
            try:
                await self._async_client.aclose()
            except Exception:
                logger.exception(
                    "Failed to close Gemini async client for model %s",
                    self.model,
                )
