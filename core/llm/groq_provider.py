"""Groq LLM provider using Groq Python SDK."""

from __future__ import annotations

import httpx
from groq import AsyncGroq, Groq

from core.errors import LLMError
from core.llm.base import GenerationSettings


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


class GroqLLMProvider:
    """Groq chat completion provider using Groq Python SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        *,
        timeout: float | None = 60.0,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ValueError("Groq model must be specified")
        if not api_key or not api_key.strip():
            raise ValueError("Groq API key must be provided")

        self.model = cleaned_model
        self.model_id = f"groq:{cleaned_model}"
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout

        # Initialize Groq clients
        client_config = {"api_key": self._api_key}
        if self._base_url:
            client_config["base_url"] = self._base_url

        self._client = Groq(**client_config, timeout=timeout)
        self._async_client = AsyncGroq(**client_config, timeout=timeout)

    def _handle_error(self, exc: Exception) -> LLMError:
        """Convert provider exceptions to normalized errors."""
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                return ProviderAuthenticationError(
                    "Groq authentication failed: invalid or missing API key"
                )
            elif status == 429:
                # Try to extract Retry-After header
                retry_after = None
                if "retry-after" in exc.response.headers:
                    try:
                        retry_after = float(exc.response.headers["retry-after"])
                    except ValueError:
                        pass
                return ProviderRateLimitError(
                    f"Groq rate limit exceeded: {exc.response.text}",
                    retry_after=retry_after,
                )
            elif status >= 500:
                return ProviderUnavailableError(f"Groq service unavailable: {exc.response.text}")
            else:
                return ProviderResponseError(f"Groq request failed ({status}): {exc.response.text}")
        elif isinstance(exc, httpx.TimeoutException):
            return ProviderUnavailableError("Groq request timed out")
        elif isinstance(exc, httpx.RequestError):
            return ProviderUnavailableError(f"Groq connection failed: {exc}")
        elif "authentication" in str(exc).lower() or "unauthorized" in str(exc).lower():
            return ProviderAuthenticationError("Groq authentication failed: invalid API key")
        elif "rate limit" in str(exc).lower():
            return ProviderRateLimitError(f"Groq rate limit: {exc}")
        elif "timeout" in str(exc).lower():
            return ProviderUnavailableError("Groq request timed out")
        else:
            return LLMError(f"Groq error: {exc}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Generate a non-streaming chat response from system and user prompts."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=False,
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Groq returned an empty response.")

        return content.strip()

    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Async version of generate for async contexts."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=False,
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Groq returned an empty response.")

        return content.strip()
