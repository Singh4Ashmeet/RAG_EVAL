"""Gemini LLM provider using Google Generative AI SDK."""

from __future__ import annotations

import httpx
from google import genai
from google.genai import types

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


class GeminiLLMProvider:
    """Gemini chat completion provider using Google Generative AI SDK."""

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
            raise ValueError("Gemini model must be specified")
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key must be provided")

        self.model = cleaned_model
        self.model_id = f"gemini:{cleaned_model}"
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout

        # Initialize the GenAI client
        client_config = {"api_key": self._api_key}
        if self._base_url:
            # For Vertex AI or custom endpoints
            client_config = {"api_key": self._api_key, "http_options": {"base_url": self._base_url}}
        else:
            client_config = {"api_key": self._api_key}

        self._client = genai.Client(**client_config)

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
        elif "API key" in str(exc) or "API_KEY" in str(exc):
            return ProviderAuthenticationError("Gemini authentication failed: invalid API key")
        elif "quota" in str(exc).lower() or "rate limit" in str(exc).lower():
            return ProviderRateLimitError(f"Gemini rate limit: {exc}")
        elif "timeout" in str(exc).lower():
            return ProviderUnavailableError("Gemini request timed out")
        else:
            return LLMError(f"Gemini error: {exc}")

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

        # Combine system and user prompts for Gemini
        # Gemini expects a single conversation with user/assistant roles
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.temperature,
                    max_output_tokens=settings.max_tokens,
                ),
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        # Extract text from response
        text = response.text
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response.")

        return text.strip()

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

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = await self._async_client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.temperature,
                    max_output_tokens=settings.max_tokens,
                ),
            )
        except Exception as exc:
            raise self._handle_error(exc) from exc

        text = response.text
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response.")

        return text.strip()
