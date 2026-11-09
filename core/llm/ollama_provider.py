"""Local chat generation provider backed by Ollama."""

from __future__ import annotations

import logging
from typing import Any

import ollama

from core.errors import LLMError, ProviderDeadlineExceededError
from core.llm.base import GenerationResponse, GenerationSettings
from core.llm.deadline import GenerationDeadline

logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    """Generate non-streaming chat responses using an available Ollama model.

    The provider does not start Ollama or download models. Model availability is
    checked by the first request and reported as :class:`LLMError` on failure.
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        client: Any | None = None,
        *,
        timeout: float | None = 180.0,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise LLMError("An Ollama generation model must be configured.")
        self.model = cleaned_model
        self.model_id = f"ollama:{cleaned_model}"
        if client is not None:
            self._client = client
        elif timeout is not None:
            self._client = ollama.Client(host=host, timeout=timeout)
        else:
            self._client = ollama.Client(host=host)

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,
    ) -> GenerationResponse:
        """Return a complete assistant message with token usage metadata."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        # Determine effective timeout: explicit timeout > client's configured timeout > None
        effective_timeout = timeout
        if effective_timeout is None:
            # Try to get timeout from the client's underlying httpx client
            if hasattr(self._client, "_client") and hasattr(self._client._client, "timeout"):
                effective_timeout = self._client._client.timeout

        # Create a client with the effective timeout for this request
        # We need to create a new client if the timeout differs from the client's timeout
        owns_client = False
        if effective_timeout is not None:
            # Check if the client is a real ollama.Client (has _client attribute with timeout)
            if hasattr(self._client, "_client") and hasattr(self._client._client, "timeout"):
                client_timeout = self._client._client.timeout
                if effective_timeout != client_timeout:
                    # Create a new client with the effective timeout
                    # base_url is an httpx.URL object, convert to string
                    base_url = str(self._client._client.base_url)
                    client = ollama.Client(host=base_url, timeout=effective_timeout)
                    owns_client = True
                else:
                    client = self._client
            else:
                # Fake client or client without _client attribute - use as is
                client = self._client
        else:
            client = self._client

        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options={
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens,
                },
            )
        except Exception as exc:
            raise LLMError(f"Ollama chat request failed for model '{self.model}': {exc}") from exc
        finally:
            # Clean up request-specific client if we created one
            if owns_client and hasattr(client, "_client") and hasattr(client._client, "close"):
                try:
                    client._client.close()
                except Exception:
                    # Cleanup errors are logged but do not mask the original generation
                    # exception (if any) or the successful response.
                    # This follows the principle that resource cleanup should not
                    # override the primary operation's outcome.
                    logger.exception(
                        "Failed to close request-specific Ollama client for model %s",
                        self.model,
                    )

        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Ollama model '{self.model}' returned an empty response.")

        # Extract token usage from Ollama response
        # Ollama returns: prompt_eval_count (input), eval_count (output)
        input_tokens = getattr(response, "prompt_eval_count", None)
        output_tokens = getattr(response, "eval_count", None)

        # Normalize: convert to int if present and valid, otherwise None
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

        input_tokens = _normalize_token_count(input_tokens)
        output_tokens = _normalize_token_count(output_tokens)

        # total_tokens: only compute if both are valid integers
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return GenerationResponse(
            text=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

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

        # Create async client with the request timeout
        # Ollama's AsyncClient accepts timeout in constructor
        base_url = None
        if hasattr(self._client, "_client"):
            base_url = self._client._client.base_url
        if request_timeout is not None:
            async_client = ollama.AsyncClient(host=base_url, timeout=request_timeout)
        else:
            async_client = ollama.AsyncClient(host=base_url)

        try:
            response = await async_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options={
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens,
                },
            )
        except Exception as exc:
            raise LLMError(f"Ollama chat request failed for model '{self.model}': {exc}") from exc

        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Ollama model '{self.model}' returned an empty response.")

        # Extract token usage from Ollama response
        # Ollama returns: prompt_eval_count (input), eval_count (output)
        input_tokens = getattr(response, "prompt_eval_count", None)
        output_tokens = getattr(response, "eval_count", None)

        # Normalize: convert to int if present and valid, otherwise None
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

        input_tokens = _normalize_token_count(input_tokens)
        output_tokens = _normalize_token_count(output_tokens)

        # total_tokens: only compute if both are valid integers
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return GenerationResponse(
            text=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    async def aclose(self) -> None:
        """Release async client resources. Idempotent."""
        # The Ollama sync client's underlying httpx client can be closed
        if hasattr(self._client, "_client") and hasattr(self._client._client, "aclose"):
            try:
                await self._client._client.aclose()
            except Exception:
                logger.exception(
                    "Failed to close Ollama client for model %s",
                    self.model,
                )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Return a complete assistant message for the supplied prompts."""
        response = self.generate_with_metadata(system_prompt, user_prompt, settings=settings)
        return response.text
