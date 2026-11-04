"""Local chat generation provider backed by Ollama."""

from __future__ import annotations

from typing import Any

import ollama

from core.errors import LLMError
from core.llm.base import GenerationSettings


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

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Return a complete assistant message for the supplied prompts."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise LLMError("System and user prompts must be non-empty.")

        try:
            response = self._client.chat(
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
        return content.strip()
