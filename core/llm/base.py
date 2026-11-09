"""Language model generation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from core.llm.deadline import GenerationDeadline


@dataclass(frozen=True)
class GenerationResponse:
    """Standardized generation response carrying text and optional token metadata.

    Missing token fields remain None. Zero is a valid token count.
    total_tokens is computed from input_tokens + output_tokens when both are provided.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_tokens is not None and self.output_tokens is not None:
            if self.total_tokens is None:
                object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)


class GenerationSettings(BaseModel):
    """Provider-neutral generation parameters supported by local providers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


@runtime_checkable
class LLMProvider(Protocol):
    """Generate one text response from a system instruction and user prompt."""

    @property
    def model_id(self) -> str:
        """Stable provider/model label to record with generated output."""
        ...

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        """Generate non-streaming text or raise a provider-specific domain error."""
        ...


@runtime_checkable
class AsyncCloseable(Protocol):
    """Protocol for resources that support asynchronous cleanup."""

    async def aclose(self) -> None:
        """Release asynchronous resources. Must be idempotent."""
        ...


@runtime_checkable
class AsyncLLMProvider(AsyncCloseable, Protocol):
    """Async contract for LLM generation with deadline propagation.

    This protocol defines the asynchronous generation interface with deadline
    propagation support. Implementations must be safe for concurrent use
    and must not store request-specific state in instance attributes.

    The `generate_async` method is NOT part of this protocol; it is a
    convenience method that implementations may optionally provide.
    """

    @property
    def model_id(self) -> str:
        """Stable provider/model label to record with generated output."""
        ...

    async def generate_with_metadata_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        """Generate a non-streaming chat response with token usage metadata.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User message for the model.
            settings: Generation parameters (temperature, max_tokens).
            timeout: Optional relative timeout in seconds. Used only when
                `deadline` is not provided. If both are supplied, `deadline`
                takes precedence and `timeout` is ignored.
            deadline: Absolute deadline for the entire logical request.
                Propagated through all wrapper layers. If not provided,
                a deadline is created from `timeout` if given.

        Returns:
            GenerationResponse with text and token metadata.

        Raises:
            ProviderDeadlineExceededError: If the deadline expires before
                or during generation.
            LLMError: If the provider request fails for other reasons.
        """
        ...

    async def aclose(self) -> None:
        """Release asynchronous resources. Must be idempotent.

        Called during application shutdown to release provider-specific
        resources such as connection pools or client sessions.
        """
        ...
