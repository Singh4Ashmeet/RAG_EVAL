"""Language model generation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


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
