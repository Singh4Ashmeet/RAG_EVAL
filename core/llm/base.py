"""Language model generation contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


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
