"""Validated evaluation runner configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class EvaluationRunConfig(BaseModel):
    """Cutoffs shared by all query cases in one reproducible evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoffs: tuple[StrictInt, ...] = Field(default=(1, 3, 5), min_length=1)

    @field_validator("cutoffs")
    @classmethod
    def validate_cutoffs(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("Evaluation cutoffs must be positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("Evaluation cutoffs must be unique.")
        return values

    @property
    def max_cutoff(self) -> int:
        """Largest retrieval depth required to compute all configured metrics."""
        return max(self.cutoffs)
