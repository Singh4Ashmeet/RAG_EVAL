"""Deadline management for generation requests."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from core.errors import ProviderDeadlineExceededError


@dataclass(frozen=True, slots=True)
class GenerationDeadline:
    """Immutable absolute deadline based on monotonic time.

    Represents a single deadline for an entire logical generation request,
    shared across retries, circuit breaker, and fallback providers.

    The deadline is an absolute monotonic timestamp, not a relative timeout.
    Once created, the deadline is immutable and cannot be extended.

    Deadline ownership:
    - The public request entry point creates exactly one absolute deadline.
    - The same deadline object is propagated through every async wrapper
      and provider in the call chain.
    - Nested wrappers and providers MUST NOT reset, recreate, or extend
      the deadline.
    - An explicit deadline takes precedence over a relative timeout.
    - No code may extend an expired deadline or grant a minimum
      replacement budget.
    - A provider receiving no deadline must not independently create
      a replacement deadline as part of the wrapper pipeline.
    """

    _expiration_monotonic: float

    @classmethod
    def after(cls, timeout_seconds: float) -> GenerationDeadline:
        """Create a deadline from a timeout duration.

        Args:
            timeout_seconds: Timeout in seconds. Must be positive and finite.

        Returns:
            GenerationDeadline instance.

        Raises:
            ValueError: If timeout_seconds is not positive or not finite.
        """
        if not isinstance(timeout_seconds, (int, float)):
            raise ValueError(
                f"timeout_seconds must be a number, got {type(timeout_seconds).__name__}"
            )
        if not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return cls(_expiration_monotonic=time.monotonic() + timeout_seconds)

    @classmethod
    def from_expiration(cls, expiration_monotonic: float) -> GenerationDeadline:
        """Create a deadline from an absolute monotonic expiration timestamp.

        This is an internal method for sharing the exact same deadline
        across wrappers.

        Args:
            expiration_monotonic: Absolute monotonic timestamp when
                the deadline expires.

        Returns:
            GenerationDeadline instance.
        """
        return cls(_expiration_monotonic=expiration_monotonic)

    def remaining_seconds(self) -> float:
        """Return remaining seconds until deadline, clamped at zero.

        Returns:
            Remaining seconds as a non-negative float.
        """
        remaining = self._expiration_monotonic - time.monotonic()
        return max(0.0, remaining)

    @property
    def expired(self) -> bool:
        """Check if the deadline has passed.

        Returns:
            True if deadline has expired, False otherwise.
        """
        return time.monotonic() >= self._expiration_monotonic

    def check_or_raise(self, context: str = "deadline") -> None:
        """Check deadline and raise if expired.

        Args:
            context: Description of what is checking the deadline
                (for error message).

        Raises:
            ProviderDeadlineExceededError: If the deadline has expired.
        """
        if self.expired:
            overrun = self._expiration_monotonic - time.monotonic()
            # Clamp overrun to zero to avoid negative values due to
            # floating-point precision
            overrun = max(0.0, overrun)
            raise ProviderDeadlineExceededError(
                f"{context}: deadline exceeded after {overrun:.2f}s overrun"
            )
