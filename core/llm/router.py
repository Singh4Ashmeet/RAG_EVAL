"""Provider registry for LLM chat and embedding providers.

Supports fallback, retry, and circuit breaker mechanisms.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.errors import LLMError
from core.llm.base import GenerationResponse, GenerationSettings, LLMProvider
from core.llm.metrics import get_metrics_collector

if TYPE_CHECKING:
    from core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationDeadline:
    """Immutable absolute deadline based on monotonic time.

    Represents a single deadline for an entire logical generation request,
    shared across retries, circuit breaker, and fallback providers.
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

        This is an internal method for sharing the exact same deadline across wrappers.

        Args:
            expiration_monotonic: Absolute monotonic timestamp when the deadline expires.

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
            context: Description of what is checking the deadline (for error message).

        Raises:
            ProviderDeadlineExceededError: If the deadline has expired.
        """
        if self.expired:
            overrun = self._expiration_monotonic - time.monotonic()
            # Clamp overrun to zero to avoid negative values due to floating-point precision
            overrun = max(0.0, overrun)
            raise ProviderDeadlineExceededError(
                f"{context}: deadline exceeded after {overrun:.2f}s overrun"
            )


def _call_generate_with_metadata(
    provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    *,
    settings: GenerationSettings,
    deadline: GenerationDeadline | None = None,
) -> GenerationResponse:
    """Call provider's metadata-aware generation method if available,
    otherwise wrap string response.

    This compatibility layer supports both metadata-aware providers (with
    generate_with_metadata) and legacy providers (with only generate). For
    legacy providers, the string response is wrapped in a GenerationResponse
    with token fields set to None.

    If a deadline is provided, the remaining time is used to enforce a timeout
    on the provider SDK call.
    """
    # Calculate remaining timeout for the provider SDK call
    remaining_timeout = None
    if deadline is not None:
        remaining = deadline.remaining_seconds()
        if remaining <= 0:
            deadline.check_or_raise("_call_generate_with_metadata")
        remaining_timeout = remaining

    # First, try to call the internal deadline-aware method if available.
    # Wrapper providers (RetryProvider, CircuitBreakerProvider, FallbackProvider)
    # implement this method to propagate the deadline without creating a new one.
    if hasattr(provider, "_generate_with_metadata_with_deadline"):
        return provider._generate_with_metadata_with_deadline(
            system_prompt,
            user_prompt,
            settings=settings,
            deadline=deadline,
        )

    # Fall back to public generate_with_metadata with timeout.
    # This handles raw providers (Ollama, Gemini, Groq) and legacy providers.
    if hasattr(provider, "generate_with_metadata"):
        import inspect

        sig = inspect.signature(provider.generate_with_metadata)
        if "timeout" in sig.parameters:
            return provider.generate_with_metadata(
                system_prompt,
                user_prompt,
                settings=settings,
                timeout=remaining_timeout,
            )
        return provider.generate_with_metadata(
            system_prompt,
            user_prompt,
            settings=settings,
        )
    # Legacy provider: call generate() and wrap in GenerationResponse
    text = provider.generate(system_prompt, user_prompt, settings=settings)
    return GenerationResponse(text=text)


class ProviderNotConfiguredError(Exception):
    """Raised when a provider is selected but not properly configured."""

    pass


class ProviderNotSupportedError(Exception):
    """Raised when an unsupported provider is requested."""

    pass


class ProviderFallbackExhaustedError(LLMError):
    """Raised when all fallback providers have been exhausted."""

    def __init__(self, message: str, failed_providers: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed_providers = failed_providers or []


class CircuitBreakerOpenError(LLMError):
    """Raised when a request is attempted against a provider whose circuit breaker is open."""

    def __init__(self, provider_name: str, message: str | None = None) -> None:
        super().__init__(message or f"Circuit breaker is OPEN for provider '{provider_name}'")
        self.provider_name = provider_name


class ProviderDeadlineExceededError(LLMError):
    """Raised when a generation request exceeds its absolute deadline.

    This is not a recoverable error - it indicates the overall deadline
    for the logical request has been exceeded.
    """

    pass


# Cache for provider instances and circuit breakers
_chat_provider_cache: dict[str, LLMProvider] = {}
_embedding_provider_cache: dict[str, LLMProvider] = {}
_circuit_breakers: dict[str, CircuitBreaker] = {}


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    enabled: bool = True
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    timeout_seconds: float = 30.0


class CircuitBreakerState:
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerPolicy:
    """Configuration for circuit breaker behavior."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    window_seconds: float = 60.0


class CircuitBreakerPermit:
    """Represents an execution permit granted by the circuit breaker for one request."""

    def __init__(
        self,
        breaker: CircuitBreaker,
        state: str,
        epoch: int,
        is_probe: bool,
    ) -> None:
        self.breaker = breaker
        self.state = state
        self.epoch = epoch
        self.is_probe = is_probe
        self._resolved = False
        self._lock = threading.Lock()

    @property
    def is_resolved(self) -> bool:
        with self._lock:
            return self._resolved

    def record_success(self) -> None:
        """Resolve this permit as a successful request."""
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
        self.breaker._resolve_success(self)

    def record_failure(self, exc: Exception) -> None:
        """Resolve this permit as a failed request."""
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
        self.breaker._resolve_failure(self, exc)

    def release(self) -> None:
        """Release this permit if execution was aborted, cancelled, or interrupted."""
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
        self.breaker._resolve_release(self)


class CircuitBreaker:
    """
    Thread-safe circuit breaker for provider failure isolation.

    Implements a three-state circuit breaker:
    - CLOSED: Normal operation, requests allowed, failures tracked
    - OPEN: Requests rejected immediately, recovery timer started
    - HALF_OPEN: Limited probe requests allowed to test recovery

    Thread-safe using threading.Lock. Never holds lock across external HTTP calls.
    """

    def __init__(
        self,
        provider_name: str,
        policy: CircuitBreakerPolicy,
        clock: Any = None,
    ) -> None:
        self._provider_name = provider_name
        self._policy = policy
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CircuitBreakerState.CLOSED
        self._epoch = 0
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._last_state_change = self._clock()

    @property
    def state(self) -> str:
        with self._lock:
            self._check_state_transition_locked()
            return self._state

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _is_failure_recoverable(self, exc: Exception) -> bool:
        """Determine if an exception counts as a circuit breaker failure."""
        if isinstance(exc, ProviderFallbackExhaustedError) and exc.__cause__ is not None:
            return _is_recoverable_error(exc.__cause__)
        return _is_recoverable_error(exc)

    def _check_state_transition_locked(self) -> None:
        """Transition from OPEN to HALF_OPEN if recovery timeout has elapsed."""
        if self._state == CircuitBreakerState.OPEN:
            now = self._clock()
            if now - self._last_state_change >= self._policy.recovery_timeout_seconds:
                self._state = CircuitBreakerState.HALF_OPEN
                self._epoch += 1
                self._half_open_calls = 0
                self._success_count = 0
                self._last_state_change = now
                logger.info(
                    "Circuit breaker HALF_OPEN for provider %s, allowing probe requests",
                    self._provider_name,
                )

    def acquire_permit(self) -> CircuitBreakerPermit | None:
        """
        Atomically check state and acquire an execution permit.
        Returns a CircuitBreakerPermit if allowed, or None if rejected.
        """
        if not self._policy.enabled:
            return CircuitBreakerPermit(
                breaker=self,
                state=CircuitBreakerState.CLOSED,
                epoch=self._epoch,
                is_probe=False,
            )

        with self._lock:
            self._check_state_transition_locked()

            if self._state == CircuitBreakerState.CLOSED:
                return CircuitBreakerPermit(
                    breaker=self,
                    state=CircuitBreakerState.CLOSED,
                    epoch=self._epoch,
                    is_probe=False,
                )

            if self._state == CircuitBreakerState.OPEN:
                return None

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self._policy.half_open_max_calls:
                    self._half_open_calls += 1
                    collector = get_metrics_collector()
                    collector.record_half_open_probe(self._provider_name)
                    return CircuitBreakerPermit(
                        breaker=self,
                        state=CircuitBreakerState.HALF_OPEN,
                        epoch=self._epoch,
                        is_probe=True,
                    )
                return None

            return None

    def allow_request(self) -> bool:
        """
        Check if a request is allowed to proceed.
        Acquires a permit if allowed.
        """
        permit = self.acquire_permit()
        return permit is not None

    def can_execute(self) -> bool:
        """Query if a request can be executed without incrementing probe calls."""
        if not self._policy.enabled:
            return True

        with self._lock:
            self._check_state_transition_locked()
            if self._state == CircuitBreakerState.CLOSED:
                return True
            if self._state == CircuitBreakerState.OPEN:
                return False
            if self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self._policy.half_open_max_calls
            return False

    def _resolve_success(self, permit: CircuitBreakerPermit) -> None:
        """Resolve a successful probe or closed request from its permit."""
        if not self._policy.enabled:
            return

        with self._lock:
            if permit.epoch != self._epoch:
                logger.debug(
                    "Ignoring stale probe success for provider %s (permit epoch %d != current %d)",
                    self._provider_name,
                    permit.epoch,
                    self._epoch,
                )
                return

            now = self._clock()
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._policy.half_open_max_calls:
                    self._state = CircuitBreakerState.CLOSED
                    self._epoch += 1
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._last_state_change = now
                    logger.info(
                        "Circuit breaker CLOSED for provider %s after successful recovery",
                        self._provider_name,
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def _resolve_failure(self, permit: CircuitBreakerPermit, exc: Exception) -> None:
        """Resolve a failed probe or closed request from its permit."""
        if not self._policy.enabled:
            return

        is_recoverable = self._is_failure_recoverable(exc)

        with self._lock:
            if permit.epoch != self._epoch:
                logger.debug(
                    "Ignoring stale probe failure for provider %s (permit epoch %d != current %d)",
                    self._provider_name,
                    permit.epoch,
                    self._epoch,
                )
                return

            now = self._clock()
            self._last_failure_time = now

            if self._state == CircuitBreakerState.HALF_OPEN:
                if is_recoverable:
                    self._state = CircuitBreakerState.OPEN
                    self._epoch += 1
                    self._failure_count = self._policy.failure_threshold
                    self._half_open_calls = 0
                    self._success_count = 0
                    self._last_state_change = now
                    collector = get_metrics_collector()
                    collector.record_circuit_opening(self._provider_name)
                    logger.warning(
                        "Circuit breaker returned to OPEN for provider %s after probe failure: %s",
                        self._provider_name,
                        exc,
                    )
                else:
                    # Non-recoverable error in HALF_OPEN: release probe permit
                    if self._half_open_calls > 0:
                        self._half_open_calls -= 1
                    logger.info(
                        "Probe for provider %s hit non-recoverable error %s; permit released",
                        self._provider_name,
                        exc,
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                if is_recoverable:
                    self._failure_count += 1
                    if self._failure_count >= self._policy.failure_threshold:
                        self._state = CircuitBreakerState.OPEN
                        self._epoch += 1
                        self._last_state_change = now
                        collector = get_metrics_collector()
                        collector.record_circuit_opening(self._provider_name)
                        logger.warning(
                            "Circuit breaker OPEN for provider %s after %d consecutive failures",
                            self._provider_name,
                            self._failure_count,
                        )

    def _resolve_release(self, permit: CircuitBreakerPermit) -> None:
        """Release an unresolved permit due to abort, cancellation, or exception."""
        if not self._policy.enabled:
            return

        with self._lock:
            if permit.epoch != self._epoch:
                return

            if self._state == CircuitBreakerState.HALF_OPEN and permit.is_probe:
                if self._half_open_calls > 0:
                    self._half_open_calls -= 1
                logger.info(
                    "Probe permit released for provider %s due to cancellation or exception",
                    self._provider_name,
                )

    def record_success(self, permit: CircuitBreakerPermit | None = None) -> None:
        """Record a successful provider request."""
        if permit is not None:
            permit.record_success()
            return

        if not self._policy.enabled:
            return

        with self._lock:
            self._check_state_transition_locked()
            now = self._clock()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._policy.half_open_max_calls:
                    self._state = CircuitBreakerState.CLOSED
                    self._epoch += 1
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._last_state_change = now
                    logger.info(
                        "Circuit breaker CLOSED for provider %s after successful recovery",
                        self._provider_name,
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def record_failure(self, exc: Exception, permit: CircuitBreakerPermit | None = None) -> None:
        """Record a failed provider request if recoverable."""
        if permit is not None:
            permit.record_failure(exc)
            return

        if not self._policy.enabled:
            return

        is_recoverable = self._is_failure_recoverable(exc)

        with self._lock:
            self._check_state_transition_locked()
            now = self._clock()
            self._last_failure_time = now

            if self._state == CircuitBreakerState.HALF_OPEN:
                if is_recoverable:
                    self._state = CircuitBreakerState.OPEN
                    self._epoch += 1
                    self._failure_count = self._policy.failure_threshold
                    self._half_open_calls = 0
                    self._success_count = 0
                    self._last_state_change = now
                    collector = get_metrics_collector()
                    collector.record_circuit_opening(self._provider_name)
                    logger.warning(
                        "Circuit breaker returned to OPEN for provider %s after probe failure: %s",
                        self._provider_name,
                        exc,
                    )
                else:
                    if self._half_open_calls > 0:
                        self._half_open_calls -= 1
                    logger.info(
                        "Probe for provider %s hit non-recoverable error %s; permit released",
                        self._provider_name,
                        exc,
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                if is_recoverable:
                    self._failure_count += 1
                    if self._failure_count >= self._policy.failure_threshold:
                        self._state = CircuitBreakerState.OPEN
                        self._epoch += 1
                        self._last_state_change = now
                        collector = get_metrics_collector()
                        collector.record_circuit_opening(self._provider_name)
                        logger.warning(
                            "Circuit breaker OPEN for provider %s after %d consecutive failures",
                            self._provider_name,
                            self._failure_count,
                        )

    def reset(self) -> None:
        """Reset the circuit breaker to clean CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._epoch += 1
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._last_state_change = self._clock()

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        with self._lock:
            self._check_state_transition_locked()
            return {
                "provider": self._provider_name,
                "state": self._state,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
            }


class CircuitBreakerProvider(LLMProvider):
    """Wraps an LLMProvider with circuit breaker protection."""

    def __init__(
        self,
        provider: LLMProvider,
        circuit_breaker: CircuitBreaker,
        provider_name: str,
    ) -> None:
        self._provider = provider
        self._circuit_breaker = circuit_breaker
        self._provider_name = provider_name

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", f"unknown:{self._provider_name}")

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._get_default_timeout()
            deadline = GenerationDeadline.after(effective_timeout)
        return self._generate_with_metadata_with_deadline(
            system_prompt, user_prompt, settings=settings, deadline=deadline
        )

    def _get_default_timeout(self) -> float:
        # Try to get timeout from the wrapped provider's retry policy if available
        if hasattr(self._provider, "_policy"):
            return self._provider._policy.timeout_seconds
        # Default to a reasonable timeout
        return 30.0

    def _generate_with_metadata_with_deadline(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # Allow deadline injection for testing; otherwise create fresh deadline
        if deadline is None:
            deadline = GenerationDeadline.after(self._get_default_timeout())
        # Check deadline before acquiring permit
        deadline.check_or_raise("circuit breaker")

        permit = self._circuit_breaker.acquire_permit()
        if permit is None:
            collector = get_metrics_collector()
            collector.record_circuit_rejection(self._provider_name, self.model_id)
            raise CircuitBreakerOpenError(
                self._provider_name,
                f"Circuit breaker is OPEN for provider '{self._provider_name}'",
            )

        start_time = time.monotonic()
        try:
            response = _call_generate_with_metadata(
                self._provider,
                system_prompt,
                user_prompt,
                settings=settings,
                deadline=deadline,
            )
            duration = time.monotonic() - start_time
            permit.record_success()
            collector = get_metrics_collector()
            collector.record_success(
                provider_name=self._provider_name,
                model_id=self.model_id,
                latency=duration,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            )
            return response
        except BaseException as exc:
            duration = time.monotonic() - start_time
            if isinstance(exc, Exception):
                permit.record_failure(exc)
                collector = get_metrics_collector()
                collector.record_failure(
                    provider_name=self._provider_name,
                    model_id=self.model_id,
                    latency=duration,
                    error=exc,
                )
            else:
                permit.release()
            raise

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        response = self.generate_with_metadata(system_prompt, user_prompt, settings=settings)
        return response.text


class RetryProvider(LLMProvider):
    """Wraps an LLMProvider with exponential backoff retry logic."""

    def __init__(
        self,
        provider: LLMProvider,
        policy: RetryPolicy,
        provider_name: str,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._provider_name = provider_name

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", f"unknown:{self._provider_name}")

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._policy.timeout_seconds
            deadline = GenerationDeadline.after(effective_timeout)
        return self._generate_with_metadata_with_deadline(
            system_prompt, user_prompt, settings=settings, deadline=deadline
        )

    def _generate_with_metadata_with_deadline(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # Allow deadline injection for testing; otherwise create fresh deadline
        if deadline is None:
            deadline = GenerationDeadline.after(self._policy.timeout_seconds)
        last_error: Exception | None = None

        for attempt in range(self._policy.max_attempts):
            # Check deadline before attempt (including initial attempt)
            if deadline.expired:
                logger.warning(
                    "Provider %s: Shared deadline exceeded before attempt %d/%d",
                    self._provider_name,
                    attempt + 1,
                    self._policy.max_attempts,
                )
                deadline.check_or_raise("retry")

            try:
                return _call_generate_with_metadata(
                    self._provider,
                    system_prompt,
                    user_prompt,
                    settings=settings,
                    deadline=deadline,
                )
            except Exception as exc:
                last_error = exc

                # If retries disabled or error is non-recoverable, fail immediately
                if not self._policy.enabled or not _is_recoverable_error(exc):
                    raise

                collector = get_metrics_collector()
                collector.record_retry(self._provider_name, self.model_id)

                # Do not sleep after final attempt
                if attempt == self._policy.max_attempts - 1:
                    break

                # Calculate backoff delay
                retry_after = _extract_retry_after(exc)
                delay = _calculate_backoff(
                    attempt,
                    self._policy.base_delay_seconds,
                    self._policy.exponential_base,
                    self._policy.max_delay_seconds,
                    self._policy.jitter_enabled,
                    retry_after=retry_after,
                )

                # Check shared deadline with planned sleep
                remaining = deadline.remaining_seconds()
                if remaining <= 0:
                    break
                actual_delay = min(delay, remaining)

                logger.warning(
                    "Provider %s: Attempt %d/%d failed, retrying in %.2fs "
                    "(deadline remaining: %.2fs): %s",
                    self._provider_name,
                    attempt + 1,
                    self._policy.max_attempts,
                    actual_delay,
                    remaining,
                    exc,
                )
                time.sleep(actual_delay)

        # All retry attempts exhausted or deadline exceeded
        if deadline.expired:
            deadline.check_or_raise("retry exhaustion")

        collector = get_metrics_collector()
        collector.record_retry_exhaustion(self._provider_name, self.model_id)
        logger.error(
            "Provider %s: All %d retry attempts exhausted, last error: %s",
            self._provider_name,
            self._policy.max_attempts,
            last_error,
        )
        msg = (
            f"Provider {self._provider_name}: All {self._policy.max_attempts} "
            f"retry attempts exhausted. Last error: {last_error}"
        )
        raise ProviderFallbackExhaustedError(
            msg,
            failed_providers=[f"{self._provider_name}: {type(last_error).__name__}"],
        ) from last_error

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        response = self.generate_with_metadata(system_prompt, user_prompt, settings=settings)
        return response.text


class FallbackProvider(LLMProvider):
    """Wraps primary and fallback LLMProviders to handle automatic failover."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_providers: list[LLMProvider],
        primary_name: str,
        fallback_names: list[str],
    ) -> None:
        self._primary_provider = primary_provider
        self._fallback_providers = fallback_providers
        self._primary_name = primary_name
        self._fallback_names = fallback_names

    @property
    def model_id(self) -> str:
        return getattr(self._primary_provider, "model_id", f"unknown:{self._primary_name}")

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._get_default_timeout()
            deadline = GenerationDeadline.after(effective_timeout)
        return self._generate_with_metadata_with_deadline(
            system_prompt, user_prompt, settings=settings, deadline=deadline
        )

    def _get_default_timeout(self) -> float:
        # Try to get timeout from the wrapped providers' retry policies
        for provider in [self._primary_provider] + self._fallback_providers:
            current = provider
            while hasattr(current, "_provider"):
                if hasattr(current, "_policy"):
                    return current._policy.timeout_seconds
                current = current._provider
        # Default to a reasonable timeout
        return 30.0

    def _generate_with_metadata_with_deadline(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # Allow deadline injection for testing; otherwise create fresh deadline
        if deadline is None:
            deadline = GenerationDeadline.after(self._get_default_timeout())
        failed_providers: list[str] = []
        last_error: Exception | None = None

        # Build list of (name, provider) preserving order and deduplicating
        providers_to_try: list[tuple[str, LLMProvider]] = [
            (self._primary_name, self._primary_provider)
        ]
        seen_names = {self._primary_name}
        for name, provider in zip(self._fallback_names, self._fallback_providers, strict=False):
            if name not in seen_names:
                seen_names.add(name)
                providers_to_try.append((name, provider))

        for idx, (name, provider) in enumerate(providers_to_try):
            # Check shared deadline before each provider attempt
            deadline.check_or_raise(f"fallback before provider {name}")

            try:
                if idx > 0:
                    logger.warning("Attempting fallback to provider %s (fallback %d)", name, idx)
                return _call_generate_with_metadata(
                    provider, system_prompt, user_prompt, settings=settings, deadline=deadline
                )
            except Exception as exc:
                last_error = exc
                failed_providers.append(f"{name}: {type(exc).__name__}")

                # Non-recoverable error stops fallback immediately
                if not _is_recoverable_error(exc):
                    logger.error(
                        "Provider %s failed with non-recoverable error %s: %s; aborting fallback",
                        name,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                logger.warning(
                    "Provider %s failed with recoverable error %s: %s; considering fallback",
                    name,
                    type(exc).__name__,
                    exc,
                )

                if idx < len(providers_to_try) - 1:
                    next_name = providers_to_try[idx + 1][0]
                    collector = get_metrics_collector()
                    collector.record_fallback(name, next_name, str(exc))

        # All providers failed
        logger.error("All %d providers failed: %s", len(providers_to_try), failed_providers)
        msg = (
            f"All providers failed ({len(providers_to_try)} attempted). "
            f"Attempted: {failed_providers}"
        )
        raise ProviderFallbackExhaustedError(
            msg,
            failed_providers=failed_providers,
        ) from last_error

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        response = self.generate_with_metadata(system_prompt, user_prompt, settings=settings)
        return response.text


def _calculate_backoff(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter_enabled: bool,
    retry_after: float | None = None,
) -> float:
    """Calculate backoff delay with exponential backoff and optional jitter."""
    if retry_after is not None:
        delay = min(retry_after, max_delay)
    else:
        delay = min(base_delay * (exponential_base**attempt), max_delay)

    if jitter_enabled:
        delay += random.uniform(0, delay * 0.1)

    return delay


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value from an exception if present."""
    response = getattr(exc, "response", None)
    if response is None:
        retry_after_attr = getattr(exc, "retry_after", None)
        if retry_after_attr is not None:
            try:
                return float(retry_after_attr)
            except (ValueError, TypeError):
                return None
        return None

    headers = getattr(response, "headers", None)
    if not headers or (not isinstance(headers, dict) and not hasattr(headers, "get")):
        return None

    retry_after_val = None
    if hasattr(headers, "get"):
        retry_after_val = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after_val is None:
        return None

    # Try numeric seconds
    try:
        val = float(retry_after_val)
        if val >= 0:
            return val
    except (ValueError, TypeError):
        pass

    # Try HTTP-date format
    try:
        from datetime import UTC, datetime
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(retry_after_val)
        now = datetime.now(UTC)
        diff = (dt - now).total_seconds()
        return max(0.0, diff)
    except Exception:
        pass

    return None


def _is_recoverable_error(exc: Exception) -> bool:
    """Determine if an exception represents a recoverable/transient error."""
    # Deadline exceeded is never recoverable - it's a hard limit for the logical request
    if isinstance(exc, ProviderDeadlineExceededError):
        return False

    if isinstance(exc, CircuitBreakerOpenError):
        return True

    if isinstance(exc, ProviderFallbackExhaustedError):
        if exc.__cause__ is not None:
            return _is_recoverable_error(exc.__cause__)
        return True

    type_name = type(exc).__name__
    if "Auth" in type_name or "Permission" in type_name:
        return False

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            if status in (400, 401, 403, 404):
                return False
            if status == 429 or status >= 500:
                return True

    exc_str = str(exc).lower()
    non_recoverable_keywords = [
        "authentication",
        "invalid api key",
        "api key",
        "unauthorized",
        "forbidden",
        "bad request",
        "model not found",
        "invalid request",
        "invalid argument",
    ]
    for kw in non_recoverable_keywords:
        if kw in exc_str:
            return False

    if (
        "Unavailable" in type_name
        or "RateLimit" in type_name
        or "Timeout" in type_name
        or "Connect" in type_name
    ):
        return True

    import httpx

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError, ConnectionError)):
        return True

    recoverable_keywords = [
        "rate limit",
        "too many requests",
        "service unavailable",
        "timed out",
        "timeout",
        "connection",
        "overloaded",
        "transient",
    ]
    for kw in recoverable_keywords:
        if kw in exc_str:
            return True

    return False


def _parse_fallback_providers(value: str) -> list[str]:
    """Parse comma-separated list of fallback provider names."""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _get_circuit_breaker_policy(settings: Settings) -> CircuitBreakerPolicy:
    """Extract circuit breaker policy from settings."""
    return CircuitBreakerPolicy(
        enabled=settings.circuit_breaker_enabled,
        failure_threshold=settings.circuit_breaker_failure_threshold,
        recovery_timeout_seconds=settings.circuit_breaker_recovery_timeout_seconds,
        half_open_max_calls=settings.circuit_breaker_half_open_max_calls,
        window_seconds=settings.circuit_breaker_window_seconds,
    )


def _get_retry_policy(settings: Settings) -> RetryPolicy:
    """Extract retry policy from settings."""
    return RetryPolicy(
        enabled=settings.retry_enabled,
        max_attempts=settings.retry_max_attempts,
        base_delay_seconds=settings.retry_base_delay_seconds,
        max_delay_seconds=settings.retry_max_delay_seconds,
        exponential_base=settings.retry_exponential_base,
        jitter_enabled=settings.retry_jitter_enabled,
        timeout_seconds=settings.retry_timeout_seconds,
    )


def get_circuit_breaker(provider_name: str, settings: Settings | None = None) -> CircuitBreaker:
    """Get or create circuit breaker for the given provider."""
    if provider_name not in _circuit_breakers:
        from core.config import get_settings

        st = settings or get_settings()
        policy = _get_circuit_breaker_policy(st)
        _circuit_breakers[provider_name] = CircuitBreaker(provider_name, policy)
    return _circuit_breakers[provider_name]


def _create_raw_provider(provider_name: str, settings: Settings) -> LLMProvider:
    """Instantiate a base chat provider."""
    if provider_name == "ollama":
        from core.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(
            model=settings.chat_model,
            host=settings.chat_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "gemini":
        from core.llm.gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(
            model=settings.gemini_chat_model,
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            timeout=settings.rq_job_timeout,
        )
    elif provider_name == "groq":
        from core.llm.groq_provider import GroqLLMProvider

        return GroqLLMProvider(
            model=settings.groq_chat_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.rq_job_timeout,
        )
    else:
        raise ProviderNotSupportedError(
            f"Unsupported chat provider: {provider_name}. Supported providers: ollama, gemini, groq"
        )


def _wrap_provider_pipeline(
    provider_name: str, raw_provider: LLMProvider, settings: Settings
) -> LLMProvider:
    """Wrap raw provider with retry and circuit breaker layers."""
    provider = raw_provider

    # 1. Retry layer (inner)
    retry_policy = _get_retry_policy(settings)
    provider = RetryProvider(provider, retry_policy, provider_name)

    # 2. Circuit breaker layer (outer)
    cb = get_circuit_breaker(provider_name, settings)
    provider = CircuitBreakerProvider(provider, cb, provider_name)

    return provider


def get_chat_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Get a chat provider instance by name, wrapped with resilience features.

    Args:
        provider_name: Provider name ("ollama", "gemini", "groq").
                       If None, uses CHAT_PROVIDER from settings.

    Returns:
        LLMProvider instance
    """
    from core.config import get_settings

    settings = get_settings()
    provider_name = provider_name or settings.chat_provider

    cache_key = f"chat:{provider_name}:{settings.fallback_enabled}:{settings.fallback_providers}"
    if cache_key in _chat_provider_cache:
        return _chat_provider_cache[cache_key]

    raw_primary = _create_raw_provider(provider_name, settings)
    wrapped_primary = _wrap_provider_pipeline(provider_name, raw_primary, settings)

    if settings.fallback_enabled:
        fallback_names = [
            f for f in _parse_fallback_providers(settings.fallback_providers) if f != provider_name
        ]
        wrapped_fallbacks = [
            _wrap_provider_pipeline(f_name, _create_raw_provider(f_name, settings), settings)
            for f_name in fallback_names
        ]
        provider = FallbackProvider(
            primary_provider=wrapped_primary,
            fallback_providers=wrapped_fallbacks,
            primary_name=provider_name,
            fallback_names=fallback_names,
        )
    else:
        provider = wrapped_primary

    _chat_provider_cache[cache_key] = provider
    # Also support direct lookup by provider_name for caching tests
    _chat_provider_cache[provider_name] = provider
    return provider


def get_embedding_provider(provider_name: str | None = None):
    """
    Get an embedding provider instance by name.

    Args:
        provider_name: Provider name ("ollama"). If None, uses EMBEDDING_PROVIDER from settings.

    Returns:
        LLMProvider instance (wraps embedding functionality)
    """
    from core.config import get_settings

    settings = get_settings()
    provider_name = provider_name or settings.embedding_provider

    cache_key = f"embedding:{provider_name}"
    if cache_key in _embedding_provider_cache:
        return _embedding_provider_cache[cache_key]

    if provider_name == "ollama":
        from core.embed.ollama_embedder import OllamaEmbedder

        provider = OllamaEmbedder(
            model=settings.embedding_model,
            host=settings.embedding_base_url,
            timeout=settings.rq_job_timeout,
        )
    else:
        raise ProviderNotSupportedError(
            f"Unsupported embedding provider: {provider_name}. Supported providers: ollama"
        )

    _embedding_provider_cache[provider_name] = provider
    return provider


def clear_provider_cache() -> None:
    """Clear provider caches, circuit breakers, and metrics (for testing)."""
    _chat_provider_cache.clear()
    _embedding_provider_cache.clear()
    _circuit_breakers.clear()
    get_metrics_collector().reset()
