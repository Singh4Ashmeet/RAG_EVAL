"""Asynchronous resilience wrappers for LLM providers."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from core.errors import (
    ProviderDeadlineExceededError,
)
from core.llm.base import AsyncLLMProvider, GenerationResponse, GenerationSettings
from core.llm.deadline import GenerationDeadline
from core.llm.metrics import get_metrics_collector
from core.llm.router import (
    CircuitBreakerOpenError,
    CircuitBreakerState,
    ProviderFallbackExhaustedError,
)

logger = logging.getLogger(__name__)


# Re-export the existing policy classes for async use
@dataclass
class AsyncRetryPolicy:
    """Configuration for async retry behavior."""

    enabled: bool = True
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    timeout_seconds: float = 30.0


@dataclass
class AsyncCircuitBreakerPolicy:
    """Configuration for async circuit breaker behavior."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    window_seconds: float = 60.0


# Reuse the existing sync policies for configuration
RetryPolicy = AsyncRetryPolicy
CircuitBreakerPolicy = AsyncCircuitBreakerPolicy


class AsyncRetryProvider(AsyncLLMProvider):
    """Wraps an AsyncLLMProvider with exponential backoff retry logic."""

    def __init__(
        self,
        provider: AsyncLLMProvider,
        policy: RetryPolicy,
        provider_name: str,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._provider_name = provider_name

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", f"unknown:{self._provider_name}")

    async def generate_with_metadata_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._policy.timeout_seconds
            deadline = GenerationDeadline.after(effective_timeout)
        return await self._generate_with_metadata_with_deadline(
            system_prompt, user_prompt, settings=settings, deadline=deadline
        )

    async def _generate_with_metadata_with_deadline(
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
                return await self._provider.generate_with_metadata_async(
                    system_prompt, user_prompt, settings=settings, deadline=deadline
                )
            except asyncio.CancelledError:
                raise
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
                await asyncio.sleep(actual_delay)

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

    async def aclose(self) -> None:
        """Release async resources. Idempotent."""
        await self._provider.aclose()

    # Convenience method for text-only generation (optional, not part of protocol)
    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> str:
        response = await self.generate_with_metadata_async(
            system_prompt, user_prompt, settings=settings, timeout=timeout, deadline=deadline
        )
        return response.text


class AsyncCircuitBreakerPermit:
    """Represents an execution permit granted by the async circuit breaker for one request."""

    def __init__(
        self,
        breaker: AsyncCircuitBreaker,
        state: str,
        epoch: int,
        is_probe: bool,
    ) -> None:
        self.breaker = breaker
        self.state = state
        self.epoch = epoch
        self.is_probe = is_probe
        self._resolved = False
        self._lock = asyncio.Lock()

    @property
    def is_resolved(self) -> bool:
        # Use a simple check without lock for the property
        return self._resolved

    async def record_success(self) -> None:
        """Resolve this permit as a successful request."""
        async with self._lock:
            if self._resolved:
                return
            self._resolved = True
        await self.breaker._resolve_success(self)

    async def record_failure(self, exc: Exception) -> None:
        """Resolve this permit as a failed request."""
        async with self._lock:
            if self._resolved:
                return
            self._resolved = True
        await self.breaker._resolve_failure(self, exc)

    async def release(self) -> None:
        """Release this permit if execution was aborted, cancelled, or interrupted."""
        async with self._lock:
            if self._resolved:
                return
            self._resolved = True
        await self.breaker._resolve_release(self)


class AsyncCircuitBreaker:
    """
    Async-compatible circuit breaker for provider failure isolation.

    Implements a three-state circuit breaker:
    - CLOSED: Normal operation, requests allowed, failures tracked
    - OPEN: Requests rejected immediately, recovery timer started
    - HALF_OPEN: Limited probe requests allowed to test recovery

    Thread-safe using asyncio.Lock. Never holds lock across external async calls.
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
        self._lock = asyncio.Lock()
        self._state = CircuitBreakerState.CLOSED
        self._epoch = 0
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._last_state_change = self._clock()

    async def get_state(self) -> str:
        """Get the current circuit breaker state."""
        async with self._lock:
            await self._check_state_transition_locked()
            return self._state

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _is_failure_recoverable(self, exc: Exception) -> bool:
        """Determine if an exception counts as a circuit breaker failure."""
        if isinstance(exc, ProviderFallbackExhaustedError) and isinstance(exc.__cause__, Exception):
            return _is_recoverable_error(exc.__cause__)
        return _is_recoverable_error(exc)

    async def _check_state_transition_locked(self) -> None:
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

    async def acquire_permit(self) -> AsyncCircuitBreakerPermit | None:
        """
        Atomically check state and acquire an execution permit.
        Returns an AsyncCircuitBreakerPermit if allowed, or None if rejected.
        """
        if not self._policy.enabled:
            return AsyncCircuitBreakerPermit(
                breaker=self,
                state=CircuitBreakerState.CLOSED,
                epoch=self._epoch,
                is_probe=False,
            )

        async with self._lock:
            await self._check_state_transition_locked()

            if self._state == CircuitBreakerState.CLOSED:
                return AsyncCircuitBreakerPermit(
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
                    return AsyncCircuitBreakerPermit(
                        breaker=self,
                        state=CircuitBreakerState.HALF_OPEN,
                        epoch=self._epoch,
                        is_probe=True,
                    )
                return None

            return None

    async def _resolve_success(self, permit: AsyncCircuitBreakerPermit) -> None:
        """Resolve a successful probe or closed request from its permit."""
        if not self._policy.enabled:
            return

        async with self._lock:
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

    async def _resolve_failure(self, permit: AsyncCircuitBreakerPermit, exc: Exception) -> None:
        """Resolve a failed probe or closed request from its permit."""
        if not self._policy.enabled:
            return

        is_recoverable = self._is_failure_recoverable(exc)

        async with self._lock:
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

    async def _resolve_release(self, permit: AsyncCircuitBreakerPermit) -> None:
        """Release an unresolved permit due to abort, cancellation, or exception."""
        if not self._policy.enabled:
            return

        async with self._lock:
            if permit.epoch != self._epoch:
                return

            if self._state == CircuitBreakerState.HALF_OPEN and permit.is_probe:
                if self._half_open_calls > 0:
                    self._half_open_calls -= 1
                logger.info(
                    "Probe permit released for provider %s due to cancellation or exception",
                    self._provider_name,
                )

    async def record_success(self, permit: AsyncCircuitBreakerPermit | None = None) -> None:
        """Record a successful provider request."""
        if permit is not None:
            await permit.record_success()
            return

        if not self._policy.enabled:
            return

        async with self._lock:
            await self._check_state_transition_locked()
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

    async def record_failure(
        self, exc: Exception, permit: AsyncCircuitBreakerPermit | None = None
    ) -> None:
        """Record a failed provider request if recoverable."""
        if permit is not None:
            await permit.record_failure(exc)
            return

        if not self._policy.enabled:
            return

        is_recoverable = self._is_failure_recoverable(exc)

        async with self._lock:
            await self._check_state_transition_locked()
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

        async def _reset() -> None:
            async with self._lock:
                self._state = CircuitBreakerState.CLOSED
                self._epoch += 1
                self._failure_count = 0
                self._success_count = 0
                self._half_open_calls = 0
                self._last_failure_time = None
                self._last_state_change = self._clock()

        asyncio.run(_reset())

    async def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        async with self._lock:
            await self._check_state_transition_locked()
            return {
                "provider": self._provider_name,
                "state": self._state,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
            }


class AsyncCircuitBreakerProvider(AsyncLLMProvider):
    """Wraps an AsyncLLMProvider with circuit breaker protection."""

    def __init__(
        self,
        provider: AsyncLLMProvider,
        circuit_breaker: AsyncCircuitBreaker,
        provider_name: str,
    ) -> None:
        self._provider = provider
        self._circuit_breaker = circuit_breaker
        self._provider_name = provider_name

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", f"unknown:{self._provider_name}")

    async def generate_with_metadata_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._get_default_timeout()
            deadline = GenerationDeadline.after(effective_timeout)
        return await self._generate_with_metadata_with_deadline(
            system_prompt, user_prompt, settings=settings, deadline=deadline
        )

    def _get_default_timeout(self) -> float:
        # Try to get timeout from the wrapped provider's retry policy if available
        if hasattr(self._provider, "_policy"):
            return self._provider._policy.timeout_seconds
        # Default to a reasonable timeout
        return 30.0

    async def _generate_with_metadata_with_deadline(
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

        permit = await self._circuit_breaker.acquire_permit()
        if permit is None:
            collector = get_metrics_collector()
            collector.record_circuit_rejection(self._provider_name, self.model_id)
            raise CircuitBreakerOpenError(
                self._provider_name,
                f"Circuit breaker is OPEN for provider '{self._provider_name}'",
            )

        start_time = time.monotonic()
        try:
            response = await self._provider.generate_with_metadata_async(
                system_prompt, user_prompt, settings=settings, deadline=deadline
            )
            duration = time.monotonic() - start_time
            await permit.record_success()
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
        except asyncio.CancelledError:
            duration = time.monotonic() - start_time
            await permit.release()
            raise
        except Exception as exc:
            duration = time.monotonic() - start_time
            await permit.record_failure(exc)
            raise
        except BaseException:
            duration = time.monotonic() - start_time
            await permit.release()
            raise

    async def aclose(self) -> None:
        """Release async resources. Idempotent."""
        await self._provider.aclose()

    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> str:
        response = await self.generate_with_metadata_async(
            system_prompt, user_prompt, settings=settings, timeout=timeout, deadline=deadline
        )
        return response.text


class AsyncFallbackProvider(AsyncLLMProvider):
    """Wraps primary and fallback AsyncLLMProviders to handle automatic failover."""

    def __init__(
        self,
        primary_provider: AsyncLLMProvider,
        fallback_providers: list[AsyncLLMProvider],
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

    async def generate_with_metadata_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> GenerationResponse:
        # If a deadline is already provided (from outer wrapper), use it.
        # Otherwise create a fresh deadline for this logical request.
        if deadline is None:
            effective_timeout = timeout if timeout is not None else self._get_default_timeout()
            deadline = GenerationDeadline.after(effective_timeout)
        return await self._generate_with_metadata_with_deadline(
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

    async def _generate_with_metadata_with_deadline(
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
        providers_to_try: list[tuple[str, AsyncLLMProvider]] = [
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
                return await provider.generate_with_metadata_async(
                    system_prompt, user_prompt, settings=settings, deadline=deadline
                )
            except asyncio.CancelledError:
                raise
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

    async def aclose(self) -> None:
        """Release async resources. Idempotent."""
        await self._primary_provider.aclose()
        for provider in self._fallback_providers:
            await provider.aclose()

    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
        timeout: float | None = None,  # noqa: ASYNC109
        deadline: GenerationDeadline | None = None,
    ) -> str:
        response = await self.generate_with_metadata_async(
            system_prompt, user_prompt, settings=settings, timeout=timeout, deadline=deadline
        )
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
        if isinstance(exc.__cause__, Exception):
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
