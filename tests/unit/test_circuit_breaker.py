"""Unit tests for provider-specific circuit breakers and retry/fallback integration."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
import pytest

from core.errors import LLMError
from core.llm.base import GenerationSettings
from core.llm.gemini_provider import (
    ProviderAuthenticationError as GeminiAuthError,
)
from core.llm.gemini_provider import (
    ProviderUnavailableError as GeminiUnavailableError,
)
from core.llm.router import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    CircuitBreakerState,
    FallbackProvider,
    ProviderFallbackExhaustedError,
    RetryPolicy,
    RetryProvider,
    clear_provider_cache,
    get_chat_provider,
)


class MockClock:
    """Deterministic simulated clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current_time = start

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        name: str,
        response: str = "OK",
        error: Exception | None = None,
    ) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._response = response
        self._error = error
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._response


class FlakyProvider:
    """Mock provider that fails a specified number of times before succeeding."""

    def __init__(
        self,
        name: str,
        fail_times: int,
        error: Exception,
        response: str = "Recovered",
    ) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._fail_times = fail_times
        self._error = error
        self._response = response
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise self._error
        return self._response


def test_circuit_breaker_initial_state_closed() -> None:
    """Test circuit breaker starts in CLOSED state."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.allow_request() is True
    assert cb.can_execute() is True
    stats = cb.get_stats()
    assert stats["failure_count"] == 0
    assert stats["state"] == CircuitBreakerState.CLOSED


def test_circuit_breaker_transitions_to_open_on_threshold() -> None:
    """Test transitions to OPEN after consecutive recoverable failures reach threshold."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    error = GeminiUnavailableError("Service down")

    cb.record_failure(error)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.get_stats()["failure_count"] == 1

    cb.record_failure(error)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.get_stats()["failure_count"] == 2

    # Third failure reaches threshold
    cb.record_failure(error)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.allow_request() is False
    assert cb.can_execute() is False


def test_circuit_breaker_fast_failure_when_open() -> None:
    """Test requests are rejected immediately when circuit is OPEN."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("groq", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Error 1"))
    cb.record_failure(GeminiUnavailableError("Error 2"))
    assert cb.state == CircuitBreakerState.OPEN

    # CircuitBreakerProvider should raise CircuitBreakerOpenError immediately
    raw_provider = MockProvider("groq", "Response")
    cb_provider = CircuitBreakerProvider(raw_provider, cb, "groq")

    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        cb_provider.generate("sys", "user", settings=GenerationSettings())

    assert exc_info.value.provider_name == "groq"
    assert "Circuit breaker is OPEN for provider 'groq'" in str(exc_info.value)
    # The underlying provider was never called
    assert raw_provider.call_count == 0


def test_circuit_breaker_recovery_timeout_transition_to_half_open() -> None:
    """Test OPEN circuit transitions to HALF_OPEN after recovery timeout."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("ollama", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down 1"))
    cb.record_failure(GeminiUnavailableError("Down 2"))
    assert cb.state == CircuitBreakerState.OPEN

    # Advance time partially (20s < 30s)
    clock.advance(20.0)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_execute() is False

    # Advance time past 30s
    clock.advance(11.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.can_execute() is True


def test_circuit_breaker_half_open_allows_limited_probes() -> None:
    """Test HALF_OPEN state strictly limits probe calls to half_open_max_calls."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=10.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(15.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # First call is the probe -> allowed
    assert cb.allow_request() is True

    # Subsequent call while probe is in flight -> rejected
    assert cb.allow_request() is False


def test_circuit_breaker_half_open_success_transitions_to_closed() -> None:
    """Test successful probe call closes the circuit and resets failures."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=2,
        recovery_timeout_seconds=15.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down 1"))
    cb.record_failure(GeminiUnavailableError("Down 2"))
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(20.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.allow_request() is True

    # Probe succeeds
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.get_stats()["failure_count"] == 0
    assert cb.allow_request() is True


def test_circuit_breaker_half_open_failure_transitions_to_open() -> None:
    """Test failure during probe immediately re-opens circuit and resets recovery timer."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=2,
        recovery_timeout_seconds=20.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down 1"))
    cb.record_failure(GeminiUnavailableError("Down 2"))
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(25.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.allow_request() is True

    # Probe fails
    cb.record_failure(GeminiUnavailableError("Probe failed"))
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.allow_request() is False

    # Must wait another full recovery_timeout_seconds
    clock.advance(15.0)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.allow_request() is False

    clock.advance(10.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN


def test_circuit_breaker_success_resets_failure_count() -> None:
    """Test successful call in CLOSED state resets consecutive failure count."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Flaky 1"))
    cb.record_failure(GeminiUnavailableError("Flaky 2"))
    assert cb.get_stats()["failure_count"] == 2

    cb.record_success()
    assert cb.get_stats()["failure_count"] == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_circuit_breaker_non_recoverable_errors_ignored() -> None:
    """Test non-recoverable errors (400, 401, 403, 404, auth) do not trigger circuit breaker."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # 401 Authentication error
    cb.record_failure(GeminiAuthError("Invalid API key"))
    assert cb.get_stats()["failure_count"] == 0

    # 400 Bad Request
    resp_400 = SimpleNamespace(status_code=400, text="Bad Request")
    cb.record_failure(httpx.HTTPStatusError("400", request=None, response=resp_400))
    assert cb.get_stats()["failure_count"] == 0

    # 404 Model Not Found
    resp_404 = SimpleNamespace(status_code=404, text="Not Found")
    cb.record_failure(httpx.HTTPStatusError("404", request=None, response=resp_404))
    assert cb.get_stats()["failure_count"] == 0

    # Generic LLMError with auth message
    cb.record_failure(LLMError("Authentication failed for service"))
    assert cb.get_stats()["failure_count"] == 0

    assert cb.state == CircuitBreakerState.CLOSED


def test_circuit_breaker_recoverable_errors_counted() -> None:
    """Test recoverable errors (429, 5xx, timeouts, connection errors) are counted."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=5, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # 429
    resp_429 = SimpleNamespace(status_code=429, text="Rate limit")
    cb.record_failure(httpx.HTTPStatusError("429", request=None, response=resp_429))
    assert cb.get_stats()["failure_count"] == 1

    # 503
    resp_503 = SimpleNamespace(status_code=503, text="Service Unavailable")
    cb.record_failure(httpx.HTTPStatusError("503", request=None, response=resp_503))
    assert cb.get_stats()["failure_count"] == 2

    # Timeout
    cb.record_failure(httpx.TimeoutException("Read timeout"))
    assert cb.get_stats()["failure_count"] == 3

    # ConnectError
    cb.record_failure(httpx.ConnectError("Connection refused"))
    assert cb.get_stats()["failure_count"] == 4

    # ProviderUnavailableError
    cb.record_failure(GeminiUnavailableError("Unavailable"))
    assert cb.get_stats()["failure_count"] == 5
    assert cb.state == CircuitBreakerState.OPEN


def test_circuit_breaker_disabled_passes_all() -> None:
    """Test disabled circuit breaker always allows requests and ignores failures."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(enabled=False, failure_threshold=1)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    for _ in range(10):
        cb.record_failure(GeminiUnavailableError("Fail"))

    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.allow_request() is True
    assert cb.can_execute() is True


def test_circuit_breaker_reset() -> None:
    """Test reset restores circuit breaker to initial CLOSED state."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    assert cb.state == CircuitBreakerState.OPEN

    cb.reset()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.get_stats()["failure_count"] == 0
    assert cb.allow_request() is True


def test_circuit_breaker_get_stats() -> None:
    """Test get_stats returns complete and accurate metrics."""
    clock = MockClock(start=500.0)
    policy = CircuitBreakerPolicy(failure_threshold=5, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Fail 1"))
    stats = cb.get_stats()

    assert stats["provider"] == "gemini"
    assert stats["state"] == "closed"
    assert stats["failure_count"] == 1
    assert stats["success_count"] == 0
    assert stats["last_failure_time"] == 500.0


def test_circuit_breaker_per_provider_isolation() -> None:
    """Test that failure in one provider does not affect another provider's circuit breaker."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30.0)
    cb_gemini = CircuitBreaker("gemini", policy, clock=clock)
    cb_groq = CircuitBreaker("groq", policy, clock=clock)

    cb_gemini.record_failure(GeminiUnavailableError("Gemini down"))
    assert cb_gemini.state == CircuitBreakerState.OPEN
    assert cb_gemini.allow_request() is False

    assert cb_groq.state == CircuitBreakerState.CLOSED
    assert cb_groq.allow_request() is True


def test_circuit_breaker_thread_safety() -> None:
    """Test thread-safety under concurrent failure and success records."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=100, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("test", policy, clock=clock)

    def worker():
        for _ in range(50):
            cb.allow_request()
            cb.record_success()
            cb.record_failure(GeminiUnavailableError("Transient"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Breaker state remains valid and doesn't throw concurrency errors
    assert cb.state in (CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)


# ==============================================================================
# Integration Tests: Circuit Breaker with Retry and Fallback
# ==============================================================================


def test_single_logical_request_failure_counting() -> None:
    """Test that internal retries do not count as multiple circuit breaker failures."""
    clock = MockClock()
    cb_policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("primary", cb_policy, clock=clock)

    retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001, jitter_enabled=False)
    raw_provider = FlakyProvider("primary", fail_times=3, error=GeminiUnavailableError("Down"))
    retry_wrapped = RetryProvider(raw_provider, retry_policy, "primary")
    cb_wrapped = CircuitBreakerProvider(retry_wrapped, cb, "primary")

    with pytest.raises(ProviderFallbackExhaustedError):
        cb_wrapped.generate("sys", "user", settings=GenerationSettings())

    # Raw provider was called 3 times internally
    assert raw_provider.call_count == 3
    # BUT circuit breaker only recorded 1 logical failure!
    assert cb.get_stats()["failure_count"] == 1
    assert cb.state == CircuitBreakerState.CLOSED


def test_successful_retry_records_success() -> None:
    """Test that a request succeeding on retry counts as success on circuit breaker."""
    clock = MockClock()
    cb_policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("primary", cb_policy, clock=clock)

    # First add 1 failure to failure_count
    cb.record_failure(GeminiUnavailableError("Previous error"))
    assert cb.get_stats()["failure_count"] == 1

    # Now make a request that fails once then succeeds on attempt 2
    retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001, jitter_enabled=False)
    raw_provider = FlakyProvider(
        "primary", fail_times=1, error=GeminiUnavailableError("Flake"), response="OK"
    )
    retry_wrapped = RetryProvider(raw_provider, retry_policy, "primary")
    cb_wrapped = CircuitBreakerProvider(retry_wrapped, cb, "primary")

    result = cb_wrapped.generate("sys", "user", settings=GenerationSettings())
    assert result == "OK"
    assert raw_provider.call_count == 2
    # Successful logical outcome reset circuit breaker failure count to 0!
    assert cb.get_stats()["failure_count"] == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_circuit_open_triggers_immediate_fallback() -> None:
    """Test that an open circuit immediately invokes fallback without attempting primary retries."""
    clock = MockClock()
    cb_policy = CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_seconds=30.0)
    cb_primary = CircuitBreaker("primary", cb_policy, clock=clock)
    cb_fallback = CircuitBreaker("fallback", cb_policy, clock=clock)

    # Force primary open
    cb_primary.record_failure(GeminiUnavailableError("Primary down"))
    assert cb_primary.state == CircuitBreakerState.OPEN

    primary_raw = MockProvider("primary", "Primary response")
    fallback_raw = MockProvider("fallback", "Fallback response")

    primary_cb = CircuitBreakerProvider(primary_raw, cb_primary, "primary")
    fallback_cb = CircuitBreakerProvider(fallback_raw, cb_fallback, "fallback")

    chain = FallbackProvider(
        primary_provider=primary_cb,
        fallback_providers=[fallback_cb],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    result = chain.generate("sys", "user", settings=GenerationSettings())
    assert result == "Fallback response"
    # Primary was immediately bypassed
    assert primary_raw.call_count == 0
    assert fallback_raw.call_count == 1


def test_circuit_breaker_wraps_in_get_chat_provider() -> None:
    """Test that get_chat_provider integrates circuit breaker."""
    import os

    os.environ["CHAT_PROVIDER"] = "ollama"
    os.environ["CIRCUIT_BREAKER_ENABLED"] = "true"
    os.environ["CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "4"

    from core.config import clear_settings_cache

    clear_provider_cache()
    clear_settings_cache()

    provider = get_chat_provider("ollama")
    assert provider is not None
    assert isinstance(provider, CircuitBreakerProvider)
    assert provider._circuit_breaker._policy.failure_threshold == 4


# ==============================================================================
# M4.5.4 Probe Lifecycle Regression Tests (F-01 Fix Verification)
# ==============================================================================


def test_probe_non_recoverable_error_does_not_wedge_circuit() -> None:
    """Test 1: Non-recoverable error in HALF_OPEN releases permit and doesn't wedge breaker."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # 1. Force OPEN
    cb.record_failure(GeminiUnavailableError("Down"))
    assert cb.state == CircuitBreakerState.OPEN

    # 2. Advance clock past recovery timeout -> HALF_OPEN
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 3. Probe with a non-recoverable 401 error
    flaky = FlakyProvider("gemini", fail_times=1, error=GeminiAuthError("Invalid API key"))
    cb_provider = CircuitBreakerProvider(flaky, cb, "gemini")

    # Error must propagate
    with pytest.raises(GeminiAuthError):
        cb_provider.generate("sys", "user", settings=GenerationSettings())

    # 4. Breaker is NOT permanently wedged in HALF_OPEN with 0 permits
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.can_execute() is True

    # 5. Subsequent probe can be admitted and succeed
    success_provider = MockProvider("gemini", response="Success")
    cb_provider_recovered = CircuitBreakerProvider(success_provider, cb, "gemini")
    result = cb_provider_recovered.generate("sys", "user", settings=GenerationSettings())
    assert result == "Success"
    assert cb.state == CircuitBreakerState.CLOSED


def test_probe_http_400_error_does_not_increment_health_failures() -> None:
    """Test 2: HTTP 400 error in HALF_OPEN releases permit and doesn't count as health failure."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=2,
        recovery_timeout_seconds=20.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Force OPEN
    cb.record_failure(GeminiUnavailableError("Down 1"))
    cb.record_failure(GeminiUnavailableError("Down 2"))
    assert cb.state == CircuitBreakerState.OPEN

    # Advance to HALF_OPEN
    clock.advance(25.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    resp_400 = SimpleNamespace(status_code=400, text="Bad Request")
    http_400_err = httpx.HTTPStatusError("400", request=None, response=resp_400)

    provider = MockProvider("gemini", error=http_400_err)
    cb_provider = CircuitBreakerProvider(provider, cb, "gemini")

    with pytest.raises(httpx.HTTPStatusError):
        cb_provider.generate("sys", "user", settings=GenerationSettings())

    # Permit was released, state remains HALF_OPEN, subsequent probe is allowed
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.can_execute() is True
    assert cb.allow_request() is True


def test_probe_unexpected_exception_cleans_up_permit() -> None:
    """Test 3: Unexpected generic exception during probe cleans up permit."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=15.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("groq", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(20.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    provider = MockProvider("groq", error=RuntimeError("Unexpected crash"))
    cb_provider = CircuitBreakerProvider(provider, cb, "groq")

    with pytest.raises(RuntimeError) as exc_info:
        cb_provider.generate("sys", "user", settings=GenerationSettings())
    assert "Unexpected crash" in str(exc_info.value)

    # Permit released; subsequent request permitted
    assert cb.can_execute() is True


def test_probe_recoverable_failure_reopens_immediately_and_resets_recovery_timer() -> None:
    """Test 4: Recoverable probe failure immediately reopens and resets timer."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Initial open at t=1000
    cb.record_failure(GeminiUnavailableError("Down"))
    assert cb.state == CircuitBreakerState.OPEN

    # Advance to t=1035 -> HALF_OPEN
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Probe fails with recoverable error at t=1035
    cb.record_failure(GeminiUnavailableError("Probe failed"))
    assert cb.state == CircuitBreakerState.OPEN

    # At t=1050 (only 15s elapsed since reopen < 30s timeout), must still be OPEN
    clock.advance(15.0)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.allow_request() is False

    # At t=1070 (35s elapsed since reopen >= 30s timeout), transitions to HALF_OPEN
    clock.advance(20.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.allow_request() is True


def test_probe_success_resolves_permit_and_closes_circuit() -> None:
    """Test 5: Successful probe resolves permit and closes the circuit."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("ollama", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    raw_provider = MockProvider("ollama", response="OK")
    cb_provider = CircuitBreakerProvider(raw_provider, cb, "ollama")

    result = cb_provider.generate("sys", "user", settings=GenerationSettings())
    assert result == "OK"
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.get_stats()["failure_count"] == 0


def test_probe_capacity_enforcement_and_permit_recycling() -> None:
    """Test 6: With half_open_max_calls=2, probe capacity is strictly enforced and recycled."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 1. Acquire permit 1 and 2
    permit1 = cb.acquire_permit()
    assert permit1 is not None
    assert permit1.is_probe is True

    permit2 = cb.acquire_permit()
    assert permit2 is not None
    assert permit2.is_probe is True

    # 2. Third request rejected (capacity full)
    permit3 = cb.acquire_permit()
    assert permit3 is None

    # 3. Resolve permit 1 with a non-recoverable error (releasing permit 1)
    permit1.record_failure(GeminiAuthError("401 Auth error"))

    # 4. Now a new probe permit can be admitted!
    permit4 = cb.acquire_permit()
    assert permit4 is not None
    assert permit4.is_probe is True


def test_stale_concurrent_probe_cannot_close_reopened_circuit() -> None:
    """Test 7: A stale probe success cannot close a newly reopened circuit."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Admit two concurrent probes
    permit1 = cb.acquire_permit()
    permit2 = cb.acquire_permit()
    assert permit1 is not None and permit2 is not None

    # Probe 1 fails with recoverable error -> circuit immediately reopens
    permit1.record_failure(GeminiUnavailableError("Probe 1 failed"))
    assert cb.state == CircuitBreakerState.OPEN

    # Later, Probe 2 finishes successfully from before the failure
    permit2.record_success()

    # The stale success MUST NOT close the circuit
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_execute() is False


def test_permit_double_resolution_protection() -> None:
    """Test 8: Resolving a permit twice cannot corrupt counters or state."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(35.0)

    permit = cb.acquire_permit()
    assert permit is not None

    # First resolution
    permit.record_success()
    stats1 = cb.get_stats()

    # Second resolution (duplicate)
    permit.record_success()
    permit.release()
    stats2 = cb.get_stats()

    # Success count must not have been double incremented
    assert stats1["success_count"] == stats2["success_count"]


def test_cancellation_interruption_cleanup_does_not_leak_or_misclassify() -> None:
    """Test 9: KeyboardInterrupt/cancellation cleans up probe permit without misclassifying."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    cb.record_failure(GeminiUnavailableError("Down"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    class InterruptedProvider:
        model_id = "mock:interrupted"

        def generate(self, *args, **kwargs):
            raise KeyboardInterrupt("Simulated user cancellation")

    cb_provider = CircuitBreakerProvider(InterruptedProvider(), cb, "gemini")

    # Cancellation must propagate cleanly without being swallowed
    with pytest.raises(KeyboardInterrupt):
        cb_provider.generate("sys", "user", settings=GenerationSettings())

    # Permit was released in except BaseException; circuit is not wedged
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.can_execute() is True


def test_closed_and_open_regressions_preserved() -> None:
    """Test 10: Standard CLOSED threshold and OPEN fast failure remain fully functional."""
    clock = MockClock()
    policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=30.0)
    cb = CircuitBreaker("gemini", policy, clock=clock)

    assert cb.state == CircuitBreakerState.CLOSED

    # Failure 1
    cb.record_failure(GeminiUnavailableError("Fail 1"))
    assert cb.state == CircuitBreakerState.CLOSED

    # Success in CLOSED resets counter
    cb.record_success()
    assert cb.get_stats()["failure_count"] == 0

    # Two consecutive failures reach threshold -> OPEN
    cb.record_failure(GeminiUnavailableError("Fail 2"))
    cb.record_failure(GeminiUnavailableError("Fail 3"))
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.allow_request() is False
    assert cb.can_execute() is False


# ---------------------------------------------------------------------------
# M4.5.4 hardening: F-03 stale failure timestamp tests
# ---------------------------------------------------------------------------


def test_stale_failure_does_not_change_last_failure_time() -> None:
    """F-03 Test 1: A stale permit failure must not update last_failure_time."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Drive to HALF_OPEN: one recoverable failure, then wait for recovery
    cb.record_failure(GeminiUnavailableError("Fail 1"))
    failure_time_after_close = cb.get_stats()["last_failure_time"]
    assert cb.state == CircuitBreakerState.OPEN

    clock.advance(35.0)  # Recovery timeout
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Acquire two probes
    permit1 = cb.acquire_permit()
    permit2 = cb.acquire_permit()
    assert permit1 is not None and permit2 is not None

    # Probe 1 succeeds → first success counted
    permit1.record_success()

    # Probe 2 succeeds → circuit closes (epoch incremented)
    permit2.record_success()
    assert cb.state == CircuitBreakerState.CLOSED

    # Record the last_failure_time after circuit closed
    stats_after_close = cb.get_stats()
    lft_before = stats_after_close["last_failure_time"]
    assert lft_before == failure_time_after_close

    # Now advance clock significantly
    clock.advance(100.0)

    # Create a new probe in CLOSED state and capture its epoch
    closed_permit = cb.acquire_permit()
    assert closed_permit is not None

    # Force a state change to make the permit stale
    # First, trip the breaker to OPEN via legacy API (bypasses permit check)
    cb.record_failure(GeminiUnavailableError("Trip again"))
    lft_after_trip = cb.get_stats()["last_failure_time"]

    # Now advance clock again
    clock.advance(200.0)

    # Resolve the stale closed_permit with a failure — it must be ignored
    closed_permit.record_failure(GeminiUnavailableError("Stale failure"))

    # last_failure_time must NOT have been updated by the stale permit
    stats_final = cb.get_stats()
    assert stats_final["last_failure_time"] == lft_after_trip


def test_stale_failure_does_not_change_failure_count() -> None:
    """F-03 Test 2: A stale permit failure must not modify current failure count."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Acquire a permit in CLOSED state at epoch 0
    permit_old = cb.acquire_permit()
    assert permit_old is not None
    epoch_old = permit_old.epoch

    # Trip circuit to OPEN (epoch incremented via legacy API)
    for _ in range(3):
        cb.record_failure(GeminiUnavailableError("Fail"))
    assert cb.state == CircuitBreakerState.OPEN
    assert cb._epoch > epoch_old  # noqa: SLF001 — test internals

    # Wait for recovery → HALF_OPEN (epoch incremented again)
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Record failure count before stale resolution
    stats_before = cb.get_stats()
    fc_before = stats_before["failure_count"]

    # Resolve the stale old permit with a recoverable failure
    permit_old.record_failure(GeminiUnavailableError("Stale recoverable"))

    # failure_count must not have changed
    stats_after = cb.get_stats()
    assert stats_after["failure_count"] == fc_before
    # State must still be HALF_OPEN (stale permit cannot reopen)
    assert cb.state == CircuitBreakerState.HALF_OPEN


def test_stale_failure_does_not_change_half_open_permit_count() -> None:
    """F-03 Test 3: A stale permit failure must not modify _half_open_calls."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # CLOSED → OPEN
    cb.record_failure(GeminiUnavailableError("Fail"))
    assert cb.state == CircuitBreakerState.OPEN

    # OPEN → HALF_OPEN
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Acquire first probe
    probe1 = cb.acquire_permit()
    assert probe1 is not None and probe1.is_probe

    # Resolve first probe with recoverable failure → OPEN (epoch changes)
    probe1.record_failure(GeminiUnavailableError("Probe failed"))
    assert cb.state == CircuitBreakerState.OPEN

    # OPEN → new HALF_OPEN cycle
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Acquire a new probe in the new epoch
    probe2 = cb.acquire_permit()
    assert probe2 is not None
    ho_before = cb.get_stats()["half_open_calls"]
    assert ho_before == 1

    # Now fabricate a stale failure using the old probe1 permit
    # (already resolved but double-resolution is a no-op anyway;
    #  to test epoch logic, create a fresh permit at old epoch)
    from core.llm.router import CircuitBreakerPermit

    stale_permit = CircuitBreakerPermit(
        breaker=cb,
        state=CircuitBreakerState.HALF_OPEN,
        epoch=probe1.epoch,  # Old epoch
        is_probe=True,
    )
    stale_permit.record_failure(GeminiAuthError("Stale non-recoverable"))

    # half_open_calls must remain unchanged
    assert cb.get_stats()["half_open_calls"] == ho_before

    # Clean up: resolve probe2 so the breaker isn't left in a dangling state
    probe2.release()


def test_current_epoch_recoverable_failure_updates_timestamp_and_reopens() -> None:
    """F-03 Test 4: A current-epoch recoverable failure updates timestamp and reopens."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Trip to OPEN, then HALF_OPEN
    cb.record_failure(GeminiUnavailableError("Initial"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    lft_before = cb.get_stats()["last_failure_time"]

    # Advance clock so the new failure has a distinct timestamp
    clock.advance(5.0)

    probe = cb.acquire_permit()
    assert probe is not None

    # Current-epoch recoverable failure
    probe.record_failure(GeminiUnavailableError("Probe 503"))

    stats = cb.get_stats()
    assert cb.state == CircuitBreakerState.OPEN
    assert stats["last_failure_time"] > lft_before


def test_current_epoch_non_recoverable_failure_updates_timestamp_and_recycles() -> None:
    """F-03 Test 5: A current-epoch non-recoverable failure updates timestamp and recycles."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Trip to OPEN, then HALF_OPEN
    cb.record_failure(GeminiUnavailableError("Initial"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    lft_before = cb.get_stats()["last_failure_time"]
    clock.advance(5.0)

    probe = cb.acquire_permit()
    assert probe is not None

    # Current-epoch non-recoverable failure
    probe.record_failure(GeminiAuthError("401 Auth error"))

    stats = cb.get_stats()
    # Circuit remains HALF_OPEN (non-recoverable does not reopen)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    # Timestamp was updated (this IS a current-epoch failure event)
    assert stats["last_failure_time"] > lft_before
    # Probe permit was recycled
    assert stats["half_open_calls"] == 0
    assert cb.can_execute() is True


def test_stale_success_cannot_close_circuit_regression() -> None:
    """F-03 Test 6: Existing stale-success protection remains intact (regression guard)."""
    clock = MockClock(start=1000.0)
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=1,
    )
    cb = CircuitBreaker("gemini", policy, clock=clock)

    # Trip to OPEN, then HALF_OPEN
    cb.record_failure(GeminiUnavailableError("Fail"))
    clock.advance(35.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Acquire a probe
    probe = cb.acquire_permit()
    assert probe is not None

    # Probe fails with recoverable error → circuit OPEN (epoch change)
    probe.record_failure(GeminiUnavailableError("Probe fail"))
    assert cb.state == CircuitBreakerState.OPEN

    # Fabricate a stale success permit from before the reopen
    from core.llm.router import CircuitBreakerPermit

    stale_success_permit = CircuitBreakerPermit(
        breaker=cb,
        state=CircuitBreakerState.HALF_OPEN,
        epoch=probe.epoch,  # Old epoch
        is_probe=True,
    )
    stale_success_permit.record_success()

    # Circuit must still be OPEN — stale success must not close it
    assert cb.state == CircuitBreakerState.OPEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
