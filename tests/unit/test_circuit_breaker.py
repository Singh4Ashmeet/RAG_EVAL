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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
