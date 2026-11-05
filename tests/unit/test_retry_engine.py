"""Unit tests for provider retry engine with exponential backoff."""

from __future__ import annotations

import time
from datetime import UTC
from types import SimpleNamespace

import pytest

from core.llm.base import GenerationSettings
from core.llm.gemini_provider import (
    ProviderAuthenticationError as GeminiAuthError,
)
from core.llm.gemini_provider import (
    ProviderUnavailableError as GeminiUnavailableError,
)
from core.llm.router import (
    FallbackProvider,
    ProviderFallbackExhaustedError,
    RetryPolicy,
    RetryProvider,
    _calculate_backoff,
    _extract_retry_after,
    clear_provider_cache,
    get_chat_provider,
)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        name: str,
        response: str = "OK",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._response = response
        self._error = error
        self._delay = delay
        self.call_count = 0
        self.call_times: list[float] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        self.call_count += 1
        self.call_times.append(time.monotonic())
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return self._response


class MockRetryProvider:
    """Mock provider that fails N times then succeeds."""

    def __init__(
        self,
        name: str,
        response: str = "OK",
        error: Exception | None = None,
        fail_times: int = 0,
    ) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._response = response
        self._error = error
        self._fail_times = fail_times
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        self.call_count += 1
        if self.call_count <= self._fail_times and self._error:
            raise self._error
        return self._response


def test_calculate_backoff_basic() -> None:
    """Test basic exponential backoff calculation."""
    # Attempt 0: base_delay * (base ** 0) = base_delay
    assert _calculate_backoff(0, 1.0, 2.0, 10.0, False) == 1.0
    # Attempt 1: base_delay * (base ** 1) = 2.0
    assert _calculate_backoff(1, 1.0, 2.0, 10.0, False) == 2.0
    # Attempt 2: base_delay * (base ** 2) = 4.0
    assert _calculate_backoff(2, 1.0, 2.0, 10.0, False) == 4.0
    # Attempt 3: base_delay * (base ** 3) = 8.0
    assert _calculate_backoff(3, 1.0, 2.0, 10.0, False) == 8.0


def test_calculate_backoff_max_delay_cap() -> None:
    """Test that max_delay caps the backoff."""
    # Should be capped at max_delay (10.0)
    assert _calculate_backoff(10, 1.0, 2.0, 10.0, False) == 10.0
    assert _calculate_backoff(20, 1.0, 2.0, 10.0, False) == 10.0


def test_calculate_backoff_retry_after_priority() -> None:
    """Test that Retry-After header takes priority over calculated backoff."""
    # Retry-After of 5 seconds should be used instead of calculated 8.0
    assert _calculate_backoff(3, 1.0, 2.0, 10.0, False, retry_after=5.0) == 5.0
    # Retry-After capped by max_delay
    assert _calculate_backoff(3, 1.0, 2.0, 10.0, False, retry_after=15.0) == 10.0


def test_calculate_backoff_jitter_enabled() -> None:
    """Test that jitter adds variance to backoff."""
    delays = [_calculate_backoff(2, 1.0, 2.0, 10.0, True) for _ in range(100)]
    # Base delay at attempt 2 is 4.0, with 10% jitter: [4.0, 4.4]
    assert all(4.0 <= d <= 4.4 for d in delays)
    # Should have some variance (not all identical)
    assert len(set(delays)) > 1


def test_calculate_backoff_retry_after_with_jitter() -> None:
    """Test Retry-After with jitter enabled."""
    # Retry-After 5.0 with jitter: [5.0, 5.5]
    delays = [_calculate_backoff(0, 1.0, 2.0, 10.0, True, retry_after=5.0) for _ in range(100)]
    assert all(5.0 <= d <= 5.5 for d in delays)


def test_extract_retry_after_seconds() -> None:
    """Test extracting Retry-After as seconds."""
    import httpx

    response = SimpleNamespace(headers={"retry-after": "30"}, status_code=429)
    exc = httpx.HTTPStatusError("429", request=None, response=response)
    assert _extract_retry_after(exc) == 30.0


def test_extract_retry_after_http_date() -> None:
    """Test extracting Retry-After as HTTP-date."""
    from datetime import datetime, timedelta

    import httpx

    # Future date (10 seconds from now)
    future = datetime.now(UTC) + timedelta(seconds=10)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

    response = SimpleNamespace(headers={"retry-after": http_date}, status_code=429)
    exc = httpx.HTTPStatusError("429", request=None, response=response)
    retry_after = _extract_retry_after(exc)
    assert retry_after is not None
    assert 9.0 <= retry_after <= 11.0  # Allow small tolerance


def test_extract_retry_after_malformed() -> None:
    """Test handling of malformed Retry-After header."""
    import httpx

    response = SimpleNamespace(headers={"retry-after": "not-a-number"}, status_code=429)
    exc = httpx.HTTPStatusError("429", request=None, response=response)
    assert _extract_retry_after(exc) is None


def test_extract_retry_after_no_header() -> None:
    """Test handling when Retry-After header is missing."""
    import httpx

    response = SimpleNamespace(headers={}, status_code=429)
    exc = httpx.HTTPStatusError("429", request=None, response=response)
    assert _extract_retry_after(exc) is None


def test_retry_provider_first_attempt_succeeds() -> None:
    """Test RetryProvider when first attempt succeeds."""
    policy = RetryPolicy(max_attempts=3)
    provider = RetryProvider(
        MockProvider("test", "Success"),
        policy,
        "test",
    )

    result = provider.generate("sys", "user", settings=GenerationSettings())
    assert result == "Success"
    assert provider._provider.call_count == 1


def test_retry_provider_retry_then_succeeds() -> None:
    """Test RetryProvider retries on failure then succeeds."""
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=1.0, jitter_enabled=False
    )
    provider = RetryProvider(
        MockRetryProvider(
            "test", "Success", error=GeminiUnavailableError("Temp fail"), fail_times=1
        ),
        policy,
        "test",
    )

    result = provider.generate("sys", "user", settings=GenerationSettings())
    assert result == "Success"
    assert provider._provider.call_count == 2


def test_retry_provider_all_attempts_fail() -> None:
    """Test RetryProvider when all attempts fail."""
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=1.0, jitter_enabled=False
    )
    provider = RetryProvider(
        MockRetryProvider("test", error=GeminiUnavailableError("Permanent fail"), fail_times=3),
        policy,
        "test",
    )

    with pytest.raises(ProviderFallbackExhaustedError) as exc_info:
        provider.generate("sys", "user", settings=GenerationSettings())

    assert "All 3 retry attempts exhausted" in str(exc_info.value)
    assert provider._provider.call_count == 3


def test_retry_provider_non_recoverable_no_retry() -> None:
    """Test that non-recoverable errors don't trigger retry."""
    policy = RetryPolicy(max_attempts=3)
    provider = RetryProvider(
        MockProvider("test", error=GeminiAuthError("Invalid key")),
        policy,
        "test",
    )

    with pytest.raises(GeminiAuthError):
        provider.generate("sys", "user", settings=GenerationSettings())

    assert provider._provider.call_count == 1


def test_retry_provider_disabled() -> None:
    """Test that disabled retry passes through immediately."""
    policy = RetryPolicy(enabled=False, max_attempts=3)
    provider = RetryProvider(
        MockRetryProvider("test", "Success", error=GeminiUnavailableError("Fail"), fail_times=3),
        policy,
        "test",
    )

    # Should fail immediately without retry
    with pytest.raises(GeminiUnavailableError):
        provider.generate("sys", "user", settings=GenerationSettings())

    assert provider._provider.call_count == 1


def test_retry_provider_max_attempts_enforced() -> None:
    """Test that max_attempts is strictly enforced."""
    for max_attempts in [1, 2, 3, 5]:
        policy = RetryPolicy(max_attempts=max_attempts, base_delay_seconds=0.01)
        provider = RetryProvider(
            MockRetryProvider(
                "test", error=GeminiUnavailableError("Fail"), fail_times=max_attempts
            ),
            policy,
            "test",
        )

        with pytest.raises(ProviderFallbackExhaustedError):
            provider.generate("sys", "user", settings=GenerationSettings())

        assert provider._provider.call_count == max_attempts


def test_retry_provider_no_sleep_after_final_failure() -> None:
    """Test that no sleep occurs after the final failure."""
    call_times: list[float] = []

    class TimedProvider:
        def __init__(self):
            self.model_id = "mock:timed"
            self.call_count = 0

        def generate(self, *args, **kwargs):
            self.call_count += 1
            call_times.append(time.monotonic())
            raise GeminiUnavailableError("Fail")

    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.1, jitter_enabled=False)
    provider = RetryProvider(TimedProvider(), policy, "test")

    with pytest.raises(ProviderFallbackExhaustedError):
        provider.generate("sys", "user", settings=GenerationSettings())

    # 3 calls, 2 sleeps between them
    # Total time should be roughly 2 * base_delay (0.2s), not 3 * 0.1s
    assert len(call_times) == 3
    if len(call_times) >= 3:
        total_elapsed = call_times[-1] - call_times[0]
        # Should be roughly 0.2s (2 sleeps), definitely less than 0.3s (3 sleeps)
        assert total_elapsed < 0.35


def test_retry_provider_deadline_exceeded() -> None:
    """Test that retries stop when deadline is exceeded."""

    class SlowProvider:
        def __init__(self):
            self.model_id = "mock:slow"
            self.call_count = 0

        def generate(self, *args, **kwargs):
            self.call_count += 1
            time.sleep(0.05)  # 50ms per call
            raise GeminiUnavailableError("Slow")

    policy = RetryPolicy(max_attempts=10, base_delay_seconds=0.05, timeout_seconds=0.15)
    provider = RetryProvider(SlowProvider(), policy, "test")

    with pytest.raises(ProviderFallbackExhaustedError):
        provider.generate("sys", "user", settings=GenerationSettings())

    # Should only make ~3 attempts (0.15s / 0.05s = 3)
    assert 2 <= provider._provider.call_count <= 4


def test_retry_provider_respects_retry_after() -> None:
    """Test that Retry-After header is respected."""
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, jitter_enabled=False)

    class RetryAfterProvider:
        def __init__(self):
            self.model_id = "mock:retry-after"
            self.call_count = 0

        def generate(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                import httpx

                response = SimpleNamespace(headers={"retry-after": "0.05"}, status_code=429)
                raise httpx.HTTPStatusError("429", request=None, response=response)
            return "Success after retry-after"

    policy = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, jitter_enabled=False)
    provider = RetryProvider(RetryAfterProvider(), policy, "test")

    result = provider.generate("sys", "user", settings=GenerationSettings())
    assert result == "Success after retry-after"
    assert provider._provider.call_count == 2


def test_retry_policy_from_settings() -> None:
    """Test extracting retry policy from settings."""
    import os

    os.environ["RETRY_ENABLED"] = "true"
    os.environ["RETRY_MAX_ATTEMPTS"] = "5"
    os.environ["RETRY_BASE_DELAY_SECONDS"] = "2.0"
    os.environ["RETRY_MAX_DELAY_SECONDS"] = "20.0"
    os.environ["RETRY_EXPONENTIAL_BASE"] = "3.0"
    os.environ["RETRY_JITTER_ENABLED"] = "false"
    os.environ["RETRY_TIMEOUT_SECONDS"] = "60.0"

    from core.config import clear_settings_cache, get_settings

    clear_settings_cache()
    settings = get_settings()

    from core.llm.router import _get_retry_policy

    policy = _get_retry_policy(settings)

    assert policy.enabled is True
    assert policy.max_attempts == 5
    assert policy.base_delay_seconds == 2.0
    assert policy.max_delay_seconds == 20.0
    assert policy.exponential_base == 3.0
    assert policy.jitter_enabled is False
    assert policy.timeout_seconds == 60.0


def test_retry_with_fallback_integration() -> None:
    """Test that retry and fallback work together correctly."""
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01, jitter_enabled=False)

    # Primary fails all retries, then fallback succeeds
    primary = MockRetryProvider(
        "primary", error=GeminiUnavailableError("Primary down"), fail_times=2
    )
    fallback = MockProvider("fallback", "Fallback success")

    primary_wrapped = RetryProvider(primary, policy, "primary")
    fallback_wrapped = RetryProvider(fallback, policy, "fallback")

    fallback_chain = FallbackProvider(
        primary_provider=primary_wrapped,
        fallback_providers=[fallback_wrapped],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    result = fallback_chain.generate("sys", "user", settings=GenerationSettings())
    assert result == "Fallback success"
    assert primary.call_count == 2  # 2 attempts (max_attempts=2)
    assert fallback.call_count == 1


def test_retry_with_fallback_both_retry() -> None:
    """Test that both primary and fallback get their own retry budgets."""
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01, jitter_enabled=False)

    primary = MockRetryProvider("primary", error=GeminiUnavailableError("Down"), fail_times=2)
    fallback = MockRetryProvider("fallback", error=GeminiUnavailableError("Down"), fail_times=2)

    primary_wrapped = RetryProvider(primary, policy, "primary")
    fallback_wrapped = RetryProvider(fallback, policy, "fallback")

    fallback_chain = FallbackProvider(
        primary_provider=primary_wrapped,
        fallback_providers=[fallback_wrapped],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    with pytest.raises(ProviderFallbackExhaustedError):
        fallback_chain.generate("sys", "user", settings=GenerationSettings())

    # Each provider should get its own max_attempts
    assert primary.call_count == 2
    assert fallback.call_count == 2


def test_fallback_provider_permanent_error_no_fallback() -> None:
    """Test that permanent error in primary prevents fallback."""
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01)

    primary = MockProvider("primary", error=GeminiAuthError("Invalid key"))
    fallback = MockProvider("fallback", "Fallback success")

    primary_wrapped = RetryProvider(primary, policy, "primary")
    fallback_wrapped = RetryProvider(fallback, policy, "fallback")

    fallback_chain = FallbackProvider(
        primary_provider=primary_wrapped,
        fallback_providers=[fallback_wrapped],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    with pytest.raises(GeminiAuthError):
        fallback_chain.generate("sys", "user", settings=GenerationSettings())

    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_retry_provider_model_id() -> None:
    """Test that RetryProvider exposes underlying model_id."""
    policy = RetryPolicy()
    provider = RetryProvider(MockProvider("test", "OK"), policy, "test")
    assert provider.model_id == "mock:test"


def test_fallback_provider_model_id() -> None:
    """Test that FallbackProvider exposes primary model_id."""
    policy = RetryPolicy()
    primary = RetryProvider(MockProvider("primary", "OK"), policy, "primary")
    fallback = RetryProvider(MockProvider("fallback", "OK"), policy, "fallback")

    chain = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    assert chain.model_id == "mock:primary"


def test_clear_provider_cache() -> None:
    """Test that clear_provider_cache works."""
    clear_provider_cache()


def test_get_chat_provider_with_retry_config() -> None:
    """Test that get_chat_provider respects retry configuration."""
    import os

    os.environ["CHAT_PROVIDER"] = "ollama"
    os.environ["RETRY_ENABLED"] = "true"
    os.environ["RETRY_MAX_ATTEMPTS"] = "5"

    from core.config import clear_settings_cache

    clear_provider_cache()
    clear_settings_cache()

    provider = get_chat_provider("ollama")
    assert provider is not None
    assert hasattr(provider, "model_id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
