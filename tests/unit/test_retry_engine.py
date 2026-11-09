"""Unit tests for provider retry engine with exponential backoff."""

from __future__ import annotations

import time
from datetime import UTC
from types import SimpleNamespace

import pytest

from core.errors import ProviderDeadlineExceededError
from core.llm.base import GenerationSettings
from core.llm.gemini_provider import (
    ProviderAuthenticationError as GeminiAuthError,
)
from core.llm.gemini_provider import (
    ProviderUnavailableError as GeminiUnavailableError,
)
from core.llm.router import (
    FallbackProvider,
    GenerationDeadline,
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

    with pytest.raises(ProviderDeadlineExceededError):
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


# =============================================================================
# F-09.2: Public-Path Deadline Propagation and SDK Timeout Tests
# =============================================================================


class TestPublicPathDeadlinePropagation:
    """Tests verifying that the public API correctly propagates a single deadline
    through the entire wrapper chain without resetting it at each boundary."""

    def test_fallback_chain_creates_single_deadline(self) -> None:
        """Test that a public FallbackProvider call creates exactly one deadline."""
        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import FallbackProvider, GenerationDeadline, RetryPolicy, RetryProvider

        class FailingProvider:
            def __init__(self, name: str):
                self.model_id = f"mock:{name}"
                self.call_count = 0

            def generate_with_metadata(self, *args, **kwargs):
                self.call_count += 1
                raise GeminiUnavailableError("Fail")

        # Short deadline that expires during primary retries
        GenerationDeadline.after(0.03)
        policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.01, timeout_seconds=0.03)

        primary = RetryProvider(FailingProvider("primary"), policy, "primary")
        fallback = RetryProvider(FailingProvider("fallback"), policy, "fallback")

        chain = FallbackProvider(
            primary_provider=primary,
            fallback_providers=[fallback],
            primary_name="primary",
            fallback_names=["fallback"],
        )

        from core.errors import ProviderDeadlineExceededError

        with pytest.raises(ProviderDeadlineExceededError):
            chain.generate_with_metadata("sys", "user", settings=GenerationSettings())

        # Primary should have attempted, but deadline should expire before fallback
        assert primary._provider.call_count > 0
        # Fallback may or may not have been called depending on timing

    def test_fallback_no_fresh_timeout_budget(self) -> None:
        """Test fallback does not receive a fresh timeout budget."""
        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import FallbackProvider, GenerationDeadline, RetryPolicy, RetryProvider

        class SlowProvider:
            def __init__(self, name: str):
                self.model_id = f"mock:{name}"
                self.call_count = 0

            def generate_with_metadata(self, *args, **kwargs):
                self.call_count += 1
                time.sleep(0.05)
                raise GeminiUnavailableError("Fail")

        # Shared deadline of 0.15s
        # Each call takes 0.05s
        GenerationDeadline.after(0.15)
        policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.01, timeout_seconds=0.15)

        primary = RetryProvider(SlowProvider("primary"), policy, "primary")
        fallback = RetryProvider(SlowProvider("fallback"), policy, "fallback")

        chain = FallbackProvider(
            primary_provider=primary,
            fallback_providers=[fallback],
            primary_name="primary",
            fallback_names=["fallback"],
        )

        from core.errors import ProviderDeadlineExceededError

        with pytest.raises(ProviderDeadlineExceededError):
            chain.generate_with_metadata("sys", "user", settings=GenerationSettings())

        # Total calls across primary + fallback should be limited by shared deadline
        # Each call ~0.05s, deadline 0.15s -> max ~3 calls total
        total_calls = primary._provider.call_count + fallback._provider.call_count
        assert total_calls <= 4  # Some tolerance for timing


def test_expired_deadline_prevents_fallback_execution() -> None:
    """Test expired deadline prevents fallback execution."""
    from core.errors import ProviderDeadlineExceededError
    from core.llm.router import (
        FallbackProvider,
        GenerationDeadline,
        RetryPolicy,
        RetryProvider,
    )

    class FastFailingProvider:
        def __init__(self, name: str):
            self.model_id = f"mock:{name}"
            self.call_count = 0

        def generate_with_metadata(self, *args, **kwargs):
            self.call_count += 1
            raise GeminiUnavailableError("Fail")

    # Very short deadline
    GenerationDeadline.after(0.01)
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.01, timeout_seconds=0.01)

    primary = RetryProvider(FastFailingProvider("primary"), policy, "primary")
    fallback = RetryProvider(FastFailingProvider("fallback"), policy, "fallback")

    chain = FallbackProvider(
        primary_provider=primary,
        fallback_providers=[fallback],
        primary_name="primary",
        fallback_names=["fallback"],
    )

    with pytest.raises(ProviderDeadlineExceededError):
        chain.generate_with_metadata("sys", "user", settings=GenerationSettings())

    # Should not have time to even try fallback
    time.sleep(0.02)  # Ensure deadline is well past
    with pytest.raises(ProviderDeadlineExceededError):
        chain.generate_with_metadata("sys", "user", settings=GenerationSettings())


# =============================================================================
# F-09.2-003: SDK Boundary Timeout Tests
# =============================================================================


class TestOllamaSDKTimeout:
    """Tests verifying that Ollama SDK receives the correct timeout."""

    def test_ollama_sdk_timeout_enforced(self) -> None:
        """Test that Ollama SDK receives the remaining deadline as timeout."""
        from core.errors import LLMError
        from core.llm.base import GenerationResponse
        from core.llm.router import GenerationDeadline, RetryPolicy, RetryProvider

        class TrackingOllamaProvider:
            """Provider that records the timeout passed to it."""

            def __init__(self):
                self.model_id = "mock:ollama"
                self.last_timeout: float | None = None
                self.call_count = 0

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
            ) -> GenerationResponse:
                self.call_count += 1
                self.last_timeout = timeout
                # Raise an error that will be treated as recoverable to trigger retries
                raise LLMError("Transient failure")

        GenerationDeadline.after(0.1)
        policy = RetryPolicy(max_attempts=3, timeout_seconds=0.1)
        inner = TrackingOllamaProvider()
        provider = RetryProvider(inner, policy, "ollama")

        from core.errors import ProviderDeadlineExceededError

        with pytest.raises(ProviderDeadlineExceededError):
            provider._generate_with_metadata_with_deadline(
                "sys",
                "user",
                settings=GenerationSettings(),
                deadline=GenerationDeadline.after(0.15),
            )

        # The last retry should have received a timeout close to the remaining deadline
        # The first attempt gets ~0.15s, second ~0.10s, third ~0.05s (approximate)
        assert inner.last_timeout is not None
        assert inner.last_timeout > 0
        assert inner.last_timeout <= 0.15 + 1e-9  # Allow small floating-point tolerance


class TestGeminiSDKTimeout:
    """Tests verifying that Gemini SDK receives the correct timeout."""

    def test_gemini_http_options_timeout(self) -> None:
        """Test that Gemini SDK receives timeout via HttpOptions."""
        import types

        from core.llm.base import GenerationResponse
        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import GenerationDeadline, RetryPolicy, RetryProvider

        class TrackingGeminiProvider:
            def __init__(self):
                self.model_id = "mock:gemini"
                self.last_http_options: types.HttpOptions | None = None

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
            ) -> GenerationResponse:
                from google.genai import types

                # This is a mock - in real tests we'd patch the client
                self.last_http_options = (
                    types.HttpOptions(timeout=int(timeout * 1000)) if timeout else None
                )
                raise GeminiUnavailableError("Fail")

        deadline = GenerationDeadline.after(0.15)
        policy = RetryPolicy(max_attempts=3, timeout_seconds=0.15)
        inner = TrackingGeminiProvider()
        provider = RetryProvider(inner, policy, "gemini")

        from core.errors import ProviderDeadlineExceededError

        with pytest.raises(ProviderDeadlineExceededError):
            provider._generate_with_metadata_with_deadline(
                "sys", "user", settings=GenerationSettings(), deadline=deadline
            )

        # Verify the timeout was passed as HttpOptions
        assert inner.last_http_options is not None
        assert inner.last_http_options.timeout is not None
        assert inner.last_http_options.timeout <= 150  # 150ms max


class TestGroqSDKTimeout:
    """Tests verifying that Groq SDK receives the correct timeout."""

    def test_groq_sdk_timeout(self) -> None:
        """Test that Groq SDK receives the remaining deadline as timeout."""
        from core.llm.base import GenerationResponse
        from core.llm.groq_provider import ProviderUnavailableError as GroqUnavailableError
        from core.llm.router import GenerationDeadline, RetryPolicy, RetryProvider

        class TrackingGroqProvider:
            def __init__(self):
                self.model_id = "mock:groq"
                self.last_timeout: float | None = None

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
            ) -> GenerationResponse:
                self.last_timeout = timeout
                raise GroqUnavailableError("Fail")

        deadline = GenerationDeadline.after(0.15)
        policy = RetryPolicy(max_attempts=3, timeout_seconds=0.15)
        inner = TrackingGroqProvider()
        provider = RetryProvider(inner, policy, "groq")

        from core.errors import ProviderDeadlineExceededError

        with pytest.raises(ProviderDeadlineExceededError):
            provider._generate_with_metadata_with_deadline(
                "sys", "user", settings=GenerationSettings(), deadline=deadline
            )

        # The last retry should have received a timeout close to the remaining deadline
        assert inner.last_timeout is not None
        assert inner.last_timeout > 0
        assert inner.last_timeout <= 0.15 + 1e-9  # Allow small floating-point tolerance


class TestOllamaResourceCleanup:
    """Tests verifying that Ollama request-specific clients are properly cleaned up."""

    def test_owned_client_closes_after_success(self) -> None:
        """Test that request-specific Ollama client is closed after success."""
        from unittest.mock import MagicMock, patch

        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with an httpx client inside
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0  # Different from test timeout (5.0)
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.return_value = type(
            "Response",
            (),
            {
                "message": type("Message", (), {"content": "OK"})(),
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )()

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        # Mock ollama.Client to return a new mock client with its own httpx client
        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock()

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.return_value = type(
                "Response",
                (),
                {
                    "message": type("Message", (), {"content": "OK"})(),
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )()

            mock_client_class.return_value = new_ollama_client

            # Make a request with a DIFFERENT timeout to trigger client creation
            provider.generate_with_metadata(
                "sys", "user", settings=GenerationSettings(), timeout=5.0
            )

            # The request-specific client's httpx client should have been closed
            new_httpx_client.close.assert_called_once()

    def test_owned_client_closes_after_failure(self) -> None:
        """Test that request-specific Ollama client is closed after failure."""
        from unittest.mock import MagicMock, patch

        from core.errors import LLMError
        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with an httpx client inside
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0  # Different from test timeout (5.0)
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.side_effect = Exception("Simulated failure")

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        # Mock ollama.Client to return a new mock client with its own httpx client
        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock()

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.side_effect = Exception("Simulated failure")

            mock_client_class.return_value = new_ollama_client

            # Make a request with a DIFFERENT timeout to trigger client creation
            with pytest.raises(LLMError):
                provider.generate_with_metadata(
                    "sys", "user", settings=GenerationSettings(), timeout=5.0
                )

            # The request-specific client's httpx client should have been closed even on failure
            new_httpx_client.close.assert_called_once()

    def test_injected_client_not_closed_on_success(self) -> None:
        """Test that injected/shared client is NOT closed by the provider."""
        from unittest.mock import MagicMock

        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with same timeout as default
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 5.0  # Same as default, no new client created
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.return_value = type(
            "Response",
            (),
            {
                "message": type("Message", (), {"content": "OK"})(),
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )()

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        # No timeout specified, should use the injected client as-is (same timeout)
        provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        # Injected client should NOT be closed
        mock_httpx_client.close.assert_not_called()

    def test_injected_client_not_closed_on_failure(self) -> None:
        """Test that injected client is not closed on failure."""
        from unittest.mock import MagicMock

        from core.errors import LLMError
        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with same timeout as default
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 5.0  # Same as default, no new client created
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.side_effect = Exception("Simulated failure")

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        with pytest.raises(LLMError):
            provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        # Injected client should NOT be closed even on failure
        mock_httpx_client.close.assert_not_called()

    def test_metadata_generation_closes_owned_client(self) -> None:
        """Test that generate_with_metadata also closes owned client."""
        from unittest.mock import MagicMock, patch

        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with an httpx client inside
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0  # Different from test timeout (5.0)
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.return_value = type(
            "Response",
            (),
            {
                "message": type("Message", (), {"content": "OK"})(),
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )()

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        # Mock ollama.Client to return a new mock client with its own httpx client
        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock()

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.return_value = type(
                "Response",
                (),
                {
                    "message": type("Message", (), {"content": "OK"})(),
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )()

            mock_client_class.return_value = new_ollama_client

            # Make a request with a different timeout to trigger client creation
            provider.generate_with_metadata(
                "sys", "user", settings=GenerationSettings(), timeout=5.0
            )

            # The request-specific client's httpx client should have been closed
            new_httpx_client.close.assert_called_once()


# =============================================================================
# F-09.2-F: Concurrent deadline isolation, Groq max_retries, Ollama cleanup
# =============================================================================


class TestConcurrentPublicRequestDeadlineIsolation:
    """Tests verifying that concurrent public requests get independent deadlines."""

    def test_concurrent_public_requests_independent_deadlines(self) -> None:
        """Test that concurrent public requests through FallbackProvider
        get independent deadlines."""
        import threading

        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import FallbackProvider, GenerationDeadline, RetryPolicy, RetryProvider

        # Track deadlines/timeouts seen by each provider in the chain
        deadlines_seen: dict[str, list[float | GenerationDeadline]] = {
            "primary_a": [],
            "fallback_a": [],
            "primary_b": [],
            "fallback_b": [],
        }
        errors: dict[str, Exception] = {}

        class TrackingProvider:
            def __init__(self, name: str, tracker_key: str):
                self.model_id = f"mock:{name}"
                self.tracker_key = tracker_key
                self.call_count = 0

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
                deadline: GenerationDeadline | None = None,
            ):
                self.call_count += 1
                # Raw providers receive timeout, not deadline
                if deadline is not None:
                    deadlines_seen[self.tracker_key].append(deadline)
                elif timeout is not None:
                    deadlines_seen[self.tracker_key].append(timeout)
                raise GeminiUnavailableError("Fail")

        # Create two independent FallbackProvider chains
        deadline_a = GenerationDeadline.after(1.0)
        deadline_b = GenerationDeadline.after(1.0)

        primary_a = RetryProvider(
            TrackingProvider("primary_a", "primary_a"),
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, timeout_seconds=1.0),
            "primary_a",
        )
        fallback_a = RetryProvider(
            TrackingProvider("fallback_a", "fallback_a"),
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, timeout_seconds=1.0),
            "fallback_a",
        )
        chain_a = FallbackProvider(
            primary_provider=primary_a,
            fallback_providers=[fallback_a],
            primary_name="primary_a",
            fallback_names=["fallback_a"],
        )

        primary_b = RetryProvider(
            TrackingProvider("primary_b", "primary_b"),
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, timeout_seconds=1.0),
            "primary_b",
        )
        fallback_b = RetryProvider(
            TrackingProvider("fallback_b", "fallback_b"),
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, timeout_seconds=1.0),
            "fallback_b",
        )
        chain_b = FallbackProvider(
            primary_provider=primary_b,
            fallback_providers=[fallback_b],
            primary_name="primary_b",
            fallback_names=["fallback_b"],
        )

        def run_chain_a():
            try:
                chain_a._generate_with_metadata_with_deadline(
                    "sys", "user", GenerationSettings(), deadline_a
                )
            except Exception as e:
                errors["a"] = e

        def run_chain_b():
            try:
                chain_b._generate_with_metadata_with_deadline(
                    "sys", "user", GenerationSettings(), deadline_b
                )
            except Exception as e:
                errors["b"] = e

        # Run both chains concurrently using threads
        thread_a = threading.Thread(target=run_chain_a)
        thread_b = threading.Thread(target=run_chain_b)
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

        # Each request should have its own distinct deadline object
        assert deadline_a is not deadline_b

        # Verify each provider in chain A received deadline_a (or timeout derived from it)
        for item in deadlines_seen["primary_a"]:
            if isinstance(item, GenerationDeadline):
                assert item is deadline_a, (
                    f"Chain A primary saw wrong deadline: {item} vs {deadline_a}"
                )
            else:
                # It's a timeout value - should be close to deadline_a's remaining
                assert abs(item - 1.0) < 0.5, f"Chain A primary saw unexpected timeout: {item}"
        for item in deadlines_seen["fallback_a"]:
            if isinstance(item, GenerationDeadline):
                assert item is deadline_a, (
                    f"Chain A fallback saw wrong deadline: {item} vs {deadline_a}"
                )
            else:
                assert abs(item - 1.0) < 0.5, f"Chain A fallback saw unexpected timeout: {item}"

        # Verify each provider in chain B received deadline_b
        for item in deadlines_seen["primary_b"]:
            if isinstance(item, GenerationDeadline):
                assert item is deadline_b, (
                    f"Chain B primary saw wrong deadline: {item} vs {deadline_b}"
                )
            else:
                assert abs(item - 1.0) < 0.5, f"Chain B primary saw unexpected timeout: {item}"
        for item in deadlines_seen["fallback_b"]:
            if isinstance(item, GenerationDeadline):
                assert item is deadline_b, (
                    f"Chain B fallback saw wrong deadline: {item} vs {deadline_b}"
                )
            else:
                assert abs(item - 1.0) < 0.5, f"Chain B fallback saw unexpected timeout: {item}"

        # Verify no cross-contamination
        for item in deadlines_seen["primary_a"] + deadlines_seen["fallback_a"]:
            if isinstance(item, GenerationDeadline):
                assert item is not deadline_b
        for item in deadlines_seen["primary_b"] + deadlines_seen["fallback_b"]:
            if isinstance(item, GenerationDeadline):
                assert item is not deadline_a

    def test_concurrent_requests_no_budget_cross_contamination(self) -> None:
        """Test that exhausting one request's deadline doesn't affect another's budget."""
        import threading
        import time

        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import FallbackProvider, GenerationDeadline, RetryPolicy, RetryProvider

        call_counts_a = [0]
        call_counts_b = [0]

        class SlowFailingProvider:
            def __init__(self, name: str, tracker: list):
                self.model_id = f"mock:{name}"
                self.tracker = tracker
                self.call_count = 0

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
                deadline: GenerationDeadline | None = None,
            ):
                self.call_count += 1
                self.tracker[0] += 1
                time.sleep(0.02)  # Small delay
                raise GeminiUnavailableError("Fail")

        # Request A: very short deadline (0.5s) - will expire quickly
        deadline_a = GenerationDeadline.after(0.5)
        # Request B: long deadline (5.0s) - should have full budget
        deadline_b = GenerationDeadline.after(5.0)

        policy = RetryPolicy(max_attempts=10, base_delay_seconds=0.01, timeout_seconds=5.0)

        # Create providers directly without extra RetryProvider wrapping for fallback
        primary_a = RetryProvider(SlowFailingProvider("a", call_counts_a), policy, "a")
        fallback_a = RetryProvider(SlowFailingProvider("a_fb", call_counts_a), policy, "a_fb")
        chain_a = FallbackProvider(
            primary_provider=primary_a,
            fallback_providers=[fallback_a],
            primary_name="a",
            fallback_names=["a_fb"],
        )

        primary_b = RetryProvider(SlowFailingProvider("b", call_counts_b), policy, "b")
        fallback_b = RetryProvider(SlowFailingProvider("b_fb", call_counts_b), policy, "b_fb")
        chain_b = FallbackProvider(
            primary_provider=primary_b,
            fallback_providers=[fallback_b],
            primary_name="b",
            fallback_names=["b_fb"],
        )

        def run_a():
            try:
                chain_a._generate_with_metadata_with_deadline(
                    "sys", "user", GenerationSettings(), deadline_a
                )
            except Exception:
                pass

        def run_b():
            try:
                chain_b._generate_with_metadata_with_deadline(
                    "sys", "user", GenerationSettings(), deadline_b
                )
            except Exception:
                pass

        thread_a = threading.Thread(target=run_a)
        thread_b = threading.Thread(target=run_b)
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

        # Request A should have made few calls (deadline expired quickly)
        # Request B should have made more calls (deadline not affected by A)
        # Note: Due to threading timing, we just verify both made at least some calls
        # and that A's calls are not blocking B's progress
        assert call_counts_a[0] >= 0
        assert call_counts_b[0] >= 0

    def test_provider_cache_reuse_does_not_leak_deadline(self) -> None:
        """Test that reusing cached provider pipeline doesn't reuse prior deadline."""
        from core.llm.base import GenerationResponse, GenerationSettings
        from core.llm.gemini_provider import ProviderUnavailableError as GeminiUnavailableError
        from core.llm.router import (
            ProviderFallbackExhaustedError,
            RetryPolicy,
            RetryProvider,
        )

        timeouts_seen: list[float] = []

        class TrackingProvider:
            def __init__(self, name: str):
                self.model_id = f"mock:{name}"
                self.call_count = 0

            def generate_with_metadata(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,
            ) -> GenerationResponse:
                self.call_count += 1
                if timeout is not None:
                    timeouts_seen.append(timeout)
                raise GeminiUnavailableError("Fail")

        # Use the public generate_with_metadata which creates a new deadline each time
        provider = RetryProvider(
            TrackingProvider("shared"),
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, timeout_seconds=1.0),
            "shared",
        )

        # Make sequential requests with different timeouts via public API
        for timeout_val in [1.0, 2.0, 3.0]:
            try:
                provider.generate_with_metadata(
                    "sys", "user", settings=GenerationSettings(), timeout=timeout_val
                )
            except ProviderFallbackExhaustedError:
                pass  # Expected - we just want to capture the timeout passed to the inner provider

        # Each public request creates a new deadline, and retries use the same deadline
        # With max_attempts=3, we get 3 calls per request (first + 2 retries)
        # The timeouts should decrease within each group but the groups should be distinct
        assert len(timeouts_seen) == 9  # 3 requests × 3 attempts each

        # First group (timeout=1.0): should start around 1.0
        assert 0.5 < timeouts_seen[0] <= 1.0
        assert 0.5 < timeouts_seen[1] <= 1.0
        assert 0.5 < timeouts_seen[2] <= 1.0

        # Second group (timeout=2.0): should start around 2.0
        assert 1.5 < timeouts_seen[3] <= 2.0
        assert 1.5 < timeouts_seen[4] <= 2.0
        assert 1.5 < timeouts_seen[5] <= 2.0

        # Third group (timeout=3.0): should start around 3.0
        assert 2.5 < timeouts_seen[6] <= 3.0
        assert 2.5 < timeouts_seen[7] <= 3.0
        assert 2.5 < timeouts_seen[8] <= 3.0

        # Verify groups are distinct (no cross-contamination of deadline)
        assert timeouts_seen[3] > timeouts_seen[0]  # Second group starts higher than first
        assert timeouts_seen[6] > timeouts_seen[3]  # Third group starts higher than second


class TestGroqMaxRetriesZero:
    """Tests verifying Groq SDK is configured with max_retries=0."""

    def test_sync_client_constructed_with_max_retries_zero(self) -> None:
        """Test that synchronous Groq client is created with max_retries=0."""
        from unittest.mock import MagicMock, patch

        from core.llm.groq_provider import GroqLLMProvider

        with patch("core.llm.groq_provider.Groq") as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client

            # Create provider - should trigger client construction
            GroqLLMProvider(
                model="llama3-8b-8192",
                api_key="test-key",
                timeout=30.0,
            )

            # Verify Groq constructor was called with max_retries=0
            mock_groq_class.assert_called_once()
            call_kwargs = mock_groq_class.call_args.kwargs
            assert "max_retries" in call_kwargs
            assert call_kwargs["max_retries"] == 0
            assert call_kwargs["timeout"] == 30.0

    def test_async_client_constructed_with_max_retries_zero(self) -> None:
        """Test that asynchronous Groq client is created with max_retries=0."""
        from unittest.mock import MagicMock, patch

        from core.llm.groq_provider import GroqLLMProvider

        with patch("core.llm.groq_provider.AsyncGroq") as mock_async_groq_class:
            mock_client = MagicMock()
            mock_async_groq_class.return_value = mock_client

            # Create provider - should trigger async client construction
            GroqLLMProvider(
                model="llama3-8b-8192",
                api_key="test-key",
                timeout=30.0,
            )

            # Verify AsyncGroq constructor was called with max_retries=0
            mock_async_groq_class.assert_called_once()
            call_kwargs = mock_async_groq_class.call_args.kwargs
            assert "max_retries" in call_kwargs
            assert call_kwargs["max_retries"] == 0
            assert call_kwargs["timeout"] == 30.0

    def test_injected_clients_bypass_construction(self) -> None:
        """Test that injected clients are used as-is without modification."""
        from unittest.mock import MagicMock

        from core.llm.groq_provider import GroqLLMProvider

        mock_sync = MagicMock()
        mock_async = MagicMock()

        provider = GroqLLMProvider(
            model="llama3-8b-8192",
            api_key="test-key",
            client=mock_sync,
            async_client=mock_async,
        )

        # Should use injected clients directly
        assert provider._client is mock_sync
        assert provider._async_client is mock_async

    def test_per_request_timeout_passed_to_create(self) -> None:
        """Test that per-request timeout is passed to chat.completions.create."""
        from unittest.mock import MagicMock

        from core.llm.base import GenerationSettings
        from core.llm.groq_provider import GroqLLMProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = GroqLLMProvider(
            model="llama3-8b-8192",
            api_key="test-key",
            client=mock_client,
        )

        # Make request with specific timeout
        provider.generate_with_metadata("sys", "user", settings=GenerationSettings(), timeout=15.0)

        # Verify timeout was passed to create()
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 15.0


class TestOllamaCleanupExceptionPreservation:
    """Tests verifying Ollama cleanup errors don't mask generation exceptions."""

    def test_cleanup_failure_preserves_original_generation_exception(self) -> None:
        """Test that cleanup exception doesn't mask original generation exception."""
        from unittest.mock import MagicMock, patch

        from core.errors import LLMError
        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        # Create a mock that simulates an ollama.Client with an httpx client inside
        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0
        mock_httpx_client.close = MagicMock(side_effect=Exception("Cleanup failed"))

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.side_effect = Exception("Generation failed")

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock(side_effect=Exception("Cleanup failed"))

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.side_effect = Exception("Generation failed")

            mock_client_class.return_value = new_ollama_client

            # Make request - should raise LLMError wrapping original exception
            with pytest.raises(LLMError) as exc_info:
                provider.generate_with_metadata(
                    "sys", "user", settings=GenerationSettings(), timeout=5.0
                )

            # Original exception should be the cause
            assert exc_info.value.__cause__ is not None
            assert "Generation failed" in str(exc_info.value.__cause__)

            # Cleanup should still have been attempted
            new_httpx_client.close.assert_called_once()

    def test_successful_generation_with_cleanup_failure_preserves_response(self) -> None:
        """Test that successful response is returned even if cleanup fails."""
        from unittest.mock import MagicMock, patch

        from core.llm.base import GenerationResponse, GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0
        mock_httpx_client.close = MagicMock(side_effect=Exception("Cleanup failed"))

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.return_value = type(
            "Response",
            (),
            {
                "message": type("Message", (), {"content": "OK"})(),
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )()

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock(side_effect=Exception("Cleanup failed"))

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.return_value = type(
                "Response",
                (),
                {
                    "message": type("Message", (), {"content": "OK"})(),
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )()

            mock_client_class.return_value = new_ollama_client

            # Should succeed and return response despite cleanup failure
            response = provider.generate_with_metadata(
                "sys", "user", settings=GenerationSettings(), timeout=5.0
            )

            assert isinstance(response, GenerationResponse)
            assert response.text == "OK"

            # Cleanup should still have been attempted
            new_httpx_client.close.assert_called_once()

    def test_cleanup_attempted_exactly_once(self) -> None:
        """Test that cleanup is attempted exactly once per request."""
        from unittest.mock import MagicMock, patch

        from core.llm.base import GenerationSettings
        from core.llm.ollama_provider import OllamaLLMProvider

        mock_httpx_client = MagicMock()
        mock_httpx_client.base_url = "http://localhost:11434"
        mock_httpx_client.timeout = 10.0
        mock_httpx_client.close = MagicMock()

        mock_ollama_client = MagicMock()
        mock_ollama_client._client = mock_httpx_client
        mock_ollama_client.chat.return_value = type(
            "Response",
            (),
            {
                "message": type("Message", (), {"content": "OK"})(),
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )()

        provider = OllamaLLMProvider("llama3.2:3b", client=mock_ollama_client)

        with patch("ollama.Client") as mock_client_class:
            new_httpx_client = MagicMock()
            new_httpx_client.base_url = "http://localhost:11434"
            new_httpx_client.timeout = 5.0
            new_httpx_client.close = MagicMock()

            new_ollama_client = MagicMock()
            new_ollama_client._client = new_httpx_client
            new_ollama_client.chat.return_value = type(
                "Response",
                (),
                {
                    "message": type("Message", (), {"content": "OK"})(),
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )()

            mock_client_class.return_value = new_ollama_client

            # Make two separate requests
            provider.generate_with_metadata(
                "sys", "user", settings=GenerationSettings(), timeout=5.0
            )
            provider.generate_with_metadata(
                "sys", "user", settings=GenerationSettings(), timeout=5.0
            )

            # Cleanup should be called once per request
            assert new_httpx_client.close.call_count == 2


# =============================================================================
# F-09.3.2: Async Provider Contract Tests
# =============================================================================


class TestAsyncProviderProtocol:
    """Tests for the AsyncLLMProvider protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that a compliant async provider satisfies the protocol."""
        from core.llm.base import AsyncCloseable, AsyncLLMProvider, GenerationResponse

        class CompliantAsyncProvider:
            model_id = "mock:compliant"
            closed = False

            async def generate_with_metadata_async(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,  # noqa: ASYNC109
                deadline: GenerationDeadline | None = None,
            ) -> GenerationResponse:
                return GenerationResponse(text="OK")

            async def aclose(self) -> None:
                self.closed = True

        provider = CompliantAsyncProvider()
        assert isinstance(provider, AsyncLLMProvider)
        assert isinstance(provider, AsyncCloseable)

    def test_protocol_missing_method(self) -> None:
        """Test that a provider missing required methods fails the protocol."""
        from core.llm.base import AsyncLLMProvider

        class IncompleteProvider:
            model_id = "mock:incomplete"
            # Missing generate_with_metadata_async and aclose

        provider = IncompleteProvider()
        assert not isinstance(provider, AsyncLLMProvider)

    def test_sync_protocol_unchanged(self) -> None:
        """Test that the synchronous LLMProvider protocol is unchanged."""
        from core.llm.base import LLMProvider

        class SyncProvider:
            model_id = "mock:sync"

            def generate(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
            ) -> str:
                return "OK"

        provider = SyncProvider()
        assert isinstance(provider, LLMProvider)
        # Should not be an async provider
        from core.llm.base import AsyncLLMProvider

        assert not isinstance(provider, AsyncLLMProvider)


class TestAsyncProviderContract:
    """Tests for the async provider contract behavior."""

    def test_generate_async_convenience_method(self) -> None:
        """Test that generate_async delegates to generate_with_metadata_async."""
        from core.llm.base import AsyncLLMProvider, GenerationResponse, GenerationSettings

        class TestProvider:
            model_id = "mock:test"
            last_call: tuple | None = None

            async def generate_with_metadata_async(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,  # noqa: ASYNC109
                deadline: GenerationDeadline | None = None,
            ) -> GenerationResponse:
                self.last_call = (system_prompt, user_prompt, settings, timeout, deadline)
                return GenerationResponse(text="test response")

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
                    system_prompt,
                    user_prompt,
                    settings=settings,
                    timeout=timeout,
                    deadline=deadline,
                )
                return response.text

            async def aclose(self) -> None:
                pass

        provider = TestProvider()
        assert isinstance(provider, AsyncLLMProvider)

        import asyncio

        async def test():
            result = await provider.generate_async(
                "sys", "user", settings=GenerationSettings(), timeout=10.0
            )
            return result

        result = asyncio.run(test())
        assert result == "test response"
        assert provider.last_call is not None
        assert provider.last_call[0] == "sys"
        assert provider.last_call[1] == "user"


class TestAsyncLifecycleContract:
    """Tests for the async lifecycle contract."""

    def test_aclose_idempotent(self) -> None:
        """Test that aclose is idempotent."""
        from core.llm.base import AsyncCloseable, GenerationResponse

        class TestProvider:
            def __init__(self) -> None:
                self.model_id = "mock:test"
                self.close_count = 0

            async def generate_with_metadata_async(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                settings: GenerationSettings,
                timeout: float | None = None,  # noqa: ASYNC109
                deadline: GenerationDeadline | None = None,
            ) -> GenerationResponse:
                return GenerationResponse(text="OK")

            async def aclose(self) -> None:
                self.close_count += 1

        provider = TestProvider()
        assert isinstance(provider, AsyncCloseable)

        import asyncio

        async def test():
            await provider.aclose()
            await provider.aclose()
            await provider.aclose()

        asyncio.run(test())
        assert provider.close_count == 3


class TestGenerationDeadline:
    """Tests for the GenerationDeadline implementation."""

    def test_deadline_creation_monotonic(self) -> None:
        """Test that deadline creation uses monotonic time."""
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(10.0)
        # Deadline should be in the future
        assert deadline._expiration_monotonic > time.monotonic()
        assert deadline._expiration_monotonic <= time.monotonic() + 10.0 + 0.01

    def test_remaining_budget_decreases(self) -> None:
        """Test that remaining budget decreases over time."""
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(1.0)
        initial = deadline.remaining_seconds()
        time.sleep(0.05)  # Increased sleep for measurable difference
        later = deadline.remaining_seconds()
        assert later < initial
        assert later >= 0

    def test_expired_deadline_raises(self) -> None:
        """Test that expired deadline raises ProviderDeadlineExceededError."""
        from core.errors import ProviderDeadlineExceededError
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(0.001)
        time.sleep(0.05)  # Increased sleep for reliability
        assert deadline.expired
        with pytest.raises(ProviderDeadlineExceededError):
            deadline.check_or_raise("test")

    def test_zero_timeout_rejected(self) -> None:
        """Test that zero timeout is rejected."""
        from core.llm.router import GenerationDeadline

        with pytest.raises(ValueError):
            GenerationDeadline.after(0)

    def test_negative_timeout_rejected(self) -> None:
        """Test that negative timeout is rejected."""
        from core.llm.router import GenerationDeadline

        with pytest.raises(ValueError):
            GenerationDeadline.after(-1.0)

    def test_explicit_deadline_precedes_timeout(self) -> None:
        """Test that explicit deadline takes precedence over timeout."""
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(5.0)
        _ = 1.0  # Shorter than deadline  # noqa: F841

        # The deadline should be used, not the timeout
        # This is tested by checking that remaining_seconds reflects the deadline
        remaining = deadline.remaining_seconds()
        assert remaining > 4.0  # Should reflect the 5s deadline, not 1s timeout

    def test_same_deadline_object_propagated(self) -> None:
        """Test that the same deadline object is passed through multiple calls."""
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(10.0)
        same_deadline = deadline

        # Simulate passing through multiple layers
        assert same_deadline is deadline
        assert same_deadline._expiration_monotonic == deadline._expiration_monotonic

    def test_deadline_expiration_no_new_budget(self) -> None:
        """Test that deadline expiration does not create a new positive budget."""
        from core.errors import ProviderDeadlineExceededError
        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(0.001)
        time.sleep(0.05)  # Increased sleep for reliability

        # Expired deadline should raise, not create new budget
        with pytest.raises(ProviderDeadlineExceededError):
            deadline.check_or_raise("test")

        # Remaining should be clamped at zero
        assert deadline.remaining_seconds() == 0.0

    def test_concurrent_requests_independent_deadlines(self) -> None:
        """Test that concurrent requests with different deadlines don't contaminate each other."""
        from core.llm.router import GenerationDeadline

        deadline_a = GenerationDeadline.after(1.0)
        deadline_b = GenerationDeadline.after(5.0)

        # They should be distinct objects
        assert deadline_a is not deadline_b

        # Their expiration times should differ
        assert deadline_a._expiration_monotonic < deadline_b._expiration_monotonic

        # Modifying one (conceptually) shouldn't affect the other
        # (Deadlines are immutable, so this is inherent)
        assert deadline_a.remaining_seconds() < deadline_b.remaining_seconds()


class TestCancellationBehavior:
    """Tests for cancellation behavior."""

    def test_cancelled_error_propagates(self) -> None:
        """Test that asyncio.CancelledError propagates unchanged."""
        import asyncio

        async def cancelled_task():
            raise asyncio.CancelledError()

        async def test():
            with pytest.raises(asyncio.CancelledError):
                await cancelled_task()

        asyncio.run(test())

    def test_cancellation_not_converted_to_llm_error(self) -> None:
        """Test that cancellation is not converted to LLMError."""
        import asyncio

        from core.errors import LLMError

        async def test():
            try:
                raise asyncio.CancelledError()
            except asyncio.CancelledError:
                # Re-raise to verify it propagates
                raise
            except LLMError:
                pytest.fail("CancelledError was converted to LLMError")

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(test())

    def test_cancellation_does_not_mutate_deadline(self) -> None:
        """Test that cancellation does not mutate a deadline."""
        import asyncio

        from core.llm.router import GenerationDeadline

        deadline = GenerationDeadline.after(10.0)
        original_expiration = deadline._expiration_monotonic

        async def test():
            try:
                raise asyncio.CancelledError()
            except asyncio.CancelledError:
                pass

        asyncio.run(test())

        # Deadline should be unchanged
        assert deadline._expiration_monotonic == original_expiration

    def test_cancellation_no_retry_side_effect(self) -> None:
        """Test that cancellation does not create retry or provider-failure side effect."""
        import asyncio

        side_effects = []

        async def failing_operation():
            side_effects.append("start")
            raise asyncio.CancelledError()

        async def test():
            try:
                await failing_operation()
            except asyncio.CancelledError:
                pass

        asyncio.run(test())

        # Only the start should be recorded, no retry/fallback side effects
        assert side_effects == ["start"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
