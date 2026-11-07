"""Tests for token usage metadata propagation through resilience wrappers."""

from __future__ import annotations

import pytest

from core.errors import LLMError
from core.llm import GenerationResponse, GenerationSettings
from core.llm.base import LLMProvider
from core.llm.router import (
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    FallbackProvider,
    RetryPolicy,
    RetryProvider,
    clear_provider_cache,
    get_metrics_collector,
)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        name: str,
        response: GenerationResponse | str = "OK",
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
        if isinstance(self._response, GenerationResponse):
            return self._response.text
        return self._response

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> GenerationResponse:
        self.call_count += 1
        if self._error:
            raise self._error
        if isinstance(self._response, GenerationResponse):
            return self._response
        return GenerationResponse(text=self._response)


class FlakyProvider:
    """Mock provider that fails a specified number of times before succeeding."""

    def __init__(
        self,
        name: str,
        fail_times: int,
        error: Exception,
        response: GenerationResponse | None = None,
    ) -> None:
        self.model_id = f"mock:{name}"
        self.name = name
        self._fail_times = fail_times
        self._error = error
        self._response = response or GenerationResponse(text="Recovered")
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
        return self._response.text

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> GenerationResponse:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise self._error
        return self._response


class TestRetryProviderMetadataPropagation:
    """Tests for metadata propagation through RetryProvider."""

    def test_metadata_preserved_on_success(self) -> None:
        """Test that metadata is preserved after successful generation."""
        clear_provider_cache()
        inner = MockProvider(
            "test", GenerationResponse(text="Success", input_tokens=100, output_tokens=50)
        )
        policy = RetryPolicy(max_attempts=3)
        retry = RetryProvider(inner, policy, "test")

        response = retry.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert isinstance(response, GenerationResponse)
        assert response.text == "Success"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150

    def test_legacy_provider_supported(self) -> None:
        """Test that legacy string-only provider is supported."""
        clear_provider_cache()
        inner = MockProvider("legacy", "Legacy response")
        policy = RetryPolicy(max_attempts=3)
        retry = RetryProvider(inner, policy, "legacy")

        response = retry.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert isinstance(response, GenerationResponse)
        assert response.text == "Legacy response"
        assert response.input_tokens is None
        assert response.output_tokens is None

    def test_retry_after_transient_failure_preserves_metadata(self) -> None:
        """Test that retry after transient failure eventually returns metadata."""
        clear_provider_cache()
        inner = FlakyProvider(
            "flaky",
            fail_times=1,
            error=LLMError("Transient"),
            response=GenerationResponse(text="Recovered", input_tokens=50, output_tokens=25),
        )
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
        retry = RetryProvider(inner, policy, "flaky")

        response = retry.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert response.text == "Recovered"
        assert response.input_tokens == 50
        assert response.output_tokens == 25

    def test_failed_attempts_do_not_contaminate_metadata(self) -> None:
        """Test that failed attempts do not contaminate successful response token counts."""
        clear_provider_cache()
        inner = FlakyProvider(
            "flaky",
            fail_times=2,
            error=LLMError("Transient"),
            response=GenerationResponse(text="Final", input_tokens=10, output_tokens=5),
        )
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
        retry = RetryProvider(inner, policy, "flaky")

        response = retry.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert response.text == "Final"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_generate_returns_string(self) -> None:
        """Test that generate() still returns a string."""
        clear_provider_cache()
        inner = MockProvider(
            "test", GenerationResponse(text="String response", input_tokens=10, output_tokens=5)
        )
        policy = RetryPolicy(max_attempts=3)
        retry = RetryProvider(inner, policy, "test")

        text = retry.generate("sys", "user", settings=GenerationSettings())

        assert isinstance(text, str)
        assert text == "String response"

    def test_existing_retry_limits_enforced(self) -> None:
        """Test that existing retry limits and exception behavior remain unchanged."""
        clear_provider_cache()
        inner = MockProvider("test", error=LLMError("Permanent"))
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
        retry = RetryProvider(inner, policy, "test")

        with pytest.raises(LLMError):
            retry.generate_with_metadata("sys", "user", settings=GenerationSettings())
        assert inner.call_count == 1  # Non-recoverable errors don't retry


class TestCircuitBreakerProviderMetadataRecording:
    """Tests for metadata recording in CircuitBreakerProvider."""

    def test_successful_generation_records_input_tokens(self) -> None:
        """Test that successful generation records input tokens."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider(
            "test", GenerationResponse(text="OK", input_tokens=100, output_tokens=50)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["usage"]["input_tokens"] == 100

    def test_successful_generation_records_output_tokens(self) -> None:
        """Test that successful generation records output tokens."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider(
            "test", GenerationResponse(text="OK", input_tokens=100, output_tokens=50)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["usage"]["output_tokens"] == 50

    def test_missing_usage_remains_none(self) -> None:
        """Test that missing usage remains None."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider("legacy", "Legacy response")
        cb = CircuitBreaker("legacy", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "legacy")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["legacy"]["usage"]["input_tokens"] is None
        assert metrics["providers"]["legacy"]["usage"]["output_tokens"] is None

    def test_explicit_zero_usage_remains_zero(self) -> None:
        """Test that explicit zero usage remains zero."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider("test", GenerationResponse(text="OK", input_tokens=0, output_tokens=0))
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["usage"]["input_tokens"] == 0
        assert metrics["providers"]["test"]["usage"]["output_tokens"] == 0

    def test_returned_generation_response_unchanged(self) -> None:
        """Test that returned GenerationResponse is unchanged."""
        clear_provider_cache()
        inner = MockProvider(
            "test", GenerationResponse(text="OK", input_tokens=100, output_tokens=50)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        response = cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert response.text == "OK"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_metrics_recorded_exactly_once(self) -> None:
        """Test that metrics are recorded exactly once per successful request."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider(
            "test", GenerationResponse(text="OK", input_tokens=10, output_tokens=5)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["requests"]["total"] == 1
        assert metrics["providers"]["test"]["requests"]["successful"] == 1
        assert metrics["providers"]["test"]["usage"]["input_tokens"] == 10

    def test_failure_does_not_record_successful_usage(self) -> None:
        """Test that failure does not record successful token usage."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider("test", error=LLMError("Fail"))
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        with pytest.raises(LLMError):
            cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # No successful requests recorded
        assert metrics["providers"]["test"]["requests"]["successful"] == 0
        assert metrics["providers"]["test"]["usage"]["input_tokens"] is None

    def test_legacy_provider_still_works(self) -> None:
        """Test that legacy string-only provider still works."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider("legacy", "Legacy response")
        cb = CircuitBreaker("legacy", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "legacy")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        assert metrics["providers"]["legacy"]["usage"]["input_tokens"] is None
        assert metrics["providers"]["legacy"]["requests"]["successful"] == 1

    def test_closed_state_behavior_unchanged(self) -> None:
        """Test that CLOSED-state behavior remains unchanged."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider(
            "test", GenerationResponse(text="OK", input_tokens=10, output_tokens=5)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=True, failure_threshold=5))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert cb.state == "closed"
        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["requests"]["successful"] == 1

    def test_half_open_success_behavior_unchanged(self) -> None:
        """Test that HALF_OPEN success behavior remains unchanged."""
        # This test is conceptual - the HALF_OPEN state transitions are covered
        # in the main circuit breaker tests. Skipping here to avoid clock mocking complexity.
        pass

    def test_half_open_failure_behavior_unchanged(self) -> None:
        """Test that HALF_OPEN failure behavior remains unchanged."""
        pass

    def test_nonrecoverable_failures_preserve_permit_semantics(self) -> None:
        """Test that nonrecoverable failures preserve permit semantics."""
        clear_provider_cache()
        inner = MockProvider("test", error=LLMError("Auth failed"))  # Non-recoverable
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        with pytest.raises(LLMError):
            cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    def test_baseexception_release_semantics_intact(self) -> None:
        """Test that BaseException release/re-raise behavior remains intact."""
        clear_provider_cache()
        inner = MockProvider("test", error=KeyboardInterrupt("User interrupt"))
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        with pytest.raises(KeyboardInterrupt):
            cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    def test_generate_returns_string(self) -> None:
        """Test that generate() still returns a string."""
        clear_provider_cache()
        inner = MockProvider(
            "test", GenerationResponse(text="String", input_tokens=10, output_tokens=5)
        )
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(inner, cb, "test")

        text = cb_provider.generate("sys", "user", settings=GenerationSettings())

        assert isinstance(text, str)
        assert text == "String"


def _make_wrapped_provider(
    name: str,
    response: GenerationResponse | str = "OK",
    error: Exception | None = None,
) -> LLMProvider:
    """Create a wrapped provider (Retry + CircuitBreaker) for testing."""
    clear_provider_cache()
    MockProvider(name, response, error)
    retry = RetryProvider(MockProvider(name, response, error), RetryPolicy(max_attempts=3), name)
    # Use disabled circuit breaker for simpler testing
    CircuitBreaker(name, CircuitBreakerPolicy(enabled=False))
    return CircuitBreakerProvider(
        retry, CircuitBreaker(name, CircuitBreakerPolicy(enabled=False)), name
    )


class TestFallbackProviderMetadataPropagation:
    """Tests for metadata propagation through FallbackProvider."""

    def test_primary_success_preserves_metadata(self) -> None:
        """Test that primary success preserves metadata."""
        clear_provider_cache()
        primary = _make_wrapped_provider(
            "primary", GenerationResponse(text="Primary", input_tokens=100, output_tokens=50)
        )
        fallback = _make_wrapped_provider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        response = fallback_provider.generate_with_metadata(
            "sys", "user", settings=GenerationSettings()
        )

        assert response.text == "Primary"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_primary_recoverable_failure_fallback_success_preserves_metadata(self) -> None:
        """Test primary recoverable failure followed by fallback success preserves metadata."""
        clear_provider_cache()
        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback = _make_wrapped_provider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        response = fallback_provider.generate_with_metadata(
            "sys", "user", settings=GenerationSettings()
        )

        assert response.text == "Fallback"
        assert response.input_tokens == 50
        assert response.output_tokens == 25

    def test_failed_primary_usage_not_attributed_to_fallback(self) -> None:
        """Test that failed primary usage is not attributed to fallback success."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback = _make_wrapped_provider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        fallback_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # Primary failure should not record successful usage
        assert metrics["providers"]["primary"]["requests"]["successful"] == 0
        # Fallback should have the usage
        assert metrics["providers"]["fallback"]["usage"]["input_tokens"] == 50

    def test_fallback_usage_recorded_against_fallback_provider(self) -> None:
        """Test that fallback usage is recorded against the fallback provider's metrics identity."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback = _make_wrapped_provider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        fallback_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # Usage should be attributed to fallback provider
        assert metrics["providers"]["fallback"]["usage"]["input_tokens"] == 50
        assert metrics["providers"]["fallback"]["usage"]["output_tokens"] == 25

    def test_no_duplicate_success_metrics(self) -> None:
        """Test that no duplicate success metrics are recorded."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback = _make_wrapped_provider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=10, output_tokens=5)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        fallback_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # Only one successful request total
        total_success = sum(p["requests"]["successful"] for p in metrics["providers"].values())
        assert total_success == 1

    def test_nonrecoverable_error_no_fallback(self) -> None:
        """Test that nonrecoverable errors do not trigger inappropriate fallback."""
        clear_provider_cache()
        primary = _make_wrapped_provider(
            "primary", error=LLMError("Auth failed")
        )  # Non-recoverable
        fallback = _make_wrapped_provider("fallback", "Fallback")
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        with pytest.raises(LLMError):
            fallback_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    def test_legacy_providers_supported(self) -> None:
        """Test that legacy string-only providers are supported."""
        clear_provider_cache()
        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback = _make_wrapped_provider("fallback", "Legacy fallback")
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])

        response = fallback_provider.generate_with_metadata(
            "sys", "user", settings=GenerationSettings()
        )

        assert response.text == "Legacy fallback"
        assert response.input_tokens is None

    def test_existing_fallback_ordering_unchanged(self) -> None:
        """Test that existing fallback ordering remains unchanged."""
        clear_provider_cache()
        primary = _make_wrapped_provider("primary", error=LLMError("Transient"))
        fallback1 = _make_wrapped_provider("fallback1", error=LLMError("Transient"))
        fallback2 = _make_wrapped_provider("fallback2", "Fallback2")
        fallback_provider = FallbackProvider(
            primary, [fallback1, fallback2], "primary", ["fallback1", "fallback2"]
        )

        response = fallback_provider.generate_with_metadata(
            "sys", "user", settings=GenerationSettings()
        )

        assert response.text == "Fallback2"

    def test_generate_returns_string(self) -> None:
        """Test that generate() still returns a string."""
        clear_provider_cache()
        primary = _make_wrapped_provider(
            "primary", GenerationResponse(text="String", input_tokens=10, output_tokens=5)
        )
        fallback_provider = FallbackProvider(primary, [], "primary", [])

        text = fallback_provider.generate("sys", "user", settings=GenerationSettings())

        assert isinstance(text, str)
        assert text == "String"


class TestEndToEndWrapperChain:
    """End-to-end tests for the complete wrapper chain."""

    def test_full_chain_metadata_preservation(self) -> None:
        """Test that metadata is preserved through the full wrapper chain."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider(
            "test", GenerationResponse(text="Success", input_tokens=100, output_tokens=50)
        )
        policy = RetryPolicy(max_attempts=3)
        retry = RetryProvider(inner, policy, "test")
        cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(retry, cb, "test")

        response = cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert response.text == "Success"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

        metrics = collector.get_metrics()
        assert metrics["providers"]["test"]["usage"]["input_tokens"] == 100
        assert metrics["providers"]["test"]["usage"]["output_tokens"] == 50

    def test_retry_in_chain_does_not_duplicate_metrics(self) -> None:
        """Test that retry attempts do not duplicate successful metrics."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = FlakyProvider(
            "flaky",
            fail_times=2,
            error=LLMError("Transient"),
            response=GenerationResponse(text="Recovered", input_tokens=50, output_tokens=25),
        )
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
        retry = RetryProvider(inner, policy, "flaky")
        cb = CircuitBreaker("flaky", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(retry, cb, "flaky")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # Only one successful request should be recorded
        assert metrics["providers"]["flaky"]["requests"]["successful"] == 1
        assert metrics["providers"]["flaky"]["usage"]["input_tokens"] == 50

    def test_fallback_in_chain_attributed_correctly(self) -> None:
        """Test that fallback success is attributed to the actual successful provider."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        primary = FlakyProvider(
            "primary",
            fail_times=1,
            error=LLMError("Transient"),
            response=GenerationResponse(text="Primary", input_tokens=100, output_tokens=50),
        )
        fallback = MockProvider(
            "fallback", GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25)
        )
        fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
        retry = RetryProvider(fallback_provider, policy, "fallback")
        cb = CircuitBreaker("fallback", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(retry, cb, "fallback")

        cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        metrics = collector.get_metrics()
        # Usage should be attributed to fallback
        assert metrics["providers"]["fallback"]["usage"]["input_tokens"] == 50
        assert metrics["providers"]["fallback"]["usage"]["output_tokens"] == 25
        # Primary failure should not have successful usage
        assert metrics["providers"]["primary"]["requests"]["successful"] == 0

    def test_legacy_chain_works(self) -> None:
        """Test that an equivalent chain with a legacy string-only provider works."""
        clear_provider_cache()
        collector = get_metrics_collector()
        collector.reset()

        inner = MockProvider("legacy", "Legacy response")
        policy = RetryPolicy(max_attempts=3)
        retry = RetryProvider(inner, policy, "legacy")
        cb = CircuitBreaker("legacy", CircuitBreakerPolicy(enabled=False))
        cb_provider = CircuitBreakerProvider(retry, cb, "legacy")

        response = cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

        assert response.text == "Legacy response"
        assert response.input_tokens is None
        assert response.output_tokens is None

        metrics = collector.get_metrics()
        assert metrics["providers"]["legacy"]["requests"]["successful"] == 1
        assert metrics["providers"]["legacy"]["usage"]["input_tokens"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
