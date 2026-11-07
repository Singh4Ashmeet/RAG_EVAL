from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from apps.api import main
from core.errors import LLMError
from core.llm import GenerationResponse, GenerationSettings
from core.llm.metrics import (
    DEFAULT_MODEL_PRICING,
    ProviderMetrics,
    ProviderMetricsCollector,
    get_metrics_collector,
)
from core.llm.router import (
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitBreakerProvider,
    FallbackProvider,
    RetryPolicy,
    RetryProvider,
    clear_provider_cache,
    get_circuit_breaker,
)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        name: str,
        response: str | GenerationResponse = "OK",
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


def test_metrics_request_counts() -> None:
    """Test calculation of total requests, success, failure, and rates."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    for _ in range(8):
        metrics.record_success(latency_seconds=0.15)
    for _ in range(2):
        metrics.record_failure(latency_seconds=0.05, error=Exception("Fail"))

    data = metrics.to_dict()
    assert data["requests"]["total"] == 10
    assert data["requests"]["successful"] == 8
    assert data["requests"]["failed"] == 2
    assert data["requests"]["error_rate"] == 0.2
    assert data["requests"]["success_rate"] == 0.8


def test_metrics_latency_percentiles() -> None:
    """Test latency statistics: avg, min, max, P50, and P95."""
    metrics = ProviderMetrics("groq")

    # Insert 100 samples from 0.001s (1ms) to 0.100s (100ms)
    for i in range(1, 101):
        metrics.record_success(latency_seconds=i / 1000.0)

    stats = metrics.get_latency_stats()
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0
    # Average of 1..100 is 50.5
    assert stats["avg_ms"] == 50.5
    # P50 (median) around 51ms
    assert 50.0 <= stats["p50_ms"] <= 52.0
    # P95 around 96ms
    assert 94.0 <= stats["p95_ms"] <= 97.0


def test_metrics_reliability_counters() -> None:
    """Test tracking of reliability events: retries, exhaustions, fallbacks, circuit state."""
    metrics = ProviderMetrics("ollama")

    metrics.record_retry()
    metrics.record_retry()
    metrics.record_retry_exhaustion()
    metrics.record_fallback()
    metrics.record_circuit_opening()
    metrics.record_circuit_rejection()
    metrics.record_half_open_probe()
    metrics.record_failure(0.1, TimeoutError("Operation timed out"))

    data = metrics.to_dict()
    rel = data["reliability"]
    assert rel["retries"] == 2
    assert rel["retry_exhaustions"] == 1
    assert rel["fallbacks"] == 1
    assert rel["circuit_openings"] == 1
    assert rel["circuit_rejections"] == 1
    assert rel["half_open_probes"] == 1
    assert rel["timeouts"] == 1


def test_metrics_token_usage_recorded_when_present() -> None:
    """Test token usage tracking when real provider metadata is provided."""
    metrics = ProviderMetrics("gemini")

    metrics.record_success(latency_seconds=0.2, input_tokens=120, output_tokens=80)
    metrics.record_success(latency_seconds=0.3, input_tokens=50, output_tokens=30)

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 170
    assert data["usage"]["output_tokens"] == 110
    assert data["usage"]["total_tokens"] == 280


def test_metrics_token_usage_unknown_when_absent() -> None:
    """Test token usage is None / unknown when metadata is absent (never fabricated)."""
    metrics = ProviderMetrics("ollama")

    metrics.record_success(latency_seconds=0.2)

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] is None
    assert data["usage"]["output_tokens"] is None
    assert data["usage"]["total_tokens"] is None
    assert data["cost"]["estimated_cost_usd"] is None


def test_metrics_cost_estimation_with_pricing() -> None:
    """Test estimated cost calculation using pricing map ($ per 1M tokens)."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    # Record 1,000,000 input tokens and 1,000,000 output tokens
    # Gemini 2.5 flash: input $0.075 / 1M, output $0.30 / 1M -> total $0.375
    metrics.record_success(
        latency_seconds=1.0,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.375
    assert data["cost"]["is_estimate"] is True


def test_metrics_cost_estimation_unknown_without_pricing() -> None:
    """Test estimated cost is None when pricing is not configured for the model."""
    metrics = ProviderMetrics("custom_provider", model_id="custom:unknown-model")
    metrics.record_success(latency_seconds=0.1, input_tokens=1000, output_tokens=500)

    # Empty pricing map
    data = metrics.to_dict(pricing_map={})
    assert data["cost"]["estimated_cost_usd"] is None


def test_metrics_collector_thread_safety() -> None:
    """Test thread-safety of collector under concurrent multi-threaded writes."""
    collector = ProviderMetricsCollector()

    def worker():
        for _ in range(50):
            collector.record_success("groq", model_id="groq:qwen", latency=0.05)
            collector.record_failure("groq", latency=0.02, error=Exception("err"))
            collector.record_retry("groq")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m = collector.get_metrics()["providers"]["groq"]
    assert m["requests"]["total"] == 1000
    assert m["requests"]["successful"] == 500
    assert m["requests"]["failed"] == 500
    assert m["reliability"]["retries"] == 500


def test_metrics_collector_bounded_memory() -> None:
    """Test that latencies deque is strictly bounded to max_latency_samples."""
    metrics = ProviderMetrics("ollama", max_latency_samples=100)

    for i in range(250):
        metrics.record_success(latency_seconds=float(i))

    assert len(metrics._latencies) == 100


def test_metrics_collector_reset() -> None:
    """Test reset clears all metrics and fallback history."""
    collector = ProviderMetricsCollector()
    collector.record_success("gemini", latency=0.1)
    collector.record_fallback("gemini", "groq", "test reason")

    metrics_before = collector.get_metrics()
    assert len(metrics_before["providers"]) == 1
    assert len(metrics_before["fallback_events"]) == 1

    collector.reset()
    metrics_after = collector.get_metrics()
    assert len(metrics_after["providers"]) == 0
    assert len(metrics_after["fallback_events"]) == 0


def test_health_status_all_healthy() -> None:
    """Test system health is 'healthy' when all circuit breakers are closed."""
    clear_provider_cache()
    collector = get_metrics_collector()
    health = collector.get_health()

    assert health["status"] == "healthy"
    for p in health["providers"].values():
        assert p["state"] == "closed"
        assert p["healthy"] is True


def test_health_status_degraded() -> None:
    """Test health is 'degraded' when one provider is open while others closed."""
    clear_provider_cache()
    cb_gemini = get_circuit_breaker("gemini")
    # Force gemini open
    for _ in range(cb_gemini._policy.failure_threshold):
        cb_gemini.record_failure(Exception("Service unavailable"))

    collector = get_metrics_collector()
    health = collector.get_health()

    assert health["status"] == "degraded"
    assert health["providers"]["gemini"]["healthy"] is False
    assert health["providers"]["gemini"]["state"] == "open"
    # Other providers remain healthy
    assert health["providers"]["ollama"]["healthy"] is True


def test_health_status_unhealthy() -> None:
    """Test system health is 'unhealthy' when all configured providers are open."""
    clear_provider_cache()
    for name in ["ollama", "gemini", "groq"]:
        cb = get_circuit_breaker(name)
        for _ in range(cb._policy.failure_threshold):
            cb.record_failure(Exception("Service unavailable"))

    collector = get_metrics_collector()
    health = collector.get_health()

    assert health["status"] == "unhealthy"
    for p in health["providers"].values():
        assert p["healthy"] is False


def test_api_endpoints_metrics_and_health_unauthenticated() -> None:
    """Test GET /api/v1/llm/metrics and GET /api/v1/llm/health
    endpoints reject unauthenticated requests.
    """
    clear_provider_cache()
    main.app.dependency_overrides.clear()
    client = TestClient(main.app)

    # Health endpoint - unauthenticated
    health_resp = client.get("/api/v1/llm/health")
    assert health_resp.status_code == 401

    # Metrics endpoint - unauthenticated
    metrics_resp = client.get("/api/v1/llm/metrics")
    assert metrics_resp.status_code == 401


def test_api_endpoints_metrics_and_health_authenticated() -> None:
    """Test GET /api/v1/llm/metrics and GET /api/v1/llm/health
    endpoints work for authenticated users.
    """
    clear_provider_cache()
    main.app.dependency_overrides.clear()

    # Setup authenticated test client
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    main.Base.metadata.create_all(engine)
    account = main.Account(
        id="test-workspace",
        github_id="test-user",
        login="researcher",
        email="researcher@example.test",
    )
    with Session(engine, expire_on_commit=False) as db:
        db.add(account)
        db.commit()

    def session_override():
        with Session(engine, expire_on_commit=False) as db:
            yield db

    def account_override():
        return account

    main.app.dependency_overrides[main.db_session] = session_override
    main.app.dependency_overrides[main.current_account] = account_override

    try:
        with TestClient(main.app) as client:
            # Health endpoint - authenticated
            health_resp = client.get("/api/v1/llm/health")
            assert health_resp.status_code == 200
            health_json = health_resp.json()
            assert "status" in health_json
            assert "providers" in health_json
            assert "gemini" in health_json["providers"]

            # Metrics endpoint - authenticated
            metrics_resp = client.get("/api/v1/llm/metrics")
            assert metrics_resp.status_code == 200
            metrics_json = metrics_resp.json()
            assert "providers" in metrics_json
            assert "fallback_events" in metrics_json

            # Security check: ensure no secrets/keys are leaked in responses
            dumped = str(health_json) + str(metrics_json)
            assert "api_key" not in dumped.lower()
            assert "authorization" not in dumped.lower()
            assert "secret" not in dumped.lower()
    finally:
        main.app.dependency_overrides.clear()
        engine.dispose()


# =============================================================================
# Phase 4: Token Aggregation & Cost Calculation Tests
# =============================================================================


def test_token_aggregation_input_only() -> None:
    """Test that input-only token usage aggregates correctly without fabricating output."""
    metrics = ProviderMetrics("test")

    # First request: only input tokens
    metrics.record_success(latency_seconds=0.1, input_tokens=100)
    # Second request: only input tokens
    metrics.record_success(latency_seconds=0.2, input_tokens=50)

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 150
    assert data["usage"]["output_tokens"] is None
    assert data["usage"]["total_tokens"] is None


def test_token_aggregation_output_only() -> None:
    """Test that output-only token usage aggregates correctly without fabricating input."""
    metrics = ProviderMetrics("test")

    # First request: only output tokens
    metrics.record_success(latency_seconds=0.1, output_tokens=80)
    # Second request: only output tokens
    metrics.record_success(latency_seconds=0.2, output_tokens=20)

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] is None
    assert data["usage"]["output_tokens"] == 100
    assert data["usage"]["total_tokens"] is None


def test_token_aggregation_partial_then_complete() -> None:
    """Test that partial usage followed by complete usage computes total correctly."""
    metrics = ProviderMetrics("test")

    # First request: only input tokens
    metrics.record_success(latency_seconds=0.1, input_tokens=100)
    # Second request: complete usage
    metrics.record_success(latency_seconds=0.2, input_tokens=50, output_tokens=30)

    data = metrics.to_dict()
    # Input tokens should aggregate
    assert data["usage"]["input_tokens"] == 150
    # Output tokens only from second request
    assert data["usage"]["output_tokens"] == 30
    # Total only from complete request (50+30=80)
    assert data["usage"]["total_tokens"] == 80


def test_token_aggregation_explicit_total_precedence() -> None:
    """Test that explicit total_tokens takes precedence over computed sum."""
    # Explicit total should be used even when different from input+output sum
    metrics = ProviderMetrics("test")

    # Provide explicit total that differs from input+output
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=200,  # Different from 100+50=150
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["output_tokens"] == 50
    assert data["usage"]["total_tokens"] == 200  # Explicit total preserved

    # Another request with explicit total
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=200,
        output_tokens=100,
        total_tokens=500,  # Different from 200+100=300
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 300  # 100+200
    assert data["usage"]["output_tokens"] == 150  # 50+100
    assert data["usage"]["total_tokens"] == 700  # 200+500 (explicit totals)


def test_token_aggregation_multiple_requests() -> None:
    """Test that multiple successful requests aggregate correctly."""
    metrics = ProviderMetrics("test")

    for i in range(5):
        metrics.record_success(
            latency_seconds=0.1,
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
        )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 1500  # 100+200+300+400+500
    assert data["usage"]["output_tokens"] == 750  # 50+100+150+200+250
    assert data["usage"]["total_tokens"] == 2250  # 150+300+450+600+750


def test_token_aggregation_provider_model_attribution() -> None:
    """Test that provider/model attribution remains correct."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    metrics.record_success(latency_seconds=0.1, input_tokens=100, output_tokens=50)

    data = metrics.to_dict()
    assert data["provider"] == "gemini"
    assert data["model_id"] == "gemini:gemini-2.5-flash"
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["output_tokens"] == 50
    assert data["usage"]["total_tokens"] == 150


def test_cost_calculation_exact_known_model() -> None:
    """Test exact cost calculation for known model with known rates."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    # 1M input @ $0.075/M = $0.075, 1M output @ $0.30/M = $0.30
    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.375


def test_cost_calculation_input_output_rates_independent() -> None:
    """Test that input and output rates are applied independently."""
    metrics = ProviderMetrics("test", model_id="custom:model")

    # Set custom pricing
    custom_pricing = {"custom:model": {"input": 10.0, "output": 20.0}}

    # 500K input @ $10/M = $5, 500K output @ $20/M = $10
    metrics.record_success(latency_seconds=1.0, input_tokens=500_000, output_tokens=500_000)

    data = metrics.to_dict(pricing_map=custom_pricing)
    assert data["cost"]["estimated_cost_usd"] == 15.0


def test_cost_calculation_pricing_unit_conversion() -> None:
    """Test correct pricing-unit conversion (per 1M tokens)."""
    metrics = ProviderMetrics("test", model_id="test:model")

    # Set pricing at $1.00 per 1M tokens for both
    custom_pricing = {"test:model": {"input": 1.0, "output": 1.0}}

    # 500K tokens each = 0.5M each, so $0.50 + $0.50 = $1.00
    metrics.record_success(latency_seconds=1.0, input_tokens=500_000, output_tokens=500_000)

    data = metrics.to_dict(pricing_map=custom_pricing)
    assert data["cost"]["estimated_cost_usd"] == 1.0


def test_cost_calculation_unknown_model_returns_none() -> None:
    """Test that unknown model returns None for cost."""
    metrics = ProviderMetrics("unknown", model_id="unknown:model")

    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] is None


def test_cost_calculation_unknown_provider_returns_none() -> None:
    """Test that unknown provider returns None for cost."""
    metrics = ProviderMetrics("totally_unknown_provider")

    metrics.record_success(latency_seconds=1.0, input_tokens=100, output_tokens=50)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] is None


def test_cost_calculation_missing_usage_returns_none() -> None:
    """Test that missing usage does not produce fabricated cost."""
    metrics = ProviderMetrics("test")

    metrics.record_success(latency_seconds=1.0)  # No token info

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] is None


def test_cost_calculation_partial_usage_conservative() -> None:
    """Test that partial usage follows conservative behavior."""
    metrics = ProviderMetrics("test", model_id="gemini:gemini-2.5-flash")

    # Only input tokens - cost should be computable for input
    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    # Input: 1M @ $0.075/M = $0.075, output: 0 (since None treated as 0 in cost)
    assert data["cost"]["estimated_cost_usd"] == 0.075


def test_cost_calculation_zero_usage_known_pricing() -> None:
    """Test that zero usage with known pricing produces zero cost."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    metrics.record_success(latency_seconds=1.0, input_tokens=0, output_tokens=0)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.0


def test_cost_calculation_local_ollama_no_cost() -> None:
    """Test that local Ollama models without configured pricing produce no cost."""
    metrics = ProviderMetrics("ollama", model_id="ollama:llama3.2:3b")

    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    # Ollama has $0 pricing in DEFAULT_MODEL_PRICING
    assert data["cost"]["estimated_cost_usd"] == 0.0


def test_retry_attempts_do_not_duplicate_usage() -> None:
    """Test that retry attempts do not duplicate successful usage."""
    clear_provider_cache()
    collector = get_metrics_collector()
    collector.reset()

    # Create a provider that fails twice then succeeds
    call_count = {"count": 0}

    class FlakyProvider:
        def __init__(self, name: str):
            self.model_id = f"mock:{name}"
            self.name = name

        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings: GenerationSettings,
        ) -> str:
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise Exception("Transient")
            return "Success"

        def generate_with_metadata(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings: GenerationSettings,
        ) -> GenerationResponse:
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise Exception("Transient")
            return GenerationResponse(text="Success")

    inner = FlakyProvider("flaky")
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
    retry = RetryProvider(inner, policy, "flaky")
    cb = CircuitBreaker("flaky", CircuitBreakerPolicy(enabled=False))
    cb_provider = CircuitBreakerProvider(retry, cb, "flaky")

    cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    metrics = collector.get_metrics()
    # Only one successful request should be recorded
    assert metrics["providers"]["flaky"]["requests"]["successful"] == 1
    assert metrics["providers"]["flaky"]["usage"]["input_tokens"] is None
    assert metrics["providers"]["flaky"]["usage"]["output_tokens"] is None


def test_fallback_success_records_only_fallback_usage() -> None:
    """Test that failed primary followed by successful fallback records only fallback usage."""
    clear_provider_cache()
    collector = get_metrics_collector()
    collector.reset()

    primary = MockProvider("primary", error=LLMError("Transient"))
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


def test_legacy_provider_continues_to_work() -> None:
    """Test that legacy providers returning plain strings continue to work."""
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
    assert metrics["providers"]["legacy"]["usage"]["output_tokens"] is None


# =============================================================================
# Fix 1 & 2: Explicit total_tokens preservation and partial cost semantics
# =============================================================================


def test_explicit_total_differs_from_sum_preserved() -> None:
    """Test that explicit total differs from input+output sum is preserved."""
    metrics = ProviderMetrics("test")

    # Explicit total (200) differs from input+output (150)
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=200,  # Explicit: 200 != 100+50=150
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["output_tokens"] == 50
    assert data["usage"]["total_tokens"] == 200  # Explicit total preserved


def test_explicit_total_with_missing_input_output_preserved() -> None:
    """Test that explicit total with missing input/output is preserved."""
    metrics = ProviderMetrics("test")

    # Only explicit total provided, no input/output
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=None,
        output_tokens=None,
        total_tokens=500,
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] is None
    assert data["usage"]["output_tokens"] is None
    assert data["usage"]["total_tokens"] == 500


def test_missing_explicit_total_with_complete_input_output_computes_sum() -> None:
    """Test missing explicit total with complete input/output computes sum."""
    metrics = ProviderMetrics("test")

    # No explicit total, but both input and output provided
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=50,
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["output_tokens"] == 50
    assert data["usage"]["total_tokens"] == 150  # Computed: 100+50


def test_missing_explicit_total_with_partial_input_output_remains_none() -> None:
    """Test missing explicit total with partial input/output remains None."""
    metrics = ProviderMetrics("test")

    # Only input provided, no explicit total
    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=None,
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 100
    assert data["usage"]["output_tokens"] is None
    assert data["usage"]["total_tokens"] is None  # Not computed


def test_explicit_zero_total_preserved() -> None:
    """Test that explicit zero total is preserved (not confused with missing)."""
    metrics = ProviderMetrics("test")

    metrics.record_success(
        latency_seconds=0.1,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,  # Explicit zero
    )

    data = metrics.to_dict()
    assert data["usage"]["input_tokens"] == 0
    assert data["usage"]["output_tokens"] == 0
    assert data["usage"]["total_tokens"] == 0  # Zero preserved, not None


def test_explicit_total_passes_through_circuit_breaker() -> None:
    """Test explicit total passes through CircuitBreakerProvider into collector."""
    clear_provider_cache()
    collector = get_metrics_collector()
    collector.reset()

    # Provider returns explicit total_tokens=500 (differs from input+output=150)
    inner = MockProvider(
        "test",
        GenerationResponse(text="Success", input_tokens=100, output_tokens=50, total_tokens=500),
    )
    policy = RetryPolicy(max_attempts=3)
    retry = RetryProvider(inner, policy, "test")
    cb = CircuitBreaker("test", CircuitBreakerPolicy(enabled=False))
    cb_provider = CircuitBreakerProvider(retry, cb, "test")

    response = cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    assert response.total_tokens == 500  # Response preserves explicit total

    metrics = collector.get_metrics()
    # Explicit total (500) should be recorded, not computed sum (150)
    assert metrics["providers"]["test"]["usage"]["total_tokens"] == 500


def test_partial_input_cost_sets_is_partial_true() -> None:
    """Test that input-only usage with known pricing sets is_partial=true."""
    metrics = ProviderMetrics("test", model_id="gemini:gemini-2.5-flash")

    # Only input tokens provided
    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.075  # Input cost only
    assert data["cost"]["is_partial"] is True  # Partial: only input


def test_partial_output_cost_sets_is_partial_true() -> None:
    """Test that output-only usage with known pricing sets is_partial=true."""
    metrics = ProviderMetrics("test", model_id="gemini:gemini-2.5-flash")

    # Only output tokens provided
    metrics.record_success(latency_seconds=1.0, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.30  # Output cost only
    assert data["cost"]["is_partial"] is True  # Partial: only output


def test_complete_usage_sets_is_partial_false() -> None:
    """Test that complete usage (both input and output) sets is_partial=false."""
    metrics = ProviderMetrics("test", model_id="gemini:gemini-2.5-flash")

    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.375
    assert data["cost"]["is_partial"] is False  # Complete usage


def test_no_usage_returns_unknown_cost() -> None:
    """Test that no usage returns unknown cost rather than zero."""
    metrics = ProviderMetrics("test", model_id="gemini:gemini-2.5-flash")

    metrics.record_success(latency_seconds=1.0)  # No token info

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] is None
    assert data["cost"]["is_partial"] is False  # Not partial, just unknown


def test_unknown_model_returns_none_cost() -> None:
    """Test that unknown model returns None cost."""
    metrics = ProviderMetrics("unknown", model_id="unknown:model")

    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] is None
    assert data["cost"]["is_partial"] is False


def test_known_zero_priced_ollama_usage_remains_zero_cost() -> None:
    """Test that known zero-priced Ollama usage remains zero cost."""
    metrics = ProviderMetrics("ollama", model_id="ollama:llama3.2:3b")

    metrics.record_success(latency_seconds=1.0, input_tokens=1_000_000, output_tokens=1_000_000)

    data = metrics.to_dict(pricing_map=DEFAULT_MODEL_PRICING)
    assert data["cost"]["estimated_cost_usd"] == 0.0
    assert data["cost"]["is_partial"] is False  # Complete usage


def test_existing_api_fields_remain_compatible() -> None:
    """Test that existing API response fields remain backward compatible."""
    metrics = ProviderMetrics("gemini", model_id="gemini:gemini-2.5-flash")

    metrics.record_success(latency_seconds=1.0, input_tokens=100, output_tokens=50)

    data = metrics.to_dict()

    # Original fields must exist and have correct types
    assert "provider" in data
    assert "model_id" in data
    assert "requests" in data
    assert "latency" in data
    assert "reliability" in data
    assert "usage" in data
    assert "cost" in data

    # Cost structure must have all original fields
    assert "estimated_cost_usd" in data["cost"]
    assert "currency" in data["cost"]
    assert "is_estimate" in data["cost"]
    # New field must be present
    assert "is_partial" in data["cost"]

    # Usage structure must have all fields
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


def test_retry_fallback_attribution_and_no_duplication_remain_intact() -> None:
    """Test that retry and fallback attribution and no-duplicate behavior remain intact."""
    clear_provider_cache()
    collector = get_metrics_collector()
    collector.reset()

    # Primary fails once, then fallback succeeds
    primary = MockProvider("primary", error=LLMError("Transient"))
    fallback = MockProvider(
        "fallback",
        GenerationResponse(text="Fallback", input_tokens=50, output_tokens=25, total_tokens=75),
    )
    fallback_provider = FallbackProvider(primary, [fallback], "primary", ["fallback"])
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.001)
    retry = RetryProvider(fallback_provider, policy, "fallback")
    cb = CircuitBreaker("fallback", CircuitBreakerPolicy(enabled=False))
    cb_provider = CircuitBreakerProvider(retry, cb, "fallback")

    cb_provider.generate_with_metadata("sys", "user", settings=GenerationSettings())

    metrics = collector.get_metrics()
    # Fallback usage should be attributed to fallback provider
    assert metrics["providers"]["fallback"]["usage"]["input_tokens"] == 50
    assert metrics["providers"]["fallback"]["usage"]["output_tokens"] == 25
    assert metrics["providers"]["fallback"]["usage"]["total_tokens"] == 75
    # Primary failure should not have successful usage
    assert metrics["providers"]["primary"]["requests"]["successful"] == 0
    # Only one successful request total
    total_success = sum(p["requests"]["successful"] for p in metrics["providers"].values())
    assert total_success == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
