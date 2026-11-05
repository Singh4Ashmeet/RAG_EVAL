"""Unit tests for LLM provider observability, metrics collector, and health endpoints."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from core.llm.metrics import (
    DEFAULT_MODEL_PRICING,
    ProviderMetrics,
    ProviderMetricsCollector,
    get_metrics_collector,
)
from core.llm.router import (
    clear_provider_cache,
    get_circuit_breaker,
)


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


def test_api_endpoints_metrics_and_health() -> None:
    """Test GET /api/v1/llm/metrics and GET /api/v1/llm/health endpoints without secret leakage."""
    clear_provider_cache()
    client = TestClient(app)

    # Health endpoint
    health_resp = client.get("/api/v1/llm/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    assert "status" in health_json
    assert "providers" in health_json
    assert "gemini" in health_json["providers"]

    # Metrics endpoint
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
