"""Provider-level operational metrics and observability.

Collects thread-safe, bounded, in-memory operational metrics for LLM providers:
- Request volume, success, failure, error rates
- Latency percentiles (P50, P95, avg, min, max)
- Reliability metrics (retries, retry exhaustions, fallbacks, circuit breaker states)
- Token usage (when provided by backend) and cost estimation
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default model pricing (USD per 1,000,000 tokens)
# Users can override or configure via set_pricing()
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini": {"input": 0.075, "output": 0.30},
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.20},
    "groq": {"input": 0.20, "output": 0.20},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "ollama": {"input": 0.0, "output": 0.0},
}


class ProviderMetrics:
    """Bounded, thread-safe operational metrics for a single provider."""

    def __init__(
        self,
        provider_name: str,
        model_id: str = "",
        max_latency_samples: int = 1000,
    ) -> None:
        self.provider_name = provider_name
        self.model_id = model_id
        self._lock = threading.Lock()

        # Request counts
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Latencies (seconds) - bounded circular buffer
        self._latencies: collections.deque[float] = collections.deque(maxlen=max_latency_samples)

        # Reliability
        self.retries = 0
        self.retry_exhaustions = 0
        self.fallbacks = 0
        self.circuit_openings = 0
        self.circuit_rejections = 0
        self.half_open_probes = 0
        self.timeouts = 0

        # Token usage
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.total_tokens: int | None = None

        # Timestamps
        self.last_success_time: float | None = None
        self.last_failure_time: float | None = None

    def record_success(
        self,
        latency_seconds: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self._latencies.append(latency_seconds)
            self.last_success_time = time.monotonic()

            if input_tokens is not None or output_tokens is not None:
                inp = input_tokens or 0
                out = output_tokens or 0
                self.input_tokens = (self.input_tokens or 0) + inp
                self.output_tokens = (self.output_tokens or 0) + out
                self.total_tokens = (self.total_tokens or 0) + (inp + out)

    def record_failure(
        self,
        latency_seconds: float,
        error: Exception,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            if latency_seconds > 0:
                self._latencies.append(latency_seconds)
            self.last_failure_time = time.monotonic()

            err_str = str(error).lower()
            err_type = type(error).__name__.lower()
            if "timeout" in err_str or "timeout" in err_type:
                self.timeouts += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_retry_exhaustion(self) -> None:
        with self._lock:
            self.retry_exhaustions += 1

    def record_fallback(self) -> None:
        with self._lock:
            self.fallbacks += 1

    def record_circuit_opening(self) -> None:
        with self._lock:
            self.circuit_openings += 1

    def record_circuit_rejection(self) -> None:
        with self._lock:
            self.circuit_rejections += 1

    def record_half_open_probe(self) -> None:
        with self._lock:
            self.half_open_probes += 1

    def get_latency_stats(self) -> dict[str, float]:
        with self._lock:
            if not self._latencies:
                return {
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                }
            sorted_latencies = sorted(self._latencies)
            n = len(sorted_latencies)
            avg_sec = sum(sorted_latencies) / n
            min_sec = sorted_latencies[0]
            max_sec = sorted_latencies[-1]

            # P50 (median)
            p50_idx = min(int(n * 0.5), n - 1)
            p50_sec = sorted_latencies[p50_idx]

            # P95
            p95_idx = min(int(n * 0.95), n - 1)
            p95_sec = sorted_latencies[p95_idx]

            return {
                "avg_ms": round(avg_sec * 1000, 2),
                "min_ms": round(min_sec * 1000, 2),
                "max_ms": round(max_sec * 1000, 2),
                "p50_ms": round(p50_sec * 1000, 2),
                "p95_ms": round(p95_sec * 1000, 2),
            }

    def compute_cost_estimate(self, pricing_map: dict[str, dict[str, float]]) -> float | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None

        # Look up pricing by model_id or provider_name
        pricing = pricing_map.get(self.model_id)
        if not pricing:
            # Try model without prefix, e.g. "gemini:gemini-2.5-flash" -> "gemini-2.5-flash"
            bare_model = self.model_id.split(":", 1)[-1] if ":" in self.model_id else self.model_id
            pricing = pricing_map.get(bare_model) or pricing_map.get(self.provider_name)

        if not pricing:
            return None

        inp_price_per_m = pricing.get("input", 0.0)
        out_price_per_m = pricing.get("output", 0.0)

        inp_cost = ((self.input_tokens or 0) / 1_000_000.0) * inp_price_per_m
        out_cost = ((self.output_tokens or 0) / 1_000_000.0) * out_price_per_m
        return round(inp_cost + out_cost, 6)

    def to_dict(self, pricing_map: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
        with self._lock:
            total = self.total_requests
            succ = self.successful_requests
            fail = self.failed_requests
            error_rate = round(fail / total, 4) if total > 0 else 0.0
            success_rate = round(succ / total, 4) if total > 0 else 0.0

        latency_stats = self.get_latency_stats()
        cost_est = (
            self.compute_cost_estimate(pricing_map or DEFAULT_MODEL_PRICING)
            if pricing_map or DEFAULT_MODEL_PRICING
            else None
        )

        with self._lock:
            data = {
                "provider": self.provider_name,
                "model_id": self.model_id,
                "requests": {
                    "total": total,
                    "successful": succ,
                    "failed": fail,
                    "error_rate": error_rate,
                    "success_rate": success_rate,
                },
                "latency": latency_stats,
                "reliability": {
                    "retries": self.retries,
                    "retry_exhaustions": self.retry_exhaustions,
                    "fallbacks": self.fallbacks,
                    "circuit_openings": self.circuit_openings,
                    "circuit_rejections": self.circuit_rejections,
                    "half_open_probes": self.half_open_probes,
                    "timeouts": self.timeouts,
                },
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens,
                },
                "cost": {
                    "estimated_cost_usd": cost_est,
                    "currency": "USD",
                    "is_estimate": True,
                },
                "last_success_time": self.last_success_time,
                "last_failure_time": self.last_failure_time,
            }
            return data


class ProviderMetricsCollector:
    """Thread-safe, bounded in-memory collector for LLM provider metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, ProviderMetrics] = {}
        self._pricing: dict[str, dict[str, float]] = dict(DEFAULT_MODEL_PRICING)
        self._fallback_events: collections.deque[dict[str, Any]] = collections.deque(maxlen=100)

    def get_or_create(self, provider_name: str, model_id: str = "") -> ProviderMetrics:
        with self._lock:
            if provider_name not in self._metrics:
                self._metrics[provider_name] = ProviderMetrics(
                    provider_name=provider_name,
                    model_id=model_id,
                )
            elif model_id and not self._metrics[provider_name].model_id:
                self._metrics[provider_name].model_id = model_id
            return self._metrics[provider_name]

    def set_pricing(
        self,
        model_or_provider: str,
        input_per_1m: float,
        output_per_1m: float,
    ) -> None:
        with self._lock:
            self._pricing[model_or_provider] = {
                "input": input_per_1m,
                "output": output_per_1m,
            }

    def record_success(
        self,
        provider_name: str,
        model_id: str = "",
        latency: float = 0.0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response: Any = None,
    ) -> None:
        try:
            m = self.get_or_create(provider_name, model_id)
            m.record_success(
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            logger.debug("Failed to record metric success: %s", exc)

    def record_failure(
        self,
        provider_name: str,
        model_id: str = "",
        latency: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        try:
            m = self.get_or_create(provider_name, model_id)
            m.record_failure(
                latency_seconds=latency,
                error=error or Exception("Unknown error"),
            )
        except Exception as exc:
            logger.debug("Failed to record metric failure: %s", exc)

    def record_retry(self, provider_name: str, model_id: str = "") -> None:
        try:
            m = self.get_or_create(provider_name, model_id)
            m.record_retry()
        except Exception as exc:
            logger.debug("Failed to record metric retry: %s", exc)

    def record_retry_exhaustion(self, provider_name: str, model_id: str = "") -> None:
        try:
            m = self.get_or_create(provider_name, model_id)
            m.record_retry_exhaustion()
        except Exception as exc:
            logger.debug("Failed to record metric retry exhaustion: %s", exc)

    def record_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str = "",
    ) -> None:
        try:
            m = self.get_or_create(from_provider)
            m.record_fallback()
            with self._lock:
                self._fallback_events.append(
                    {
                        "from_provider": from_provider,
                        "to_provider": to_provider,
                        "reason": reason,
                        "timestamp": time.time(),
                    }
                )
        except Exception as exc:
            logger.debug("Failed to record metric fallback: %s", exc)

    def record_circuit_opening(self, provider_name: str) -> None:
        try:
            m = self.get_or_create(provider_name)
            m.record_circuit_opening()
        except Exception as exc:
            logger.debug("Failed to record metric circuit opening: %s", exc)

    def record_circuit_rejection(self, provider_name: str, model_id: str = "") -> None:
        try:
            m = self.get_or_create(provider_name, model_id)
            m.record_circuit_rejection()
        except Exception as exc:
            logger.debug("Failed to record metric circuit rejection: %s", exc)

    def record_half_open_probe(self, provider_name: str) -> None:
        try:
            m = self.get_or_create(provider_name)
            m.record_half_open_probe()
        except Exception as exc:
            logger.debug("Failed to record metric half-open probe: %s", exc)

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            pricing_copy = dict(self._pricing)
            providers_copy = list(self._metrics.items())
            fallback_events_copy = list(self._fallback_events)

        providers_dict = {}
        for name, metrics_obj in providers_copy:
            data = metrics_obj.to_dict(pricing_map=pricing_copy)
            # Add circuit breaker info if available
            try:
                from core.llm.router import get_circuit_breaker

                cb = get_circuit_breaker(name)
                data["circuit_breaker"] = cb.get_stats()
            except Exception:
                pass
            providers_dict[name] = data

        return {
            "providers": providers_dict,
            "fallback_events": fallback_events_copy,
        }

    def get_health(self) -> dict[str, Any]:
        from core.llm.router import get_circuit_breaker

        # Known providers or providers with recorded metrics
        provider_names = {"ollama", "gemini", "groq"}
        with self._lock:
            provider_names.update(self._metrics.keys())

        providers_health = {}
        overall_healthy = True
        overall_degraded = False

        for name in sorted(provider_names):
            try:
                cb = get_circuit_breaker(name)
                stats = cb.get_stats()
                cb_state = stats["state"]
                failure_count = stats["failure_count"]
                last_failure = stats["last_failure_time"]
            except Exception:
                cb_state = "closed"
                failure_count = 0
                last_failure = None

            with self._lock:
                m = self._metrics.get(name)
                last_success = m.last_success_time if m else None

            is_healthy = cb_state in ("closed", "half_open")
            if cb_state == "open":
                overall_healthy = False
                status_str = "unhealthy"
            elif cb_state == "half_open":
                overall_degraded = True
                status_str = "degraded"
            else:
                status_str = "healthy"

            providers_health[name] = {
                "status": status_str,
                "state": cb_state,
                "healthy": is_healthy,
                "consecutive_failures": failure_count,
                "last_failure_time": last_failure,
                "last_success_time": last_success,
            }

        if not overall_healthy:
            any_healthy = any(p["healthy"] for p in providers_health.values())
            system_status = "degraded" if any_healthy else "unhealthy"
        elif overall_degraded:
            system_status = "degraded"
        else:
            system_status = "healthy"

        return {
            "status": system_status,
            "providers": providers_health,
        }

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._fallback_events.clear()


_collector = ProviderMetricsCollector()


def get_metrics_collector() -> ProviderMetricsCollector:
    """Get the global provider metrics collector instance."""
    return _collector
