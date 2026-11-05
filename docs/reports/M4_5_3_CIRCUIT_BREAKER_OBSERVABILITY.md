# M4.5.3 — Circuit Breaker & Provider Observability

**Date:** 2026-11-05
**Milestone:** M4.5 — Production Reliability & Resilience
**Phase:** M4.5.3 — Circuit Breaker & Provider Observability
**Status:** Complete ✅
**Previous Baseline:** 507 passing tests
**New Baseline:** 540 passing tests (+33 tests)

---

## 1. Executive Summary

Milestone M4.5.3 introduces two critical production resilience and monitoring capabilities to `RAG_EVAL`:
1. **Provider-Specific Circuit Breakers:** Process-local, thread-safe three-state circuit breakers (`CLOSED`, `OPEN`, `HALF_OPEN`) providing immediate failure isolation for each LLM provider. This prevents retry storms against impaired backends and enables instant failover without redundant latency.
2. **Provider-Level Observability & Metrics:** Thread-safe, bounded in-memory metrics tracking request volume, success/error rates, latency percentiles (avg, min, max, P50, P95), reliability events (retries, exhaustions, fallbacks, circuit openings, rejections, timeouts), token usage, and model cost estimation.
3. **HTTP Health & Metrics Endpoints:** Read-only, unauthenticated monitoring endpoints at `GET /api/v1/llm/metrics` and `GET /api/v1/llm/health`, strictly scrubbed of secrets or credentials.

---

## 2. Architecture & Layering

The generation pipeline cleanly composes provider layers to guarantee strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Request                           │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FallbackProvider                            │
│  Tries primary; if CircuitBreakerOpen or exhausted, fails over  │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CircuitBreakerProvider                         │
│  - Checks allow_request(): fast-fails if OPEN                   │
│  - Records logical success/failure on breaker                   │
│  - Records metrics (requests, latency, tokens, cost)            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RetryProvider                              │
│  - Attempts generation with exponential backoff & jitter        │
│  - Respects Retry-After header and provider timeout deadline    │
│  - Single logical outcome returned to CircuitBreaker            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Base LLM Provider (Raw SDK)                    │
│               Ollama / Gemini / Groq Provider                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Behavioral Invariants

1. **Logical Request Outcome Accounting:**
   - A single user generation request that retries 3 times internally before succeeding counts as **0 circuit breaker failures** and **1 circuit breaker success**.
   - A request that exhausts its retry budget with recoverable errors counts as **1 single circuit breaker failure**.
   - Retries within budget do not falsely open the circuit breaker.
2. **Error Filtering:**
   - **Counted as Circuit Breaker Failures:** Transient/recoverable errors only (HTTP 429, HTTP 5xx, timeouts, connection errors, `ProviderUnavailableError`, `ProviderRateLimitError`).
   - **Ignored by Circuit Breaker:** Permanent client errors (HTTP 400, 401, 403, 404, invalid API keys, authentication errors, invalid request format). These raise immediately and do not impact provider health.
3. **Locking Discipline:**
   - Thread safety is managed using standard `threading.Lock`.
   - Locks are acquired strictly during counter inspection and state transitions (microseconds).
   - Locks are **never held** across network I/O or LLM generation calls.
4. **Immediate Fallback on Open Circuit:**
   - When a provider's circuit breaker is in `OPEN` state, calls fast-fail with `CircuitBreakerOpenError`.
   - `FallbackProvider` catches `CircuitBreakerOpenError` and immediately routes the request to the next healthy provider in the fallback chain without attempting retries against the open provider.

---

## 3. Circuit Breaker State Machine

```
              ┌───────────────────────────────────────────────────┐
              │                                                   │
              ▼                                                   │
       ┌──────────────┐      failure_count >= threshold     ┌───────────┐
       │    CLOSED    │ ──────────────────────────────────> │   OPEN    │
       └──────────────┘                                     └───────────┘
              ▲                                                   │
              │                                                   │ recovery_timeout_seconds
              │                                                   │ elapsed
              │ probes == half_open_max_calls                     ▼
              │ (all successful)                            ┌───────────┐
              └──────────────────────────────────────────── │ HALF_OPEN │
                                                            └───────────┘
                                         probe failure            │
                                         (any recoverable error)  │
                                         ─────────────────────────┘
```

- **CLOSED:** Normal operation. Requests are allowed. Successes reset the consecutive failure counter. Reaching `CIRCUIT_BREAKER_FAILURE_THRESHOLD` consecutive recoverable failures transitions state to `OPEN`.
- **OPEN:** Fast failure. Requests are rejected immediately with `CircuitBreakerOpenError`. After `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS`, state transitions automatically to `HALF_OPEN`.
- **HALF_OPEN:** Probe phase. Only `CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS` probe requests are allowed through to verify provider recovery. If probe succeeds, state transitions back to `CLOSED`. If any probe fails with a recoverable error, state immediately returns to `OPEN` and the recovery timer resets.

---

## 4. Observability & Operational Metrics

Implemented in `core/llm/metrics.py` via `ProviderMetricsCollector`:

| Metric Category | Metrics Tracked | Description |
|-----------------|-----------------|-------------|
| **Request Counts** | `total`, `successful`, `failed`, `error_rate`, `success_rate` | Volume and reliability rates per provider. |
| **Latency Statistics** | `avg_ms`, `min_ms`, `max_ms`, `p50_ms`, `p95_ms` | Calculated from a bounded circular buffer (`deque(maxlen=1000)`). Bounded memory footprint. |
| **Reliability Events** | `retries`, `retry_exhaustions`, `fallbacks`, `circuit_openings`, `circuit_rejections`, `half_open_probes`, `timeouts` | Granular operational counters for troubleshooting. |
| **Usage (Tokens)** | `input_tokens`, `output_tokens`, `total_tokens` | Derived strictly from real provider response metadata. Left as `None` when absent (never fabricated). |
| **Cost Estimation** | `estimated_cost_usd`, `currency`, `is_estimate` | Configurable pricing map ($ per 1M tokens). Omitted when pricing or token counts are unavailable. |
| **Circuit Breaker State** | `state`, `failure_count`, `last_failure_time`, `last_state_change` | Real-time health reflection per provider. |

### Fail-Safe Metric Collection
All metric recordings are wrapped in exception handlers. Metric recording failures are logged and swallowed; metric collection failures will **never** cause an LLM generation call to fail.

---

## 5. API Endpoints

### 1. `GET /api/v1/llm/metrics`
Returns comprehensive operational metrics for all providers and fallback events:

```json
{
  "providers": {
    "gemini": {
      "provider": "gemini",
      "model_id": "gemini:gemini-2.5-flash",
      "requests": {
        "total": 100,
        "successful": 98,
        "failed": 2,
        "error_rate": 0.02,
        "success_rate": 0.98
      },
      "latency": {
        "avg_ms": 312.45,
        "min_ms": 120.10,
        "max_ms": 1150.00,
        "p50_ms": 280.50,
        "p95_ms": 890.20
      },
      "reliability": {
        "retries": 4,
        "retry_exhaustions": 1,
        "fallbacks": 1,
        "circuit_openings": 0,
        "circuit_rejections": 0,
        "half_open_probes": 0,
        "timeouts": 0
      },
      "usage": {
        "input_tokens": 15420,
        "output_tokens": 4210,
        "total_tokens": 19630
      },
      "cost": {
        "estimated_cost_usd": 0.002419,
        "currency": "USD",
        "is_estimate": true
      },
      "circuit_breaker": {
        "provider": "gemini",
        "state": "closed",
        "failure_count": 0,
        "success_count": 0,
        "half_open_calls": 0,
        "last_failure_time": null,
        "last_state_change": 1730821234.56
      }
    }
  },
  "fallback_events": []
}
```

### 2. `GET /api/v1/llm/health`
Returns system and per-provider health status:

```json
{
  "status": "healthy",
  "providers": {
    "gemini": {
      "status": "healthy",
      "state": "closed",
      "healthy": true,
      "consecutive_failures": 0,
      "last_failure_time": null,
      "last_success_time": 1730821250.12
    },
    "groq": {
      "status": "healthy",
      "state": "closed",
      "healthy": true,
      "consecutive_failures": 0,
      "last_failure_time": null,
      "last_success_time": null
    },
    "ollama": {
      "status": "healthy",
      "state": "closed",
      "healthy": true,
      "consecutive_failures": 0,
      "last_failure_time": null,
      "last_success_time": null
    }
  }
}
```

**Security Assurance:** Neither endpoint exposes API keys, tokens, or credential headers.

---

## 6. Multi-Worker Considerations

The metrics and circuit breakers implemented here are **process-local** (in-memory per Python process).
- **Current Architecture:** Each worker process tracks its own consecutive failure counts and local latency reservoir.
- **Multi-Worker Note:** If running multiple Uvicorn/FastAPI workers or RQ background workers, circuit states are maintained independently within each worker process. In the future, if distributed consensus across workers is required, the `CircuitBreaker` and `ProviderMetricsCollector` interfaces can be backed by a Redis-based distributed lock / sliding window counter without altering the caller contracts.

---

## 7. Configuration Reference

Documented in `.env.example` and configured in `core/config.py`:

```bash
# ==============================================================================
# Provider Circuit Breaker Configuration
# ==============================================================================
# Enable circuit breaker for provider failure isolation
CIRCUIT_BREAKER_ENABLED=true

# Number of consecutive provider failures before opening circuit
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5

# Seconds to wait before attempting recovery (HALF_OPEN state)
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=30.0

# Maximum probe calls allowed in HALF_OPEN state
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=1

# Rolling window in seconds for failure counting
CIRCUIT_BREAKER_WINDOW_SECONDS=60.0
```

---

## 8. Verification & Test Summary

### Test Results

```
tests/unit/test_provider_fallback.py: 13 passed
tests/unit/test_retry_engine.py:      26 passed
tests/unit/test_circuit_breaker.py:   19 passed
tests/unit/test_provider_metrics.py:  14 passed
Total Resilience Test Suite:          72 passed
Total Repository Test Suite:         540 passed (0 failed, 1 warning)
```

### Static Analysis & Linting

```
uv run ruff check .               --> All checks passed!
uv run ruff format --check .      --> 141 files already formatted
```

---

## 9. Artifacts Created & Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `core/llm/metrics.py` | **New** | In-memory thread-safe metrics collector, bounded latency buffer, cost estimation, health reporter. |
| `core/llm/router.py` | **Modified** | Added `CircuitBreaker`, `CircuitBreakerPolicy`, `CircuitBreakerProvider`, integrated with retry & fallback. |
| `apps/api/main.py` | **Modified** | Added `GET /api/v1/llm/metrics` and `GET /api/v1/llm/health` endpoints. |
| `tests/unit/test_circuit_breaker.py` | **New** | 19 unit & integration tests covering state transitions, error filtering, retry integration, thread safety. |
| `tests/unit/test_provider_metrics.py` | **New** | 14 unit tests covering percentiles, token/cost estimation, memory limits, and API endpoints. |
| `docs/reports/M4_5_3_CIRCUIT_BREAKER_OBSERVABILITY.md` | **New** | This comprehensive completion report. |
