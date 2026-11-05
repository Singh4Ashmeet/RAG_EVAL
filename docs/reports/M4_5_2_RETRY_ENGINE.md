# M4.5.2 — Intelligent Retry Engine with Exponential Backoff

**Date:** 2026-11-05
**Status:** Complete ✅
**Branch:** main
**Based on:** M4.5.1 (Provider Fallback)

---

## 1. Executive Summary

This milestone implements a production-oriented retry engine with exponential backoff, jitter, Retry-After support, and deadline awareness. The retry engine integrates cleanly with the existing M4.5.1 fallback mechanism, maintaining strict separation between **retry layer** (same provider, multiple attempts) and **fallback layer** (different provider).

**Key Achievement:** All three providers (Ollama, Gemini, Groq) now support intelligent retry with configurable policies, integrated with provider fallback.

---

## 2. Architecture

### Retry vs. Fallback Separation

| Layer | Responsibility | Configuration |
|-------|---------------|---------------|
| **Retry Layer** | Multiple attempts against the SAME provider | `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY_SECONDS`, etc. |
| **Fallback Layer** | Switch to another provider after provider exhausts retry budget | `FALLBACK_ENABLED`, `FALLBACK_PROVIDERS` |

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Request                           │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Primary Provider (with Retry)                      │
│  Attempt 1 → [backoff] → Attempt 2 → [backoff] → Attempt 3     │
│                      │                                          │
│              ┌────────┴────────┐                                │
│              ▼                 ▼                                │
│         Success            All retries exhausted                │
│           │                      │                              │
│           ▼                      ▼                              │
│      Return Response    Fallback Provider (with Retry)          │
│                                  │                              │
│                                  ▼                              │
│                           All fallbacks exhausted               │
│                              │                                  │
│                              ▼                                  │
│                     Raise ProviderFallbackExhaustedError        │
└─────────────────────────────────────────────────────────────────┘
```

### Maximum Attempts Calculation

With 3 providers and 3 retry attempts per provider:
- **Maximum 9 provider generation attempts** (3 providers × 3 attempts each)
- Each provider gets its own independent retry budget
- SDK retries are accounted for (see SDK section)

---

## 3. Configuration

### New Environment Variables (`.env.example`)

```bash
# ==============================================================================
# Provider Retry Configuration
# ==============================================================================
# Enable retry for transient provider failures
RETRY_ENABLED=true

# Maximum generation attempts per provider (including initial attempt)
RETRY_MAX_ATTEMPTS=3

# Initial backoff delay in seconds
RETRY_BASE_DELAY_SECONDS=1.0

# Maximum backoff delay cap in seconds
RETRY_MAX_DELAY_SECONDS=10.0

# Exponential base for backoff calculation
RETRY_EXPONENTIAL_BASE=2.0

# Apply jitter to prevent synchronized retry storms
RETRY_JITTER_ENABLED=true

# Maximum total time budget for all retry attempts per provider (seconds)
RETRY_TIMEOUT_SECONDS=30.0
```

### Settings Class (`core/config.py`)

```python
retry_enabled: bool = Field(
    default=True,
    validation_alias="RETRY_ENABLED",
    description="Enable retry for transient provider failures",
)
retry_max_attempts: int = Field(
    default=3,
    validation_alias="RETRY_MAX_ATTEMPTS",
    ge=1,
    le=10,
    description="Maximum generation attempts per provider (including initial attempt)",
)
retry_base_delay_seconds: float = Field(
    default=1.0,
    validation_alias="RETRY_BASE_DELAY_SECONDS",
    ge=0.0,
    le=60.0,
    description="Initial backoff delay in seconds",
)
retry_max_delay_seconds: float = Field(
    default=10.0,
    validation_alias="RETRY_MAX_DELAY_SECONDS",
    ge=0.0,
    le=300.0,
    description="Maximum backoff delay cap in seconds",
)
retry_exponential_base: float = Field(
    default=2.0,
    validation_alias="RETRY_EXPONENTIAL_BASE",
    ge=1.0,
    le=5.0,
    description="Exponential base for backoff calculation",
)
retry_jitter_enabled: bool = Field(
    default=True,
    validation_alias="RETRY_JITTER_ENABLED",
    description="Apply jitter to prevent synchronized retry storms",
)
retry_timeout_seconds: float = Field(
    default=30.0,
    validation_alias="RETRY_TIMEOUT_SECONDS",
    ge=1.0,
    le=300.0,
    description="Maximum total time budget for all retry attempts per provider",
)
```

---

## 4. Exponential Backoff Implementation

### Formula

```python
delay = min(base_delay * (exponential_base**attempt), max_delay)
```

With optional jitter:
```python
delay += random.uniform(0, delay * 0.1)  # Up to 10% jitter
```

### Retry-After Priority

If server provides `Retry-After` header:
- Uses server value as base delay
- Still capped by `max_delay`
- Jitter still applied on top

### Examples (defaults: base=1.0, base=2.0, max=10.0)

| Attempt | Calculated | With Jitter (0-10%) |
|---------|------------|---------------------|
| 0 | 1.0s | 1.0–1.1s |
| 1 | 2.0s | 2.0–2.2s |
| 2 | 4.0s | 4.0–4.4s |
| 3 | 8.0s | 8.0–8.8s |
| 4+ | 10.0s (capped) | 10.0–11.0s |

---

## 5. Retry-After Support

### Supported Formats

| Format | Example | Handling |
|--------|---------|----------|
| Integer seconds | `Retry-After: 30` | Parsed as `float` seconds |
| HTTP-date | `Retry-After: Wed, 21 Oct 2025 07:28:00 GMT` | Parsed via `email.utils.parsedate_to_datetime` |

### Behavior

- Server `Retry-After` takes priority over calculated backoff
- Still capped by `RETRY_MAX_DELAY_SECONDS`
- Jitter applied on top
- Malformed headers gracefully ignored

---

## 6. Failure Classification

Built on M4.5.1's `_is_recoverable_error()` with retry-specific additions.

### Retryable (Triggers Retry)

| Category | Examples |
|----------|----------|
| Rate Limit | HTTP 429, `ProviderRateLimitError` |
| Server Errors | HTTP 500, 502, 503, 504 |
| Connection | Timeout, DNS failure, connection refused |
| Temporary | `ProviderUnavailableError`, `ProviderFallbackExhaustedError` |

### Non-Retryable (No Retry, No Fallback)

| Category | Examples |
|----------|----------|
| Authentication | HTTP 401, invalid API key, `ProviderAuthenticationError` |
| Invalid Request | HTTP 400, malformed, invalid parameters |
| Not Found | HTTP 404, model not found, decommissioned |
| Forbidden | HTTP 403 |
| Validation | Invalid model, unsupported parameters |

### Classification Logic

- Checks exception type hierarchy (`ProviderAuthenticationError`, etc.)
- HTTP status codes from `httpx.HTTPStatusError`
- Keyword matching in error messages for generic `LLMError`
- **`ProviderFallbackExhaustedError` is now retryable** (enables fallback retry)

---

## 6. Timeout & Deadline Awareness

### Overall Time Budget

- Each provider gets `RETRY_TIMEOUT_SECONDS` total budget
- Budget checked before each attempt
- Backoff capped by remaining time
- No sleep beyond remaining deadline

### Behavior

```
Attempt 1 starts at t=0
  → Fails at t=0.5s
  → Backoff = 1.0s, Remaining = 29.5s → Sleep 1.0s

Attempt 2 starts at t=1.5s
  → Fails at t=2.0s
  → Backoff = 2.0s, Remaining = 28.0s → Sleep 2.0s

Attempt 3 starts at t=4.0s
  → Fails at t=4.5s
  → All attempts exhausted → Raise
```

---

## 7. SDK Retry Interaction

### Known SDK Behaviors

| Provider | SDK | Built-in Retries |
|----------|-----|------------------|
| Gemini | `google-genai` | Yes (tenacity-based, 3 retries default) |
| Groq | `groq` | Yes (httpx default, limited) |
| Ollama | `ollama` | No |

### Integration Strategy

Our application-level retries **wrap** SDK calls. SDK retries happen *inside* each attempt:

```
Our Attempt 1
  └─ SDK Retry 1
  └─ SDK Retry 2
  └─ SDK Retry 3 → Fails → Our Retry Logic kicks in
```

**Total max requests** = Our `max_attempts` × SDK internal retries (if any)

To disable SDK retries, set provider-specific config (not implemented in M4.5.2).

---

## 8. Observability & Logging

### Log Events

| Event | Level | Format |
|-------|-------|--------|
| Primary provider selected | DEBUG | `Using primary provider {name}` |
| Retry triggered | WARNING | `Provider {name}: Attempt {N}/{M} failed ({elapsed}s), retrying in {backoff}s: {error}` |
| Retry success | DEBUG | Implicit via return |
| Retry exhausted | ERROR | `Provider {name}: All {M} attempts exhausted, last error: {error}` |
| Deadline exceeded | WARNING | `Provider {name}: Deadline exceeded after {N} attempts, {elapsed}s elapsed` |
| Retry-After used | WARNING | Implicit in backoff calculation |
| Fallback triggered | WARNING | `Falling back to provider {name} after {N} failures` |
| Fallback exhausted | ERROR | `All providers exhausted, last error: {error}` |

### Never Logged

- ❌ API keys, credentials
- ❌ Authorization headers
- ❌ Full prompts, retrieved documents
- ❌ Sensitive user content

---

## 9. Test Coverage

### New Test File: `tests/unit/test_retry_engine.py`

| Test | Description |
|------|-------------|
| **Backoff** | |
| `test_calculate_backoff_basic` | Exponential progression correct |
| `test_calculate_backoff_max_delay_cap` | Max delay enforced |
| `test_calculate_backoff_retry_after_priority` | Retry-After > calculated |
| `test_calculate_backoff_jitter_enabled` | Jitter adds variance |
| `test_calculate_backoff_retry_after_with_jitter` | Retry-After + jitter |
| **Retry-After** | |
| `test_extract_retry_after_seconds` | Integer seconds parsed |
| `test_extract_retry_after_http_date` | HTTP-date parsed |
| `test_extract_retry_after_malformed` | Malformed handled gracefully |
| `test_extract_retry_after_no_header` | Missing header → None |
| **Retry Behavior** | |
| `test_retry_provider_first_attempt_succeeds` | No retry on success |
| `test_retry_provider_retry_then_succeeds` | Retry then success |
| `test_retry_provider_all_attempts_fail` | All fail → exception |
| `test_retry_provider_non_recoverable_no_retry` | Auth error → no retry |
| `test_retry_provider_disabled` | Disabled → pass through |
| `test_retry_provider_max_attempts_enforced` | 1,2,3,5 attempts enforced |
| `test_retry_provider_no_sleep_after_final_failure` | No sleep after last fail |
| `test_retry_provider_deadline_exceeded` | Timeout stops retries |
| `test_retry_provider_respects_retry_after` | Retry-After honored |
| **Integration** | |
| `test_retry_policy_from_settings` | Settings → policy mapping |
| `test_retry_with_fallback_integration` | Primary retries, then fallback |
| `test_retry_with_fallback_both_retry` | Each provider gets own budget |
| `test_fallback_provider_permanent_error_no_fallback` | Auth → no fallback |
| **Model ID** | |
| `test_retry_provider_model_id` | Exposes primary model_id |
| `test_fallback_provider_model_id` | Exposes primary model_id |
| **Cache** | |
| `test_clear_provider_cache` | Cache clearing |
| `test_get_chat_provider_with_retry_config` | Settings respected |

**Test Results:** 26/26 passed

### Total Test Count
- **Before M4.5.2:** 481 tests
- **After M4.5.2:** 507 tests (+26 new)
- **All tests pass:** ✅ 507 passed

---

## 10. Regression Verification

| Check | Result |
|-------|--------|
| `pytest -q` | ✅ 507 passed |
| `ruff check .` | ✅ All checks passed |
| `ruff format --check .` | ✅ 137 files formatted |
| Hosted API tests | ✅ 9/9 passed |
| Provider fallback tests | ✅ 13/13 passed |
| Retry engine tests | ✅ 26/26 passed |

### Unchanged Artifacts
- `eval/golden/` — Unchanged
- `eval/corpora/` — Unchanged
- M3 baseline runs — Unchanged
- M4.2 chunk-ablation results — Unchanged

---

## 11. Files Modified

| File | Change Type |
|------|-------------|
| `core/config.py` | Added retry configuration fields |
| `core/llm/router.py` | Added `RetryPolicy`, `RetryProvider`, backoff, Retry-After, deadline logic |
| `tests/unit/test_retry_engine.py` | **New** - 26 unit tests |
| `.env.example` | Documented retry configuration |
| `docs/reports/M4_5_2_RETRY_ENGINE.md` | **New** - This documentation |

---

## 12. Known Limitations

1. **SDK retries multiply** — Application retries × SDK internal retries
2. **No per-provider retry config** — Single policy applies to all providers
3. **Sync sleep only** — Async uses `asyncio.sleep`, sync uses `time.sleep`
4. **No metrics** — Retry events not exposed as metrics (M4.5.3)
5. **No circuit breaker** — No automatic provider isolation (M4.5.4)

---

## 13. Verification Results

| Scenario | Status |
|----------|--------|
| First attempt succeeds | ✅ Verified |
| Retry then success | ✅ Verified |
| All attempts fail | ✅ Verified |
| Non-recoverable no retry | ✅ Verified |
| Retry disabled | ✅ Verified |
| Max attempts enforced (1,2,3,5) | ✅ Verified |
| No sleep after final failure | ✅ Verified |
| Deadline exceeded | ✅ Verified |
| Retry-After header respected | ✅ Verified |
| Retry + Fallback integration | ✅ Verified |
| Each provider gets own retry budget | ✅ Verified |
| Permanent error → no fallback | ✅ Verified |
| Backoff formula correct | ✅ Verified |
| Max delay cap enforced | ✅ Verified |
| Jitter within bounds | ✅ Verified |
| Retry-After seconds | ✅ Verified |
| Retry-After HTTP-date | ✅ Verified |
| Deadline stops retries | ✅ Verified |
| No sleep beyond deadline | ✅ Verified |

---

## 14. Conclusion

The intelligent retry engine is **complete and production-ready**. It provides:

- **Exponential backoff** with configurable base, exponent, max delay
- **Jitter** to prevent thundering herd
- **Retry-After** header support (seconds + HTTP-date)
- **Deadline awareness** with time budget
- **Clean separation** from fallback layer
- **Full observability** with structured logging
- **Comprehensive tests** (26 new tests)
- **Full backward compatibility** when disabled

**Next Steps (M4.5.3+):**
- Per-provider retry policies
- Circuit breaker pattern
- Retry metrics/monitoring
- SDK retry configuration

---

*Report generated: 2026-11-05*
*Branch: main*
*Test Count: 507 passed*