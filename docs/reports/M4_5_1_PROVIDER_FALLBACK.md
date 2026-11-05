# M4.5.1 — Intelligent LLM Provider Fallback

**Date:** 2026-11-05
**Status:** Complete ✅
**Branch:** main
**Commit:** To be committed

---

## 1. Executive Summary

This milestone implements a configurable, intelligent fallback mechanism for LLM providers in RAG_EVAL. When the primary LLM provider fails with a recoverable error, the system automatically attempts generation with the next configured fallback provider, preserving the full RAG pipeline behavior.

**Key Achievement:** All three providers (Ollama, Gemini, Groq) now support automatic fallback without modifying the RAG pipeline, API response schema, or citation behavior.

---

## 2. Architecture Analysis

### Before Implementation

| Component | Role |
|-----------|------|
| `core/llm/router.py` | Provider registry with caching (`get_chat_provider`, `get_embedding_provider`) |
| `core/config.py` | Centralized Pydantic Settings with provider-specific fields |
| Providers | `OllamaLLMProvider`, `GeminiLLMProvider`, `GroqLLMProvider` implementing `LLMProvider` protocol |
| Error Handling | Provider-specific exceptions: `ProviderAuthenticationError`, `ProviderRateLimitError`, `ProviderUnavailableError`, `ProviderResponseError` |

### After Implementation

| Component | Changes |
|-----------|---------|
| `core/config.py` | Added `FALLBACK_ENABLED`, `FALLBACK_PROVIDERS` settings |
| `core/llm/router.py` | Added `FallbackProvider` wrapper, failure classification, fallback logic in `get_chat_provider` |
| `core/llm/router.py` | Added `_is_recoverable_error()` for failure classification |
| `tests/unit/test_provider_fallback.py` | New test file with 13 unit tests |

---

## 3. Configuration Variables

### New Environment Variables (in `.env.example`)

```bash
# Provider Fallback Configuration
# Enable automatic fallback to alternative providers on failure
FALLBACK_ENABLED=true

# Comma-separated list of fallback providers in priority order
# Example: gemini,groq,ollama
FALLBACK_PROVIDERS=gemini,groq,ollama
```

### Settings Class Additions (`core/config.py`)

```python
fallback_enabled: bool = Field(
    default=False,
    validation_alias="FALLBACK_ENABLED",
    description="Enable automatic fallback to alternative providers on failure",
)
fallback_providers: str = Field(
    default="",
    validation_alias="FALLBACK_PROVIDERS",
    description="Comma-separated list of fallback providers in priority order (e.g., 'gemini,groq,ollama')",
)
```

### Configuration Examples

**Example 1: Gemini primary, Groq fallback, Ollama last resort**
```bash
CHAT_PROVIDER=gemini
FALLBACK_ENABLED=true
FALLBACK_PROVIDERS=groq,ollama
```

**Example 2: Groq primary, Ollama fallback**
```bash
CHAT_PROVIDER=groq
FALLBACK_ENABLED=true
FALLBACK_PROVIDERS=ollama
```

**Example 3: Disable fallback (default behavior)**
```bash
CHAT_PROVIDER=gemini
FALLBACK_ENABLED=false
```

---

## 4. Fallback Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Request                           │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Resolve Primary Provider                           │
│  (from CHAT_PROVIDER or explicit provider_name)                 │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Execute Generation                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
               ┌──────────┴──────────┐
               ▼                     ▼
         Success                   Failure
               │                     ▼
               ▼              ┌────────────────────┐
         Return Response      │ Classify Failure   │
                              └────────┬───────────┘
                                       ▼
                          ┌────────────┴────────────┐
                          ▼                         ▼
                    Recoverable              Non-recoverable
                          │                         ▼
                          ▼                  Raise Original Error
                    ┌───────┴───────┐
                    ▼               ▼
            Fallback Enabled?    Fallback Disabled?
                    │               │
                    ▼               ▼
              Yes / No          Raise Original Error
                    │
                    ▼
            ┌───────┴───────┐
            ▼               ▼
        Has Fallbacks?   Raise Original Error
            │
            ▼
        Try Next Fallback
            │
            ▼
        ┌───┴───┐
        ▼       ▼
      Success  Exhausted
        │       │
        ▼       ▼
     Return   Raise
     Response ProviderFallbackExhaustedError
```

### Key Behaviors

1. **Primary provider never retried as fallback** - Filtered from fallback list
2. **No duplicate provider attempts** - Each provider tried at most once
3. **No recursive fallback** - One attempt per provider
4. **Original prompt/context preserved** - Same prompts passed to fallbacks
5. **Citation instructions preserved** - Unchanged through fallback chain
6. **Response structure unchanged** - Compatible with existing API schema

---

## 5. Failure Classification

The `_is_recoverable_error()` function classifies failures:

### Recoverable (Triggers Fallback)
| Error Type | Examples |
|------------|----------|
| Provider Unavailable | 503, 502, 504 errors |
| Rate Limit | 429 with Retry-After |
| Connection Errors | Timeout, DNS failure, connection refused |
| 5xx Server Errors | Any 5xx HTTP status |

### Non-Recoverable (No Fallback)
| Error Type | Examples |
|------------|----------|
| Authentication | 401, invalid API key, unauthorized |
| Invalid Model | 404 model not found, decommissioned |
| Invalid Request | 400 malformed request, invalid parameters |
| Configuration | Missing API key, unsupported provider |

### Classification Logic
- Checks exception type (`ProviderAuthenticationError`, `ProviderRateLimitError`, etc.)
- Checks HTTP status codes (401=non-recoverable, 429/5xx=recoverable)
- Keyword matching in error messages for generic `LLMError`
- Respects provider-specific error subclasses

---

## 6. Observability & Logging

### Log Levels
| Event | Level | Message |
|-------|-------|---------|
| Primary provider selected | DEBUG | `Using primary provider {name}` |
| Fallback triggered | WARNING | `Falling back to provider {name} after {N} failure(s): {details}` |
| Fallback success | INFO | Implicit via successful return |
| Fallback exhausted | ERROR | `All providers exhausted, last error: {error}` |
| Fallback provider unavailable | WARNING | `Fallback provider {name} unavailable: {reason}` |

### Never Logged
- ❌ API keys or credentials
- ❌ Authorization headers
- ❌ Full user prompts or retrieved documents
- ❌ Sensitive environment variables

---

## 7. Test Coverage

### New Test File: `tests/unit/test_provider_fallback.py`

| Test | Description |
|------|-------------|
| `test_parse_fallback_providers` | Parses comma-separated fallback strings |
| `test_is_recoverable_error_recoverable` | 503, 429, timeout, connection errors → recoverable |
| `test_is_recoverable_error_non_recoverable` | 401, 400, 404, auth keywords → non-recoverable |
| `test_fallback_provider_primary_succeeds` | Primary returns response, no fallback |
| `test_fallback_provider_primary_fails_fallback_succeeds` | Primary fails, fallback returns |
| `test_fallback_provider_all_fail` | All providers fail → `ProviderFallbackExhaustedError` |
| `test_fallback_provider_non_recoverable_error_no_fallback` | Auth error → no fallback |
| `test_fallback_provider_preserves_original_exception` | `__cause__` chain preserved |
| `test_fallback_provider_model_id` | Exposes primary `model_id` |
| `test_fallback_provider_duplicate_primary_not_in_fallback` | Primary not called twice |
| `test_clear_provider_cache` | Cache clearing works |
| `test_get_chat_provider_caching` | Registry caching preserved |
| `test_provider_fallback_exhausted_error_stores_failed` | Error stores failed providers list |

**Test Results:** 13/13 passed

### Total Test Count
- **Before:** 468 tests
- **After:** 481 tests (+13 new)
- **All tests pass:** ✅ 481 passed

---

## 8. Regression Verification

| Check | Result |
|-------|--------|
| `pytest -q` | ✅ 481 passed |
| `ruff check .` | ✅ All checks passed |
| `ruff format --check .` | ✅ 135 files formatted |
| Hosted API tests | ✅ 9/9 passed |
| Provider fallback tests | ✅ 13/13 passed |

### Unchanged Artifacts
- `eval/golden/` — Unchanged
- `eval/corpora/` — Unchanged
- M3 baseline runs — Unchanged
- M4.2 chunk-ablation results — Unchanged

---

## 9. Files Modified

| File | Change Type |
|------|-------------|
| `core/config.py` | Added `FALLBACK_ENABLED`, `FALLBACK_PROVIDERS` fields |
| `core/llm/router.py` | Added `FallbackProvider`, `_is_recoverable_error`, fallback logic in `get_chat_provider` |
| `tests/unit/test_provider_fallback.py` | **New** - 13 unit tests |
| `.env.example` | Added fallback configuration documentation |
| `docs/reports/M4_5_1_PROVIDER_FALLBACK.md` | **New** - This documentation |

---

## 10. Known Limitations

1. **One attempt per provider** — No retry loops or exponential backoff (M4.5.2)
2. **Sync fallback only** — Async fallback uses `generate_async` if available, else falls back to sync
3. **Configuration-only** — Fallback order cannot be changed at runtime without restart
4. **Embedding providers** — Fallback only for chat providers (embedding remains Ollama-only)
5. **No metrics** — Fallback events not yet exposed as metrics (M4.5.3)

---

## 10. Verification Results

| Scenario | Status |
|----------|--------|
| Primary succeeds, no fallback | ✅ Verified |
| Primary fails (recoverable), fallback succeeds | ✅ Verified |
| Primary fails (non-recoverable), no fallback | ✅ Verified |
| All providers fail | ✅ Verified (raises `ProviderFallbackExhaustedError`) |
| Fallback disabled | ✅ Verified (original behavior preserved) |
| Authentication error → no fallback | ✅ Verified |
| Invalid request → no fallback | ✅ Verified |
| Duplicate providers not attempted | ✅ Verified |
| Original prompts/context preserved | ✅ Verified |
| Response schema unchanged | ✅ Verified |
| Provider caching works | ✅ Verified |
| Backward compatibility | ✅ Verified |

---

## 11. Conclusion

The intelligent provider fallback mechanism is **complete and production-ready**. It integrates seamlessly with the existing provider registry, preserves all RAG pipeline behavior, and provides comprehensive observability. The implementation follows the existing architecture patterns and maintains full backward compatibility when fallback is disabled.

**Next Steps (M4.5.2):**
- Add retry logic with exponential backoff per provider
- Add per-provider max retry configuration
- Add fallback metrics/monitoring

---

*Report generated: 2026-11-05*
*Branch: main*
*Test Count: 481 passed*