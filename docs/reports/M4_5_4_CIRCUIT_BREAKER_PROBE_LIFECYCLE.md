# M4.5.4 — Circuit Breaker HALF_OPEN Probe Lifecycle Fix

**Date:** 2026-11-05
**Milestone:** M4.5 — Production Reliability & Resilience
**Phase:** M4.5.4
**Fixes:** Audit finding F-01 from M4.5.3 read-only engineering audit; F-03 stale failure timestamp

---

## 1. Root Cause — F-01: HALF_OPEN Permanent Wedge

The M4.5.3 engineering audit identified a critical defect in the circuit breaker's
HALF_OPEN probe lifecycle:

1. `allow_request()` increments `_half_open_calls` when granting a probe.
2. `record_failure()` returns early for non-recoverable exceptions
   (`if not self._is_failure_recoverable(exc): return`).
3. The acquired probe permit is never released.
4. Once `_half_open_calls` reaches `half_open_max_calls` (default: 1), all
   subsequent requests are rejected **permanently**.

**Impact:** A single 400 or 401 response during a HALF_OPEN probe permanently
disables the provider with no recovery path except process restart.

---

## 2. Design Overview

The fix introduces an **epoch-based permit lifecycle** with three key components:

### 2.1 CircuitBreakerPermit (lines 91–136 in router.py)

Each permitted request receives a `CircuitBreakerPermit` object that:

- Captures the breaker's **epoch** at acquisition time.
- Carries a per-permit **threading.Lock** and **resolved flag**.
- Exposes exactly three resolution methods:
  - `record_success()` — probe succeeded.
  - `record_failure(exc)` — probe failed with a classifiable exception.
  - `release()` — probe was cancelled, interrupted, or aborted.

**Guarantee:** Each permit resolves exactly once. The first resolution method
acquires the permit's lock, sets `_resolved = True`, and dispatches to the
breaker. Subsequent calls are no-ops.

### 2.2 Epoch-Based Stale Probe Protection

Every state transition increments `CircuitBreaker._epoch`:

| Transition              | Epoch Incremented |
| ----------------------- | :---------------: |
| CLOSED → OPEN           | ✓                 |
| OPEN → HALF_OPEN        | ✓                 |
| HALF_OPEN → OPEN        | ✓                 |
| HALF_OPEN → CLOSED      | ✓                 |
| reset()                 | ✓                 |

When a permit is resolved, the breaker compares `permit.epoch` with
`self._epoch`. If they differ, the resolution is silently discarded. This
prevents a slow or concurrent probe from corrupting a circuit that has already
transitioned to a new state.

### 2.3 Resolution Dispatch

| Resolution Path               | HALF_OPEN Behavior                                     | CLOSED Behavior          |
| ----------------------------- | ------------------------------------------------------ | ------------------------ |
| `_resolve_success(permit)`    | Increment `_success_count`; close circuit when quorum met | Reset `_failure_count`   |
| `_resolve_failure(permit, e)` — recoverable   | Reopen circuit (OPEN), increment epoch | Increment `_failure_count`; open if threshold met |
| `_resolve_failure(permit, e)` — non-recoverable | Decrement `_half_open_calls` (recycle permit) | No health impact |
| `_resolve_release(permit)`    | Decrement `_half_open_calls` if probe permit | No-op |

---

## 3. Permit Acquisition Flow

```
   acquire_permit()
         │
    ┌────▼────┐
    │ Enabled? │──No──► Return permit (is_probe=False)
    └────┬────┘
         │ Yes
    ┌────▼────────────┐
    │ Check/transition │
    │ state under lock │
    └────┬────────────┘
         │
    ┌────▼───────┐
    │   CLOSED   │──► Return permit (is_probe=False, current epoch)
    └────────────┘
    ┌────▼───────┐
    │    OPEN    │──► Return None (rejected)
    └────────────┘
    ┌────▼───────────────────────┐
    │ HALF_OPEN                  │
    │ _half_open_calls < max?    │
    │  Yes → increment, return   │
    │        permit (is_probe=   │
    │        True, current epoch)│
    │  No  → Return None         │
    └────────────────────────────┘
```

---

## 4. Non-Recoverable Exception Handling

Non-recoverable exceptions (HTTP 400, 401, 403, etc.) during a HALF_OPEN probe
are handled in `_resolve_failure()`:

```python
if self._state == CircuitBreakerState.HALF_OPEN:
    if is_recoverable:
        # Reopen: provider still failing
        self._state = CircuitBreakerState.OPEN
        self._epoch += 1
        ...
    else:
        # Non-recoverable: recycle the probe permit
        if self._half_open_calls > 0:
            self._half_open_calls -= 1
```

This ensures:
- The probe slot is **recycled**, not consumed.
- The circuit does **not** reopen (non-recoverable errors are not provider health signals).
- A new probe can be admitted **immediately** if `_half_open_calls < half_open_max_calls`.
  No additional recovery timeout is required because the circuit remains HALF_OPEN.

---

## 5. Cancellation and Interruption Handling

`CircuitBreakerProvider.generate()` catches `BaseException`:

```python
except BaseException as exc:
    if isinstance(exc, Exception):
        permit.record_failure(exc)
    else:
        permit.release()
    raise
```

- **`Exception` subclasses** (provider errors): routed to `record_failure()` for classification.
- **Non-Exception `BaseException`** (`KeyboardInterrupt`, `SystemExit`): routed to `release()`.

`_resolve_release()` decrements `_half_open_calls` for probe permits in the current epoch, without recording a health event.

---

## 6. Concurrent Probe Handling

**Scenario:** Circuit is HALF_OPEN. Probe P1 starts. Circuit reopens due to a
concurrent recoverable failure from another code path. Probe P1 later resolves.

**Protection:** P1's permit captured epoch N. The reopen incremented epoch to N+1.
When P1 resolves, `permit.epoch (N) != self._epoch (N+1)` → resolution is discarded.

**Stale failure timestamp (F-03, fixed in hardening pass):** `_resolve_failure()`
now validates `permit.epoch` before modifying any breaker state. A stale permit
cannot update `_last_failure_time`, `_failure_count`, `_half_open_calls`, or
trigger a state transition.

**Double-resolution:** If a code path accidentally calls both `record_failure()`
and `release()` on the same permit, the per-permit lock ensures only the first
call takes effect. The second is a no-op.

---

## 7. Metrics Semantics

Two independent metrics systems interact during provider request lifecycle:

| Layer | Responsibility |
| --- | --- |
| **Provider request metrics** (`collector.record_success()` / `collector.record_failure()`) | Records every provider request outcome — success or failure — regardless of error recoverability. Called in `CircuitBreakerProvider.generate()`. |
| **Circuit breaker health** (`_failure_count`, `_half_open_calls`, state transitions) | Tracks provider health for circuit state decisions. Only affected by recoverable errors. |

### Per-event behavior:

| Event | Provider request metric | Circuit health effect |
| --- | --- | --- |
| Successful provider response | `record_success(...)` recorded | `_failure_count` reset (CLOSED) or `_success_count` incremented (HALF_OPEN) |
| Non-recoverable provider error (e.g., 401) | `record_failure(...)` recorded | No health impact; probe permit recycled in HALF_OPEN |
| Recoverable provider error (e.g., 503) | `record_failure(...)` recorded | `_failure_count` incremented (CLOSED) or circuit reopened (HALF_OPEN) |
| `KeyboardInterrupt` / `SystemExit` | **No** provider metric recorded | Permit released; no health impact |
| Circuit breaker rejection (OPEN state) | `record_circuit_rejection(...)` recorded | N/A |

Note: A non-recoverable error will still appear in provider request-failure
metrics. The distinction is that it does **not** count as a circuit breaker
health failure and does **not** cause the circuit to reopen.

---

## 8. Backward Compatibility and Legacy API

The public `record_success()` and `record_failure()` methods on `CircuitBreaker`
retain their `permit=None` default parameter:

```python
def record_success(self, permit: CircuitBreakerPermit | None = None) -> None:
    if permit is not None:
        permit.record_success()
        return
    # Legacy fallback path (existing behavior preserved)
    ...
```

Callers that do not use permits continue to work unchanged. The legacy path
preserves the original behavior without epoch checking.

### Legacy API investigation

The boolean-based `allow_request()` API acquires a permit internally but cannot
expose the permit handle to the caller.

**Production callers:** No production code outside `router.py` uses
`allow_request()`. The only production call path is
`CircuitBreakerProvider.generate()`, which uses `acquire_permit()` directly.

**Test callers:** `allow_request()` is used in 18 test assertions for state
checking. These tests follow it with the legacy `record_success()` /
`record_failure()` methods (without permits), which is the intended backward-
compatible code path.

**Permit leak risk:** `allow_request()` acquires a permit and discards it. In
HALF_OPEN state, this consumes a probe slot. However, since no production code
uses `allow_request()`, this is not a production defect. Test code that calls
`allow_request()` in HALF_OPEN should be aware of this side effect.

**Recommendation:** `acquire_permit()` is the preferred API for any new
integration that needs explicit permit lifecycle management. `allow_request()`
should not be used in new code that interacts with HALF_OPEN state.

---

## 9. Files Modified

| File | Change Type | Description |
| --- | --- | --- |
| `core/llm/router.py` | Modified (tracked) | Added `CircuitBreakerPermit` class, epoch counter, `acquire_permit()`, `_resolve_success()`, `_resolve_failure()`, `_resolve_release()`. Updated `CircuitBreakerProvider.generate()` to use permit lifecycle. F-03: moved `_last_failure_time` update after epoch validation. |
| `tests/unit/test_circuit_breaker.py` | New (untracked) | Added 16 new tests for probe lifecycle and stale permit protection, totaling 35 tests. |

No other files were modified.

---

## 10. Tests Added

### M4.5.4 initial implementation (10 tests)

| # | Test Name | Validates |
| --- | --- | --- |
| 1 | `test_probe_non_recoverable_error_does_not_wedge_circuit` | 401 auth error releases probe permit |
| 2 | `test_probe_http_400_error_does_not_increment_health_failures` | HTTP 400 does not count as health failure |
| 3 | `test_probe_unexpected_exception_cleans_up_permit` | RuntimeError in probe releases permit |
| 4 | `test_probe_recoverable_failure_reopens_immediately_and_resets_recovery_timer` | 503 in probe reopens circuit |
| 5 | `test_probe_success_resolves_permit_and_closes_circuit` | Successful probe closes circuit |
| 6 | `test_probe_capacity_enforcement_and_permit_recycling` | half_open_max_calls=2 enforcement and recycling |
| 7 | `test_stale_concurrent_probe_cannot_close_reopened_circuit` | Epoch-based stale detection |
| 8 | `test_permit_double_resolution_protection` | Double-resolution is a no-op |
| 9 | `test_cancellation_interruption_cleanup_does_not_leak_or_misclassify` | KeyboardInterrupt permit release |
| 10 | `test_closed_and_open_regressions_preserved` | Existing CLOSED/OPEN behavior unchanged |

### M4.5.4 hardening — F-03 stale timestamp tests (6 tests)

| # | Test Name | Validates |
| --- | --- | --- |
| 11 | `test_stale_failure_does_not_change_last_failure_time` | Stale permit cannot update `last_failure_time` |
| 12 | `test_stale_failure_does_not_change_failure_count` | Stale permit cannot modify failure counters |
| 13 | `test_stale_failure_does_not_change_half_open_permit_count` | Stale permit cannot modify `_half_open_calls` |
| 14 | `test_current_epoch_recoverable_failure_updates_timestamp_and_reopens` | Current-epoch recoverable failure updates timestamp and reopens circuit |
| 15 | `test_current_epoch_non_recoverable_failure_updates_timestamp_and_recycles` | Current-epoch non-recoverable failure updates timestamp and recycles permit |
| 16 | `test_stale_success_cannot_close_circuit_regression` | Stale success protection regression guard |

---

## 11. Verification Results

```
uv run pytest -v tests/unit/test_circuit_breaker.py
  → 35 passed

uv run pytest -q tests/unit/test_circuit_breaker.py tests/unit/test_provider_metrics.py tests/unit/test_retry_engine.py tests/unit/test_provider_fallback.py
  → 88 passed

uv run pytest -q
  → 556 passed, 0 failed (previous baseline: 550)

uv run ruff check .
  → All checks passed!

uv run ruff format --check .
  → 143 files already formatted

git diff --check
  → Clean (no whitespace errors)
```

---

## 12. F-03 — Stale Failure Timestamp Fix

### Defect

In `_resolve_failure()`, `self._last_failure_time = now` was written **before**
the epoch validation check. A stale permit from a previous circuit cycle could
update the current cycle's failure timestamp, corrupting telemetry.

### Fix

Moved `_last_failure_time = now` after epoch validation:

```python
# Before (defective):
with self._lock:
    now = self._clock()
    self._last_failure_time = now  # Written before epoch check
    if permit.epoch != self._epoch:
        return  # State change rejected, but timestamp already modified

# After (fixed):
with self._lock:
    if permit.epoch != self._epoch:
        return  # Early return — no state modification at all
    now = self._clock()
    self._last_failure_time = now  # Only updated for current-epoch permits
```

Stale permits now cannot modify any breaker state: `_last_failure_time`,
`_failure_count`, `_half_open_calls`, or circuit state.

---

## 13. Remaining Limitations (Inherited from M4.5.3 Audit)

These are **not regressions** — they are known limitations from the M4.5.3 audit
that were out of scope for this fix:

1. **Process-local state:** Circuit breaker state is in-memory. Not shared across
   processes or containers. Acceptable for single-process deployment.

2. **Token gap during HALF_OPEN:** Between permit acquisition and resolution,
   there is a brief window where the probe is in-flight and the breaker cannot
   admit additional probes. This is by design (probe isolation).

3. **`window_seconds` unused:** The `CircuitBreakerPolicy.window_seconds` field
   is defined but not used in failure counting logic. Failure count resets only
   on circuit close. A sliding-window implementation would be a future enhancement.

4. **`allow_request()` consumes probe slot:** The boolean-based `allow_request()`
   API acquires and discards a permit. In HALF_OPEN, this consumes a probe slot
   without providing a handle for resolution. No production code uses this API in
   HALF_OPEN. Use `acquire_permit()` for new integrations.
