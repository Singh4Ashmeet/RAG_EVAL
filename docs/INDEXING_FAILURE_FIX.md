# Document Indexing Failure: Root Cause Analysis and Remediation

## 1. Executive Summary

A critical indexing failure occurred during document ingestion where documents were stored and their 37 source elements successfully extracted, but indexing subsequently failed, leaving the document in status `failed` with the UI notice:
`"Document stored with failed indexing status. 37 source elements are ready for dataset labels."`

This document details the complete forensic reproduction, root cause analysis across SQLite schema drift and timezone mismatch, the code and schema remediation, and verification ensuring production stability.

---

## 2. Ingestion Lifecycle Trace

The intended ingestion and indexing lifecycle proceeds as follows:

```text
POST /api/v1/documents
       ↓
Source File Persistence (Disk/S3)
       ↓
Document Record Creation (`status = "queued"`)
       ↓
Element Extraction (`NormDocument` -> 37 `CorpusElement` records)
       ↓
Job Enqueued (`enqueue_job` -> `kind = "index"`)
       ↓
Worker Claim (`claim_next_job` -> lease token & attempt recorded)
       ↓
Embedding Generation (Ollama `POST /api/embed` -> 37 vectors)
       ↓
Fenced DB Transaction (Phase C: updates `embedding_json` + `Document.status = "indexed"`)
       ↓
Job Completion (`complete_job` validates lease & closes attempt)
       ↓
Document is Retrievable via `/api/v1/ask` (Dense + BM25 RRF)
```

### Stage Diagnostics Before Fix:
- **POST /documents & Extraction**: Succeeded. 37 source elements extracted from `docs/PHASE_18_VALIDATION.md` and staged for persistence.
- **Job Enqueue (`enqueue_job`)**: Failed on stale SQLite installations where the `jobs` table was missing columns (`sqlite3.OperationalError: table jobs has no column named payload_json`).
- **Worker Execution (`index_document`)**: When jobs were claimed, Phase B (Ollama embeddings) succeeded and returned 37 embeddings (768 dimensions each). In Phase C, after updating `CorpusElement.embedding_json` and `Document.status = "indexed"`, the transaction called `complete_job()`.
- **Job Completion (`complete_job`)**: Failed with `TypeError: can't compare offset-naive and offset-aware datetimes` at line 398 because SQLite stores timestamps without timezone offsets, returning naive datetimes, while `db_time` is UTC-aware.
- **Worker Error Handling (`fail_job`)**: The worker caught the exception and invoked `fail_job()`, which crashed at line 480 with the identical `TypeError: can't compare offset-naive and offset-aware datetimes`.
- **Reaper / Terminal State**: The lease expired unrenewed; the reaper (`reap_expired_jobs()`) synchronized domain entities and set `Document.status = "failed"`.
- **UI State**: The UI reported 37 elements ready for labels, but document status was `failed`.

---

## 3. Root Cause Analysis

Two interacting defects produced the failure:

### A. Missing Column Drift on Existing SQLite Databases
While `20261109_0003_durable_job_queue.py` added queue columns to PostgreSQL and fresh databases, SQLite instances created prior to this migration lacked 14 columns on the `jobs` table (`payload_json`, `lease_token`, `lease_generation`, `lease_expires_at`, `cycle_attempts`, `cumulative_attempts`, `max_attempts`, `manual_retry_count`, `scheduled_at`, `last_error`, `started_at`, `updated_at`, `completed_at`, `worker_id`).
Because `Base.metadata.create_all()` uses `CREATE TABLE IF NOT EXISTS`, existing tables were not updated upon application startup.

### B. Offset-Naive vs Offset-Aware Datetime Comparison
SQLite does not preserve Python `timezone.utc` objects and returns naive `datetime` instances (`tzinfo is None`). In `apps/worker/pg_queue.py`:
- `complete_job`:
  ```python
  if job.lease_expires_at is not None and job.lease_expires_at <= db_time:
  ```
- `fail_job`:
  ```python
  if job.lease_expires_at is not None and job.lease_expires_at <= db_time:
  ```
Comparing `job.lease_expires_at` (offset-naive) to `db_time = datetime.now(UTC)` (offset-aware) raised `TypeError`. This rolled back Phase C of `index_document`, discarding the vectors and leaving `Document.status` uncommitted.

---

## 4. Remediation Implemented

### 1. Timezone-Safe Lease Validation
In `apps/worker/pg_queue.py` (`complete_job` and `fail_job`), normalized datetime comparisons:
```python
lease_expires = job.lease_expires_at
if lease_expires is not None:
    if lease_expires.tzinfo is None and db_time.tzinfo is not None:
        lease_expires = lease_expires.replace(tzinfo=UTC)
    elif lease_expires.tzinfo is not None and db_time.tzinfo is None:
        db_time = db_time.replace(tzinfo=UTC)
    if lease_expires <= db_time:
        db.commit()
        return False
```

### 2. SQLite Schema Column Synchronization
In `apps/api/main.py`, introduced `_sync_sqlite_columns(eng, metadata)` executed upon application startup when using SQLite (`engine.dialect.name == "sqlite"`). It inspects existing tables and automatically adds any newly defined model columns via `ALTER TABLE ... ADD COLUMN` with appropriate default values.

---

## 5. Verification & Test Evidence

### 1. Unit & Regression Tests (`tests/unit/test_indexing_lifecycle.py`)
- `test_sqlite_schema_synchronization`: Verified `_sync_sqlite_columns` accurately detects missing columns and applies non-destructive DDL migrations.
- `test_offset_naive_datetime_handling_complete_and_fail`: Verified that naive datetimes from SQLite do not throw `TypeError` in either `complete_job` or `fail_job`.
- `test_full_indexing_lifecycle_success`: Tested full 37-element document lifecycle (upload, claim, 37 vectors embedded, `indexed` status reached).
- `test_reindex_failed_document_success`: Verified that a document in `failed` status can be re-indexed and successfully completes.
- `test_api_upload_and_ask_e2e`: Verified FastAPI `POST /api/v1/documents`, background queue processing, and status transition.

All 5 tests pass:
```text
tests/unit/test_indexing_lifecycle.py ..... [100%]
5 passed in 6.00s
```

### 2. Corpus Recovery Tests (`tests/unit/test_corpus_recovery.py`)
All 10 tests pass:
```text
tests/unit/test_corpus_recovery.py .......... [100%]
10 passed in 5.20s
```

### 3. Production Smoke Test (`python main.py --smoke-test`)
```text
[SMOKE TEST] Initializing production RAG smoke test...
[SMOKE TEST] All production pipeline integration checks PASSED.
```

### 4. Phase 19 Parent-Mapping Invariance
```text
tests/unit/test_phase19_parent_mapping.py ........ [100%]
8 passed in 1.29s
```

### 5. Quality & Formatting
- `ruff check apps/ tests/unit/test_indexing_lifecycle.py`: All checks passed.
- `ruff format --check apps/ tests/unit/test_indexing_lifecycle.py`: Formatted cleanly.
- `mypy core eval`: Success: no issues found in 102 source files.

---

## 6. Phase 19 Non-Regression Audit

Phase 19 introduced `core/retrieval/parent_mapping.py` for multi-granularity retrieval forensics. Audit confirms:
1. `parent_mapping.py` does not interface with or alter `apps/api/main.py`, `apps/worker/tasks.py`, or `apps/worker/pg_queue.py`.
2. Document models, Job queue mechanisms, and CorpusElement schemas were unaffected by Phase 19.
3. The indexing bug was strictly localized to worker lease comparison timezone handling and SQLite schema synchronization.
4. All Phase 19 parent mapping tests continue to pass without modifications.
