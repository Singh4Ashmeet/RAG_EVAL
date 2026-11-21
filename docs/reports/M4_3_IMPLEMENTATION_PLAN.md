# M4.3 Implementation Plan — Hosted Deployment Reliability & Security Hardening

## 1. Architecture Overview

### Key Components
| Component | File | Purpose |
|-----------|------|---------|
| FastAPI Application | `apps/api/main.py` | Main API server with auth, document management, RQ job enqueueing |
| Background Workers | `apps/worker/tasks.py` | RQ workers for indexing, evaluation, experiments |
| LLM Providers | `core/llm/` | Ollama (local), Gemini, Groq (placeholders) |
| Embedding Provider | `core/embed/ollama_embedder.py` | Ollama-based embedding |
| Job Queue | Redis + RQ | Background job processing |
| Database | SQLAlchemy + PostgreSQL/SQLite | Workspace isolation via RLS |
| Storage | Local filesystem + S3-compatible | Original document storage |

### Current Environment Variables (from `.env.example`)
```
DATABASE_URL, CREDENTIAL_ENCRYPTION_KEY, INVITATION_ADMIN_TOKEN
COOKIE_SECURE, PUBLIC_BASE_URL, AUTH_DEV_MODE, RESEND_API_KEY, AUTH_EMAIL_FROM
GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL
GEMINI_CHAT_MODEL, GROQ_CHAT_MODEL
MAX_UPLOAD_BYTES, PRIVATE_DATA_DIR
# Missing: REDIS_URL, S3_*, CORS_ALLOWED_ORIGINS, APP_ENV
```

---

## 2. Files to Modify

### Phase 2: Centralized Configuration & Production Safeguards
| File | Changes |
|------|---------|
| `apps/api/main.py` | Add centralized settings class, environment detection, docs disable in prod |
| `core/config.py` (NEW) | Centralized `Settings` class with validation |
| `core/__init__.py` | Export settings |
| `.env.example` | Add all required variables with placeholders |

### Phase 3: Redis & Enqueue Reliability
| File | Changes |
|------|---------|
| `apps/api/main.py` | Structured enqueue logging, error responses, state consistency |
| `apps/api/main.py` | Redis connection validation on startup |
| `core/redis.py` (NEW) | Redis connection manager with health checks |

### Phase 4: RQ Timeout Configuration
| File | Changes |
|------|---------|
| `apps/worker/tasks.py` | RQ worker timeout configuration via env vars |
| `apps/api/main.py` | Enqueue with explicit timeout parameters |

### Phase 5: Inference Provider Configuration
| File | Changes |
|------|---------|
| `core/llm/router.py` (NEW) | Provider registry with explicit config |
| `core/llm/ollama_provider.py` | Add remote endpoint support |
| `core/llm/gemini_provider.py` | Implement Gemini provider |
| `core/llm/groq_provider.py` | Implement Groq provider |
| `core/embed/ollama_embedder.py` | Support remote Ollama endpoints |
| `apps/api/main.py` | Use provider registry |
| `apps/worker/tasks.py` | Use provider registry |

### Phase 6: S3 Storage Abstraction
| File | Changes |
|------|---------|
| `core/storage/s3.py` (NEW) | S3-compatible storage interface |
| `core/storage/__init__.py` | Export storage interface |
| `apps/api/main.py` | Use storage abstraction |

### Phase 7: Production Docs & Dev-Mode Controls
| File | Changes |
|------|---------|
| `apps/api/main.py` | Conditionally disable `/docs`, `/redoc`, `/openapi.json` |
| `core/config.py` | Add `APP_ENV` validation |

### Phase 8: Health Checks
| File | Changes |
|------|---------|
| `apps/api/main.py` | Enhanced `/health` (liveness) + `/ready` (readiness) |

### Tests
| File | Changes |
|------|---------|
| `tests/unit/test_config.py` (NEW) | Settings validation tests |
| `tests/unit/test_redis.py` (NEW) | Redis connection/enqueue tests |
| `tests/unit/test_rq_timeout.py` (NEW) | Timeout configuration tests |
| `tests/unit/test_providers.py` (NEW) | Provider registry tests |
| `tests/unit/test_s3_storage.py` (NEW) | S3 storage abstraction tests |
| `tests/unit/test_production_mode.py` (NEW) | Production/dev mode tests |
| `tests/unit/test_health.py` (NEW) | Health check tests |

---

## 3. Configuration Strategy

### New Environment Variables (add to `.env.example`)

```bash
# ==============================================================================
# Application Environment
# ==============================================================================
APP_ENV=development|production|test
APP_DEBUG=false

# ==============================================================================
# Redis / Queue
# ==============================================================================
REDIS_URL=redis://localhost:6379/0
RQ_JOB_TIMEOUT=600
RQ_RESULT_TTL=86400
RQ_FAILURE_TTL=604800

# ==============================================================================
# Inference Providers
# ==============================================================================
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_API_KEY=

CHAT_PROVIDER=ollama
CHAT_MODEL=llama3.2:3b
CHAT_BASE_URL=http://localhost:11434
CHAT_API_KEY=

GEMINI_CHAT_MODEL=gemini-3.8-flash
GROQ_CHAT_MODEL=openai/gpt-oss-120b

# ==============================================================================
# S3-Compatible Object Storage
# ==============================================================================
STORAGE_BACKEND=local
S3_BUCKET=
S3_REGION=us-east-1
S3_ENDPOINT_URL=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_PREFIX=rag-eval/

# ==============================================================================
# CORS & Security
# ==============================================================================
CORS_ALLOWED_ORIGINS=
```

---

## 4. Experimental Isolation Strategy

All M4.3 changes are production infrastructure. No evaluation artifacts are affected. Existing M3/M4.2 runs in `eval/runs/` remain untouched. Tests use isolated fixtures with mocked Redis/S3.

---

## 5. Test Plan

### New Test Modules
1. `tests/unit/test_config.py` - Settings validation, environment detection
2. `tests/unit/test_redis.py` - Connection pooling, health checks, enqueue logging
3. `tests/unit/test_rq_timeout.py` - Worker timeout configuration
4. `tests/unit/test_providers.py` - Provider registry, remote endpoints
4. `tests/unit/test_s3_storage.py` - S3 upload/download with moto mock
5. `tests/unit/test_production_mode.py` - Docs disabled/enabled, route guards
6. `tests/unit/test_health.py` - Liveness/readiness behavior

### Existing Tests Must Pass
- All 468 existing tests must continue passing
- No weakening of assertions
- Integration tests must work with mocked dependencies

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Redis unavailable in CI | Use fakeredis or mock for unit tests; document integration test requirements |
| S3 credentials not available | Use `moto` mock for tests; document live verification steps |
| Hosted model endpoints unavailable | Mock provider clients; document deployment verification steps |
| Breaking existing defaults | All new env vars have safe defaults matching current behavior |
| Silent fallback to localhost | Validate `REDIS_URL` and inference URLs in production mode; fail fast |
| Secret leakage in logs | Structured logging with explicit redaction; never log full connection strings |
| Docs exposure in prod | Conditional FastAPI initialization based on `APP_ENV` |

---

## 4. Implementation Sequence

1. **Phase 1**: Repository inspection & implementation plan (this document)
2. **Phase 2**: Centralized config (`core/config.py`), `APP_ENV` detection, production safeguards
3. **Phase 3**: Redis connection manager, structured enqueue logging, state consistency
4. **Phase 4**: RQ `job_timeout`, result/retention TTLs via env vars
5. **Phase 5**: Provider registry, implement Gemini/Groq, remote Ollama support
6. **Phase 6**: S3-compatible storage abstraction with local fallback
7. **Phase 7**: Conditional API docs, dev-mode route guards, `APP_ENV` validation
6. **Phase 8**: Enhanced health checks (liveness/readiness)
7. **Phase 9**: Full test run, regression verification, final deliverables

---

## 5. Verification Checklist

After implementation:
- [ ] `uv run pytest -q` passes (468+ tests)
- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] Production mode: `/docs`, `/redoc`, `/openapi.json` return 404
- [ ] Development mode: `/docs`, `/redoc`, `/openapi.json` accessible
- [ ] `APP_ENV=production` fails fast on missing `REDIS_URL`, `CREDENTIAL_ENCRYPTION_KEY`
- [ ] `APP_ENV=development` allows localhost defaults
- [ ] Redis enqueue logs structured errors with doc ID, job ID, operation type
- [ ] Failed enqueue leaves document/job in `failed` status
- [ ] RQ worker respects `RQ_JOB_TIMEOUT` env var
- [ ] Provider registry selects correct provider from env vars
- [ ] S3 storage uploads/retrieves with mocked client
- [ ] `/health` returns liveness, `/ready` checks Redis/DB/storage
- [ ] `.env.example` documents all required variables

---

**Next Step**: Begin Phase 2 implementation — centralized configuration and production safeguards.