# Hosted beta deployment

The hosted beta is a single-region deployment with self-service registration. It uses a same-origin FastAPI API and React client, PostgreSQL for tenant records, Redis/RQ for long jobs, and private S3-compatible storage for uploaded originals. Users can authenticate with a one-time email link, Google, or GitHub. Ollama supplies embeddings and the default chat model. Tenant Gemini or Groq keys are reserved for chat.

## Configuration

Copy the variable names below into the deployment secret manager. Do not commit actual values.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL SQLAlchemy URL (`postgresql+psycopg://…`) |
| `REDIS_URL` | Redis connection for RQ |
| `S3_BUCKET`, `S3_ENDPOINT_URL` | Private object storage bucket and optional S3-compatible endpoint |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Narrowly scoped bucket credentials |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | GitHub OAuth app credentials; callback `/api/v1/auth/callback` |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Google OAuth web client credentials; callback `/api/v1/auth/google/callback` |
| `PUBLIC_BASE_URL` | Canonical HTTPS origin used to form OAuth callback and email links |
| `RESEND_API_KEY`, `AUTH_EMAIL_FROM` | Resend transactional email API key and verified sender for passwordless sign-in links |
| `CREDENTIAL_ENCRYPTION_KEY` | Fernet key generated with `uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `INVITATION_ADMIN_TOKEN` | Secret for the invitation management API |
| `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_CHAT_MODEL` | Server-managed model service and preloaded models |
| `GEMINI_CHAT_MODEL`, `GROQ_CHAT_MODEL` | Chat model IDs used with tenant-supplied provider keys |
| `COOKIE_SECURE` | Keep `true` behind HTTPS |
| `MAX_UPLOAD_BYTES` | Upload ceiling; defaults to 25 MiB |

Copy `deploy/.env.hosted.example` to `.env`, generate the secrets, and set `PUBLIC_BASE_URL` to the canonical HTTPS origin. Register OAuth callbacks at `/api/v1/auth/callback` for GitHub and `/api/v1/auth/google/callback` for Google. `docker compose --env-file .env -f deploy/docker-compose.yml up --build` starts a local reference stack and provisions its private MinIO bucket. Adapt the compose file to the hosting provider's TLS, secret manager, backups, alerting, and object lifecycle controls.

## Current beta behavior and operational limits

- Registration is open. Users can sign in with a single-use, 15-minute email link, or with Google or GitHub. Email links are delivered through Resend; Google and GitHub only provide identity and verified email data.
- Each account is assigned a private workspace. PostgreSQL row-level security policies use a transaction-local workspace ID for tenant tables; the API and worker also include explicit workspace predicates. Hosted containers run versioned Alembic migrations before serving. Local development uses SQLAlchemy `create_all`. Validate migrations against the target managed PostgreSQL service before public use.
- `.txt` and `.md` originals are private objects and are removed when their document or workspace is deleted. Existing evaluation reports retain retrieved excerpts when an individual document is deleted. Workspace deletion removes active workspace data. Backup encryption and 30-day expiry must be configured and monitored in the hosting platform.
- Upload, evaluation, and experiment jobs persist queued/running/complete/partial/failed states. Ollama must be reachable from both API and worker. Current indexing embeds stable source elements; chunked indexing and pgvector are still needed before large corpora.
- Ask uses exact cosine over workspace source-element vectors. Ollama is the managed default; Gemini and Groq keys are encrypted and can be checked before use for chat. Selecting a tenant provider sends the question and retrieved excerpts to that provider. Embeddings always use server Ollama.
- Dataset revisions store canonical JSON and a digest. Source-element references are checked against the current workspace. Retrieval runs preserve per-query evidence, answer metrics, status, and denominators. Experiments compare named retrieval-depth variants on the same revision and report paired deltas and bootstrap intervals.
- The invitation administration API remains available for controlled access if needed, but it is not required for sign-up.
- Credential encryption keys must be backed up separately from the database. Rotating or losing the key requires a credential re-encryption plan.
- Same-origin checks protect browser mutations; OAuth callbacks validate state, and email links are single use and expire after 15 minutes. Email link requests use an account-level cooldown. Distributed abuse rate limiting, export, automated backup expiry, credential-key rotation, comprehensive audit logging, and independent security review remain required operational work before broad access.

## API contract

The live FastAPI OpenAPI schema is available at `/openapi.json`; `/api/v1/openapi.json` is an alias. Generate TypeScript types from a running API with `npm run api:types` from `frontend/`.
