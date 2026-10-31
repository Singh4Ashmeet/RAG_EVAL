# Hosted beta deployment

The hosted beta is a single-region, invite-only deployment. It uses a same-origin FastAPI API and React client, PostgreSQL for tenant records, Redis/RQ for long jobs, and private S3-compatible storage for uploaded originals. Ollama supplies embeddings and the default chat model. Tenant Gemini or Groq keys are reserved for chat.

## Configuration

Copy the variable names below into the deployment secret manager. Do not commit actual values.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL SQLAlchemy URL (`postgresql+psycopg://…`) |
| `REDIS_URL` | Redis connection for RQ |
| `S3_BUCKET`, `S3_ENDPOINT_URL` | Private object storage bucket and optional S3-compatible endpoint |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Narrowly scoped bucket credentials |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | GitHub OAuth app credentials; callback `/api/v1/auth/callback` |
| `CREDENTIAL_ENCRYPTION_KEY` | Fernet key generated with `uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `INVITATION_ADMIN_TOKEN` | Secret for the invitation management API |
| `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_CHAT_MODEL` | Server-managed model service and preloaded models |
| `GEMINI_CHAT_MODEL`, `GROQ_CHAT_MODEL` | Chat model IDs used with tenant-supplied provider keys |
| `COOKIE_SECURE` | Keep `true` behind HTTPS |
| `MAX_UPLOAD_BYTES` | Upload ceiling; defaults to 25 MiB |

Copy `deploy/.env.hosted.example` to `.env`, generate the secrets, and set the GitHub OAuth callback to `https://<host>/api/v1/auth/callback`. `docker compose --env-file .env -f deploy/docker-compose.yml up --build` starts a local reference stack and provisions its private MinIO bucket. Adapt the compose file to the hosting provider's TLS, secret manager, backups, alerting, and object lifecycle controls.

## Current beta behavior and operational limits

- GitHub sign-in requires a verified primary email that an administrator has invited. OAuth asks only for profile and email identity access.
- Each invited account is a private workspace. PostgreSQL row-level security policies use a transaction-local workspace ID for tenant tables; the API and worker also include explicit workspace predicates. Hosted containers run versioned Alembic migrations before serving. Local development uses SQLAlchemy `create_all`. The initial migration has been verified on SQLite; validate it against the target managed PostgreSQL service before inviting users.
- `.txt` and `.md` originals are private objects and are removed when their document or workspace is deleted. Existing evaluation reports retain retrieved excerpts when an individual document is deleted. Workspace deletion removes active workspace data. Backup encryption and 30-day expiry must be configured and monitored in the hosting platform.
- Upload, evaluation, and experiment jobs persist queued/running/complete/partial/failed states. Ollama must be reachable from both API and worker. Current indexing embeds stable source elements; chunked indexing and pgvector are still needed before large corpora.
- Ask uses exact cosine over workspace source-element vectors. Ollama is the managed default; Gemini and Groq keys are encrypted and can be checked before use for chat. Selecting a tenant provider sends the question and retrieved excerpts to that provider. Embeddings always use server Ollama.
- Dataset revisions store canonical JSON and a digest. Source-element references are checked against the current workspace. Retrieval runs preserve per-query evidence, answer metrics, status, and denominators. Experiments compare named retrieval-depth variants on the same revision and report paired deltas and bootstrap intervals.
- Invitation listing, creation, and revocation are protected by `INVITATION_ADMIN_TOKEN` on the `/api/v1/invitations` API and the `/admin/invitations` owner page. The page keeps the token in memory for that tab only.
- Credential encryption keys must be backed up separately from the database. Rotating or losing the key requires a credential re-encryption plan.
- Same-origin checks protect browser mutations; OAuth callback also validates state. Rate limiting, export, automated backup expiry, credential-key rotation, comprehensive audit logging, and independent security review remain required operational work before broad access.

## API contract

The live FastAPI OpenAPI schema is available at `/openapi.json`; `/api/v1/openapi.json` is an alias. Generate TypeScript types from a running API with `npm run api:types` from `frontend/`.
