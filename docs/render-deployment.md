# Deploy RAG_EVAL to Render

The repository includes a Render Blueprint in [`render.yaml`](../render.yaml). It creates a **single web service** that runs both the FastAPI API and the in-process job processor (worker). The web service runs the API and starts the `JobProcessor` through FastAPI's lifespan. Render's generated `*.onrender.com` address serves both the frontend and API from one origin. Render can wait for GitHub checks to pass before deploying subsequent commits.

**Supabase PostgreSQL is external to Render** — the Blueprint does not provision a managed database. You must provision Supabase PostgreSQL separately and configure `DATABASE_URL` and `WORKER_DATABASE_URL` environment variables.

**No Redis/Key Value dependency** — the durable queue uses PostgreSQL for job persistence and does not require Redis.

## External services required

Render does not provision the app's S3-compatible object bucket, embedding endpoint, or optional Ollama chat endpoint. Configure these before using uploads or asking/indexing documents:

- A private AWS S3 bucket (or S3-compatible service) and a narrowly scoped access key. The app stores uploaded originals in this bucket. Configure via `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
- An embedding endpoint reachable from the Render web service. For Cloudflare Workers AI, set `EMBEDDING_PROVIDER=openai_compatible`, use the account-scoped HTTPS `/ai/v1` base URL in `EMBEDDING_BASE_URL`, select `@cf/baai/bge-small-en-v1.5`, and store its API token only in `EMBEDDING_API_KEY` in Render. Keep endpoint credentials restricted to the app.
- If using Ollama for chat, an Ollama HTTP endpoint reachable from Render. Configure `OLLAMA_BASE_URL` and `CHAT_BASE_URL` with its remote HTTPS base URL; localhost and the developer's PC are not reachable from Render.

## Render Free tier limitations

- **Free web services spin down after inactivity** — the in-process `JobProcessor` only runs while the web service is awake. Durable jobs, leases, and the reaper support recovery after restart, but **continuous background processing is not available on the Free tier**.
- For production workloads requiring continuous background processing, use a paid Render web service plan with "Always On" or provision a separate worker (requires architecture changes).
- Durable jobs, leases, and reaping support recovery after spin-down/restart.

## Create the Render services

1. Push the repository to GitHub and confirm the latest CI run on `main` succeeds.
2. Provision **Supabase PostgreSQL** (or equivalent PostgreSQL 17+) separately. Create two database roles:
   - `rag_api` (for API operations)
   - `rag_worker` (for worker operations, distinct from `rag_api`)
   Run the bootstrap SQL (`deploy/sql/01_worker_roles_and_privileges.sql`) as `postgres` superuser, then run the Alembic migration (`migrations/versions/20261109_0003_durable_job_queue.py`).
2. In Render, create a **Blueprint** and connect the repository, branch `main`, using the repository's `render.yaml`.
3. Select a plan suitable for the web service. For production, use a paid plan with "Always On" to avoid spin-down.
4. During Blueprint setup, enter the requested environment variables (see below). Render generates the invitation-admin and credential-encryption secrets. Save the credential-encryption key in a separately protected backup location; losing it makes saved provider credentials unreadable.
5. Wait for the initial deployment. The web service runs `alembic upgrade head` before starting the API (via `SKIP_SCHEMA_INIT=false` in non-production; in production, migrations are applied out-of-band). Its health endpoint is `/api/v1/health`.
6. Copy the generated web URL from the Render dashboard, for example `https://rag-eval-web.onrender.com`.

## Required environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_ENV` | Yes | `production` |
| `STORAGE_BACKEND` | Yes | `s3` |
| `DATABASE_URL` | Yes | Connection string for `rag_api` role (production requires distinct credentials) |
| `WORKER_DATABASE_URL` | Yes (production) | Connection string for `rag_worker` role; **must differ from `DATABASE_URL`** |
| `SKIP_RLS_SETUP` | Yes (production) | `true` (RLS is set up via bootstrap SQL) |
| `SKIP_SCHEMA_INIT` | Yes (production) | `true` (migrations run out-of-band) |
| `COOKIE_SECURE` | Yes | `true` (requires HTTPS) |
| `PUBLIC_BASE_URL` | Yes | Public Render origin (e.g., `https://rag-eval-web.onrender.com`) |
| `CREDENTIAL_ENCRYPTION_KEY` | Yes | Fernet key for encrypting provider credentials |
| `INVITATION_ADMIN_TOKEN` | Yes | Admin token for workspace invitations |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Yes | S3-compatible storage configuration |
| `EMBEDDING_PROVIDER` | Yes | `openai_compatible` for a hosted OpenAI-compatible embedding service; `ollama` remains the local-development default |
| `EMBEDDING_BASE_URL` | Yes | HTTPS OpenAI-compatible API root (Cloudflare: account-scoped `/ai/v1` URL) |
| `EMBEDDING_MODEL` | Yes | Cloudflare example: `@cf/baai/bge-small-en-v1.5` |
| `EMBEDDING_API_KEY` | Yes for external embeddings | Store only in Render secret settings |
| `OLLAMA_BASE_URL`, `CHAT_BASE_URL` | Only for Ollama chat | Remote Ollama HTTPS endpoint; never a local machine address |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | Optional | GitHub OAuth credentials |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional | Google OAuth credentials |
| `RESEND_API_KEY`, `AUTH_EMAIL_FROM` | Optional | Email delivery via Resend |

## Configure account sign-in

Set `PUBLIC_BASE_URL` to the exact public Render origin, for example `https://rag-eval-web.onrender.com`.

For Google sign-in, create a **Web application** OAuth client in Google Cloud Console and set its authorized redirect URI to:

```text
https://<your-render-service>.onrender.com/api/v1/auth/google/callback
```

Add the Google client ID and secret to Render as `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.

To retain GitHub sign-in, configure its callback as `https://<your-render-service>.onrender.com/api/v1/auth/callback` and save the credentials as `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`. No repository permissions or personal access token are needed.

For passwordless email sign-in, configure a sender domain with Resend, create an API key, and add `RESEND_API_KEY` and `AUTH_EMAIL_FROM` to Render. The sender must be authorized by Resend for production delivery. Email links expire after 15 minutes and can be used once. Keep all OAuth secrets and email API keys in Render's secret environment settings; do not add them to Git or share them in chat.

Redeploy after setting the environment variables. The sign-in page only displays methods with configured credentials. Provider registration and email delivery remain dependent on correct callback URLs, verified sender domains, and working provider credentials.

## Future deploys

The Blueprint uses Render's `checksPass` auto-deploy behavior. Link the service to this GitHub repository and keep the GitHub Actions checks enabled; Render will deploy changes to `main` after checks pass. The existing workflow also publishes the container image to GHCR. Monitor Render deploy logs and the `/api/v1/health` endpoint after each release.

## Before inviting beta users

- Confirm S3 bucket access, private ACL/block-public-access settings, encryption, and lifecycle/backup policy.
- Confirm Ollama is reachable from the Render web service and both configured model names exist.
- Confirm database migrations and workspace isolation on the deployed PostgreSQL database.
- Set up database backups, alerting, and a 30-day encrypted backup expiry process for deleted workspace data.
- Review the [hosted Terms](hosted-terms.md), [Privacy Notice](hosted-privacy.md), and [beta limitations](hosted-beta.md).
- **Free tier does not provide continuous background processing** — jobs only process while the service is awake. For production, use a paid plan with "Always On".

Render's official references: [Blueprints](https://render.com/docs/blueprint-spec), [Docker services](https://render.com/docs/docker), [PostgreSQL](https://render.com/docs/postgresql-creating-connecting).
