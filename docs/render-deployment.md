# Deploy RAG_EVAL to Render

The repository includes a Render Blueprint in [`render.yaml`](../render.yaml). It creates a web service, background worker, PostgreSQL database, and private Redis-compatible Key Value instance in Render's Singapore region. Render's generated `*.onrender.com` address serves both the frontend and API from one origin. Render can wait for GitHub checks to pass before deploying subsequent commits.

## External services required

Render does not provision the app's S3-compatible object bucket or Ollama model endpoint in this Blueprint. Configure these before using uploads or asking/indexing documents:

- A private AWS S3 bucket (or S3-compatible service) and a narrowly scoped access key. The app stores uploaded originals in this bucket. The Blueprint asks for bucket name, region, access key ID, and secret access key.
- An Ollama HTTP endpoint reachable from both Render services. Configure the full base URL in `OLLAMA_BASE_URL`; the selected models must already be available there: `nomic-embed-text` and `llama3.2:3b` by default. Keep this endpoint private or protected and allow requests only from the app's services where possible.

The Render web and worker plans must have enough memory for this app and workload. Use paid persistent plans for production services, especially the database and Key Value queue. Choose the same Singapore region for the external object store and Ollama host when available.

## Create the Render services

1. Push the repository to GitHub and confirm the latest CI run on `main` succeeds.
2. In Render, create a **Blueprint** and connect `Singh4Ashmeet/RAG_EVAL`, branch `main`, using the repository's `render.yaml`.
3. Select paid plans suitable for the web service, worker, PostgreSQL, and Key Value. Keep Key Value private (the Blueprint sets an empty external IP allow list) and keep its persistence enabled.
4. During Blueprint setup, enter the requested private S3 and Ollama values. Render generates the invitation-admin and credential-encryption secrets. Save the credential-encryption key in a separately protected backup location; losing it makes saved provider credentials unreadable.
5. Wait for the initial deployment. The web service runs `alembic upgrade head` before starting the API. Its health endpoint is `/api/v1/health`.
6. Copy the generated web URL from the Render dashboard, for example `https://rag-eval-web.onrender.com`.

## Configure account sign-in

Set `PUBLIC_BASE_URL` to the exact public Render origin, for example `https://rag-eval-koyb.onrender.com`.

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
- Confirm Ollama is reachable from both the web service and worker and both configured model names exist.
- Confirm database migrations and workspace isolation on the deployed PostgreSQL database.
- Set up database backups, alerting, and a 30-day encrypted backup expiry process for deleted workspace data.
- Review the [hosted Terms](hosted-terms.md), [Privacy Notice](hosted-privacy.md), and [beta limitations](hosted-beta.md).

Render's official references: [Blueprints](https://render.com/docs/blueprint-spec), [Docker services](https://render.com/docs/docker), [PostgreSQL](https://render.com/docs/postgresql-creating-connecting), and [Key Value](https://render.com/docs/key-value).
