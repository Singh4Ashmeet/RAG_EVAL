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

## Configure GitHub sign-in

Create a dedicated GitHub OAuth App for the hosted environment under GitHub **Settings → Developer settings → OAuth Apps**. Set its homepage to the Render web URL and its authorization callback to:

```text
https://<your-render-service>.onrender.com/api/v1/auth/callback
```

After the Render URL exists, add `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to the **web service's Environment** in Render and redeploy. No repository permissions or personal access token are needed; the app requests `read:user` and `user:email`. Use a separate OAuth App for local development if needed.

Open `https://<your-render-service>.onrender.com/admin/invitations`, retrieve `INVITATION_ADMIN_TOKEN` from the web service's Render Environment settings, and invite the verified primary email address associated with the GitHub account. Do not paste secrets into GitHub issues, commits, or logs.

## Future deploys

The Blueprint uses Render's `checksPass` auto-deploy behavior. Link the service to this GitHub repository and keep the GitHub Actions checks enabled; Render will deploy changes to `main` after checks pass. The existing workflow also publishes the container image to GHCR. Monitor Render deploy logs and the `/api/v1/health` endpoint after each release.

## Before inviting beta users

- Confirm S3 bucket access, private ACL/block-public-access settings, encryption, and lifecycle/backup policy.
- Confirm Ollama is reachable from both the web service and worker and both configured model names exist.
- Confirm database migrations and workspace isolation on the deployed PostgreSQL database.
- Set up database backups, alerting, and a 30-day encrypted backup expiry process for deleted workspace data.
- Review the [hosted Terms](hosted-terms.md), [Privacy Notice](hosted-privacy.md), and [beta limitations](hosted-beta.md).

Render's official references: [Blueprints](https://render.com/docs/blueprint-spec), [Docker services](https://render.com/docs/docker), [PostgreSQL](https://render.com/docs/postgresql-creating-connecting), and [Key Value](https://render.com/docs/key-value).
