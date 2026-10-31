# Development runner and CI/CD

## Run the app locally

Install Python 3.12, `uv`, Node.js 20.19+ (or 22.12+), and npm. From the repository root:

```powershell
uv sync
python main.py
```

On systems where `python` maps to another version, use `py -3.12 main.py`. The runner starts the FastAPI backend at `http://127.0.0.1:8000` and the Vite frontend at `http://127.0.0.1:5173`. The first run installs frontend dependencies with `npm ci` if they are missing. Press Ctrl+C to stop both processes. The Vite proxy forwards API and legal-page requests to FastAPI.

You can also invoke each service separately with `uv run uvicorn apps.api.main:app --reload` and `cd frontend; npm run dev`.

## GitHub Actions

`.github/workflows/ci-cd.yml` runs on pull requests to `main`, pushes to `main`, and manual dispatches. It checks Python lint and tests, applies database migrations to a temporary SQLite database, runs frontend unit and browser tests, and builds the production frontend. A push to `main` publishes the Docker image only after all checks pass.

Images are published to `ghcr.io/<owner>/<repository>` with `latest` and commit-specific `sha-…` tags. The workflow does not deploy to a hosting provider. Enable GitHub Actions for the repository and allow the workflow `GITHUB_TOKEN` to write packages. Depending on the repository's package visibility settings, make the resulting GHCR package public or grant the deployment service access.

## Deployment

Use the published image with the services and environment described in [the hosted beta guide](hosted-beta.md). Configure OAuth credentials, encryption secrets, PostgreSQL, Redis, private object storage, and the server-managed Ollama endpoint in the deployment environment. Do not put secrets in workflow files or commit environment files. A hosting provider and its deploy credentials must be selected before adding an automated deployment job.
