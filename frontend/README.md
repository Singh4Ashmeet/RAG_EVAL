# RAG_EVAL Frontend

React + TypeScript workbench, built with Vite and served by FastAPI from the same origin in hosted deployments.

## Development

From this directory, run `npm ci`, then `npm run dev`. Start the API from the repository root with `uv run uvicorn apps.api.main:app --reload`; Vite proxies `/api` to `http://127.0.0.1:8000`.

## Build and API types

- `npm run build` type-checks and creates the production bundle in `dist/`.
- With the API running on port 8000, `npm run api:types` generates `src/api.generated.ts` from FastAPI OpenAPI.
- `npm test` runs the frontend component tests.

The visual system uses warm paper, ink, and forest green. Keep empty and failure states factual, and do not add unearned model availability claims or fabricated evaluation metrics.
