FROM node:24-alpine AS frontend
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --locked
COPY . ./
COPY --from=frontend /web/dist ./frontend/dist
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
