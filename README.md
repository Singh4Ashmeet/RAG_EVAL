# RAG_EVAL

**A local-first toolkit for building and evaluating retrieval-augmented generation pipelines.**

RAG_EVAL can parse and chunk local text documents, embed and index chunks, retrieve evidence, generate answers from that evidence, score runs against a JSONL dataset, compare pipeline variants, and summarize observed failures. The current end-to-end path uses Ollama and SQLite. Tests use deterministic fakes and do not need Ollama or model downloads.

> The repository's golden evaluation dataset is intentionally empty. It contains no project-specific ground truth, so no RAG quality score or improvement claim is currently available.

## Current scope

- Default ingestion supports UTF-8 `.txt` and `.md` files. It produces normalized documents and chunks; the ingestion CLI does not index them.
- Dense embeddings and chat generation use already-installed local Ollama models.
- SQLite persists chunks and vectors. Similarity search is an exact cosine scan, suitable for local development and small or medium collections.
- Evaluation supports chunk-level ranking metrics and deterministic lexical answer metrics.
- Experiment records compare named variants on the same dataset and report paired bootstrap confidence intervals.
- Diagnostics identify observed retrieval, generation, lexical-overlap, and citation-resolution issues.

The repository also includes an invite-only hosted beta workbench. Its remaining deployment constraints and unimplemented production safeguards are listed in [the hosted beta guide](docs/hosted-beta.md). PDF, DOCX, image, BM25, hybrid, and reranking modules elsewhere in the repository are not wired into the verified default workflow. See [the implementation boundaries](docs/architecture.md#current-scope-and-limitations).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Ollama installed separately for the local embedding and generation workflow
- One Ollama model suitable for embeddings and one chat model, already pulled into Ollama
- Node.js and npm for the React frontend
- Docker Compose for the reference hosted service stack

Unit and integration tests use fakes and do not require the Ollama service or model files.

## Install and verify

From the repository root:

```powershell
uv sync
uv run task test
uv run task typecheck
```

Optional quality commands:

```powershell
uv run task lint
uv run ruff format --check .
```

`task format` rewrites files. Review `git status` before running it in a working tree with existing edits.

## Configure local Ollama models

RAG_EVAL does not start Ollama, download models, or load `.env` files. Start Ollama using its normal installation flow, then make sure an embedding-capable model and a chat model are available locally. Choose the model names that are installed on your machine.

PowerShell example for the current session:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:OLLAMA_EMBED_MODEL = "<installed-embedding-model>"
$env:OLLAMA_MODEL = "<installed-chat-model>"
$env:RAG_EVAL_DB_PATH = "data/rag_eval.sqlite3"
```

See [.env.example](.env.example) for variable names. It is a template, not an automatically loaded configuration file. `configs/baseline.yaml` documents baseline settings but is not loaded by a runner.

## Hosted beta workbench

The frontend lives in `frontend/`. Run the FastAPI API and frontend together from the repository root:

```powershell
uv sync
python main.py
```

Open `http://127.0.0.1:5173`; API docs are at `http://127.0.0.1:8000/docs`. The runner installs frontend dependencies on its first run when needed and stops both services with Ctrl+C. The hosted sign-in flow requires a GitHub OAuth app and an administrator-created invitation. For the container reference stack, see [docs/hosted-beta.md](docs/hosted-beta.md) and [deploy/docker-compose.yml](deploy/docker-compose.yml). See [docs/ci-cd.md](docs/ci-cd.md) for GitHub Actions checks and container publishing.

The workbench supports private document uploads, Ollama indexing and grounded asking, element-labeled immutable datasets, queued retrieval evaluations, paired retrieval-depth experiments, encrypted Gemini/Groq chat keys, and workspace deletion. Read the hosted [Terms](docs/hosted-terms.md) and [Privacy Notice](docs/hosted-privacy.md) before inviting beta users. It remains an early beta; validate PostgreSQL migrations and RLS on the target service, and add rate limits, abuse controls, backup expiry automation, and independent security review before broader access.

For the Render Blueprint, generated `onrender.com` domain, OAuth setup, and required S3/Ollama services, see [the Render deployment guide](docs/render-deployment.md).

## Quickstart: ingest, index, retrieve, and answer

Create a UTF-8 `.txt` or `.md` file, configure the environment above, and run this from the repository root:

```python
import os
from pathlib import Path

from core.embed import OllamaEmbedder
from core.generate import GroundedAnswerGenerator
from core.ingest import create_default_ingestion_pipeline
from core.llm import GenerationSettings, OllamaLLMProvider
from core.retrieval import DenseRetriever
from core.storage import SQLiteVectorStore

source = Path("data/example.md")
ingested = create_default_ingestion_pipeline().ingest(source)
if ingested.document is None or not ingested.chunks:
    raise RuntimeError(f"Ingestion failed: {ingested.errors}")

embedder = OllamaEmbedder(
    model=os.environ["OLLAMA_EMBED_MODEL"],
    host=os.environ.get("OLLAMA_HOST") or None,
)
provider = OllamaLLMProvider(
    model=os.environ["OLLAMA_MODEL"],
    host=os.environ.get("OLLAMA_HOST") or None,
)
database = os.environ.get("RAG_EVAL_DB_PATH", "data/rag_eval.sqlite3")

with SQLiteVectorStore(database) as store:
    retriever = DenseRetriever(embedder, store)
    retriever.index_document(ingested.document_id, ingested.chunks)

    answerer = GroundedAnswerGenerator(
        retriever,
        provider,
        top_k=5,
        settings=GenerationSettings(temperature=0.0, max_tokens=512),
    )
    result = answerer.answer("What does the document say?")

    print(result.answer)
    for source_result in result.sources:
        print(source_result.chunk.chunk_id, source_result.score)
    print("Unresolved citations:", result.unresolved_citation_labels)
```

The one-command ingestion CLI only parses and chunks; indexing is shown explicitly above. Re-indexing a document replaces its stored chunks. A database is tied to one embedding model identity and dimension; clear and rebuild it to switch models. See the [runbook](docs/runbook.md) for details and evaluation workflows.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and implementation boundaries](docs/architecture.md)
- [Setup, workflow, evaluation, experiments, and diagnosis runbook](docs/runbook.md)
- [Metrics, experiment interpretation, and benchmark results](docs/benchmarks.md)
- [Current findings and known limitations](docs/findings.md)
- [Privacy and data handling](docs/privacy.md)
- [Changelog](CHANGELOG.md)

## License and project terms

Copyright © 2026 Ashmeet Singh. RAG_EVAL is proprietary and all rights are reserved; it is not offered under an open-source license. See [LICENSE](LICENSE) and [Terms and Conditions](TERMS.md). Third-party dependencies and models remain under their own licenses. Public repository access is subject to GitHub's platform terms, including its applicable terms for viewing and forking public repositories.

Project process documents: [Contributing](CONTRIBUTING.md) and [Security reporting](SECURITY.md).

## Project status

The local RAG and evaluation path includes repeatable experiment tracking and failure diagnosis. The hosted UI is an invite-only technical workbench; real scores require reviewed ground truth and configured model services. The hosted deployment guide tracks its beta limits.
