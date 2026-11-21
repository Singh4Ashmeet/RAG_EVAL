# Phase 0 Validation

## Environment

- **Python version**: 3.12.13 (CPython 64-bit)
- **OS/environment**: Windows (x86_64)
- **Package manager**: `uv` 0.11.29 (lockfile: `uv.lock`, package mode: false)
- **Project version**: `rag-eval` 0.1.0 (`pyproject.toml`)

## Repository State

- **Git working tree status**: Verified. Stale untracked bytecode was cleaned, `.gitignore` updated, and UI screenshot assets restored.
- **Tracked secret findings**: Clean. Comprehensive scan across git-tracked files and working tree found zero committed API tokens, credentials, or keys.
- **Generated artifact findings**: All generated run outputs (`eval/runs/`), caches (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`), and bytecode (`__pycache__/`, `*.pyc`) are strictly ignored.
- **Cleanup performed**: 
  - Purged all non-venv `__pycache__` directories across workspace.
  - Added `.mypy_cache/`, `.ruff_cache/`, and `/eval/runs/` to `.gitignore`.
  - Added `apps/__init__.py` to resolve Python package hierarchy and mypy module resolution.

## Dependency Validation

- **Dependency installation result**: `uv sync` verified against `uv.lock`. All 88 runtime and development dependencies resolved cleanly with zero extraneous or missing packages.
- **Declared development dependencies**: `pytest>=9.2.0`, `pytest-asyncio>=1.4.1`, `pytest-cov>=7.1.0`, `ruff>=0.17.3`, `mypy>=2.4.1`, `taskipy>=1.14.1`, `pre-commit>=4.7.0` verified and active.

## Static Validation

- **Ruff result**: `uv run ruff check .` -> `All checks passed!` (Exit code 0).
- **Ruff format result**: `uv run ruff format --check .` -> `157 files already formatted` (Exit code 0).
- **MyPy result**: `uv run mypy core apps eval` -> `Success: no issues found in 75 source files` (Exit code 0). All type errors resolved with precise annotations; no global suppressions.

## Test Validation

- **Total tests**: 801 collected
- **Passed**: 800 non-live passed (54.57s) + 1 live integration test passed (16.79s)
- **Failed**: 0
- **Skipped**: 1 (live test skipped by default when `--live` is not passed)
- **Xfailed**: 0

## Live Service Validation

- **Ollama available**: Yes (operational on `http://127.0.0.1:11434`, version 0.40.13)
- **Configured model(s)**:
  - Embedding: `nomic-embed-text:latest` (768 dimensions)
  - Chat Generation: `llama3.2:3b`
- **Model availability**: Confirmed active via `GET /api/tags`
- **Live tests executed**: `tests/integration/test_rag_generation.py::test_live_ollama_retrieval_and_answer_generation`
- **Live tests passed**: 1 passed (16.79s)
- **Credentials/Secrets**: Safe local service; no secrets exposed.

## RAG Smoke Test

- **Input corpus**: Markdown document ingested via `create_default_ingestion_pipeline(chunk_size=200, chunk_overlap=20)`
- **Document count**: 1 document (`sample.md`)
- **Chunk count**: 1 chunk extracted by `ElementAwareChunker`
- **Embedding execution**: Embedded via `OllamaEmbedder(model="nomic-embed-text")` yielding 768-dimensional float vectors
- **Vector persistence**: Chunks and embeddings persisted and verified in `SQLiteVectorStore`
- **Retrieval execution**: Query embedded and retrieved with top score `0.7848` via `DenseRetriever`
- **Generation execution**: Grounded answer generated using `GroundedAnswerGenerator` backed by `OllamaLLMProvider(model="llama3.2:3b")`
- **Citation/response validation**: Generated response contained verified citation label `[S1]` matching chunk ID `doc_0f67a936e95a6cd0_c0000`, with `abstained=False`.

## Baseline

- **Baseline command**: `uv run python scripts/run_baseline.py --track track_a`
- **Configuration**:
  - Track: `track_a`
  - Dataset: `eval/golden/track_a.jsonl` (14 queries)
  - Corpus: `eval/corpora/track_a`
  - Embedding Model: `nomic-embed-text`
  - Generation Model: `llama3.2:3b`
  - Cutoffs: `(1, 3, 5)`
  - Chunk Size / Overlap: 1000 / 200
  - Top K: 5
- **Result**:
  - Run ID: `baseline_track_a_20261116T160502Z`
  - Status: `complete` (14/14 queries succeeded, 0 failed)
  - Duration: 396.30s
  - Metrics:
    - `retrieval.hit_rate@1`: 0.5714
    - `retrieval.hit_rate@3`: 0.8571
    - `retrieval.hit_rate@5`: 0.9286
    - `retrieval.mrr@5`: 0.7202
    - `answer.token_f1`: 0.3406
  - Artifacts generated:
    - `aggregate_report.json`
    - `config_snapshot.json`
    - `coverage_report.json`
    - `evaluation_report.json`
    - `per_query_results.jsonl`
    - `run_manifest.json`
    - `validation_report.json`
    - `vector_store.sqlite3`
- **Limitations**: On CPU-only systems, LLM generation takes ~20-25s per query (396s total for 14 queries). Fast retrieval evaluation without generation is supported via `--no-generation` (executes in 12.77s).

## Remaining Known Issues

None. All static checks, type annotations, unit tests, live integrations, baseline execution, and component startup routines are verified and passing.

## Phase 0 Gate

PASS

## Phase 1 Readiness

Phase 0 is complete and the repository is ready for Phase 1 implementation.
