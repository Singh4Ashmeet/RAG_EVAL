# Current findings and open gaps

This page separates implemented behavior from evidence that still needs to be collected. For usage, see the [runbook](runbook.md); for component boundaries, see the [architecture](architecture.md).

## Verified capabilities

- The default parser handles UTF-8 plain text and Markdown, producing normalized documents with source checksums.
- The element-aware chunker creates deterministic chunk IDs and preserves element provenance.
- Ollama embedding and chat providers are implemented. They require an Ollama service and models already installed locally.
- SQLite stores chunk/vector records, replaces documents transactionally, enforces embedding model/dimension compatibility, and searches by exact cosine similarity.
- Grounded answer generation keeps retrieved source provenance, validates citation label mapping, and returns a deterministic insufficient-context response when retrieval is empty.
- Evaluation supports retrieval ranking metrics, lexical answer metrics, per-query reports, and exact-byte dataset digests.
- Experiment reports compare paired per-query deltas with deterministic bootstrap intervals and preserve caller-supplied configuration metadata.
- Diagnosis reports connect observed failure patterns to query IDs and evidence without changing pipeline settings.

Automated tests use fakes for provider calls and do not require Ollama or model downloads. See the repository test suite and the commands in the root README.

## Evidence not yet available

- `eval/golden/questions.jsonl` is empty. There are no reviewed project questions, relevant chunk labels, or reference answers, so there are no real retrieval or answer metrics.
- No controlled experiment has established that one chunk size, model, or retrieval setting is better for project data.
- Model inference benchmark artifacts are local performance samples only. They do not measure RAG quality and should be re-run on the target machine.
- Citation resolution confirms a label refers to a retrieved chunk; no component currently verifies that the chunk supports the cited claim.

## Product and engineering gaps

- There is no frontend. The API and worker modules are placeholders; the usable workflows are Python APIs and a minimal ingestion CLI.
- The default ingest path only registers `.txt` and `.md`, although other parser modules exist in the repository.
- BM25, hybrid retrieval, and reranking modules are not connected to the current verified pipeline.
- SQLite uses an exact vector scan and will become slow as a corpus grows.
- Configuration YAML and `.env.example` document settings but are not loaded automatically by an application runner.
- Evaluation answer metrics are lexical only; semantic answer relevance and claim-level faithfulness are not measured.
- Experiment reports capture latency but not memory or other resource consumption.

## Next planning step

The next product discussion is the frontend. Before choosing a framework, define the target user flow and decide whether the frontend should call a new application API or operate as a local desktop-style interface around the existing Python core. The API package is currently a stub, so an API-backed frontend would require a separately scoped backend milestone.
