# Changelog

Significant project changes are recorded here. This file documents development milestones; entries do not imply a tagged or packaged release.

## Unreleased

### Added

- UTF-8 text and Markdown parsing, element-aware chunking, and an ingestion CLI.
- Local Ollama embeddings and chat generation with persistent SQLite dense retrieval.
- Grounded answers that preserve retrieved-source provenance and citation-label audits.
- Validated JSONL evaluation datasets, deterministic retrieval and lexical answer metrics, and per-query reports.
- Named experiment variants with paired bootstrap confidence intervals and latency measurements.
- Evidence-linked failure diagnoses for retrieval, generation, answer overlap, and citation resolution.
- Project architecture, runbook, benchmark, privacy, security, license, terms, and contribution documentation.

### Known limitations

- The golden evaluation dataset is empty; no project RAG scores are available.
- The default ingestion path supports `.txt` and `.md`; other parser modules are not connected to it.
- Dense SQLite search is an exact scan; BM25, hybrid search, reranking, a service API, and a frontend are not part of the verified workflow.
- The project is proprietary and all rights are reserved, subject to GitHub's platform terms for public content.
