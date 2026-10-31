# Local workflow runbook

This guide covers the verified local path: parse and chunk `.txt`/`.md`, index with Ollama embeddings and SQLite, retrieve, generate with Ollama, and evaluate against reviewed JSONL ground truth. For a shorter overview, see the [README](../README.md).

## Prerequisites

- Use Python 3.12 and install the repository environment with `uv sync`.
- Install and start Ollama separately, and make both an embedding-capable model and a generation model available locally. RAG_EVAL does not download models or start Ollama for you.
- Set `OLLAMA_MODEL` to the generation model name. `llama3.2:3b` was present during M2.3 implementation; verify it remains available on your machine.
- Set `OLLAMA_EMBED_MODEL` to the exact local model name. Set `OLLAMA_HOST` only if Ollama is not listening at its default address.
- Set `RAG_EVAL_DB_PATH` to a writable SQLite database path. The default is `data/rag_eval.sqlite3`.

The variables in `.env.example` are a template; the code reads process environment variables and does not load `.env` files automatically. `configs/baseline.yaml` documents the baseline values but is not yet loaded by a configuration runner.

### Parse and chunk one file

The CLI runs only parsing and chunking. It does not index results into SQLite:

```powershell
uv run python -m apps.cli.ingest data/example.md
```

The default pipeline accepts UTF-8 `.txt` and `.md` files. The Python API below shows how to carry the output into indexing.

For an embedder implementation that does not expose a `model_id` attribute, pass a stable identity explicitly to `DenseRetriever(..., model_id="provider:model-version")` so incompatible vectors cannot be mixed in one index.

## Use the dense retrieval API

After parsing and chunking a file with the ingestion pipeline, index its chunks and query them:

```python
import os
from pathlib import Path

from core.embed import OllamaEmbedder
from core.ingest import create_default_ingestion_pipeline
from core.retrieval import DenseRetriever
from core.storage import SQLiteVectorStore

embedder = OllamaEmbedder(
    model=os.environ["OLLAMA_EMBED_MODEL"],
    host=os.environ.get("OLLAMA_HOST") or None,
)
with SQLiteVectorStore(os.environ.get("RAG_EVAL_DB_PATH", "data/rag_eval.sqlite3")) as store:
    retriever = DenseRetriever(embedder, store)
    result = create_default_ingestion_pipeline().ingest(Path("data/example.txt"))
    if result.document is not None:
        retriever.index_document(result.document_id, result.chunks)

    matches = retriever.retrieve("What does the document say?", top_k=5)
    for match in matches:
        print(match.score, match.chunk.text)
```

Similarity scores are cosine similarity in `[-1, 1]`, where larger values rank first. Ties are ordered by chunk index and then chunk ID. Searching an empty index returns an empty list.

Re-indexing replaces the selected document's previous chunks. To delete one document, call `retriever.delete_document(document_id)`. To change embedding models or dimensions, explicitly call `store.clear()` and rebuild every document; a mismatch otherwise raises `VectorStoreError`.

## Generate a grounded answer

After constructing the `retriever` as above, attach an Ollama provider and answer a question:

```python
from core.generate import GroundedAnswerGenerator
from core.llm import GenerationSettings, OllamaLLMProvider

provider = OllamaLLMProvider(
    model=os.environ["OLLAMA_MODEL"],
    host=os.environ.get("OLLAMA_HOST") or None,
)
answerer = GroundedAnswerGenerator(
    retriever,
    provider,
    top_k=5,
    settings=GenerationSettings(temperature=0.0, max_tokens=512),
)
result = answerer.answer("What does the document say?")
print(result.answer)
for source in result.sources:
    print(source.chunk.chunk_id, source.chunk.text)
print("Unresolved citation labels:", result.unresolved_citation_labels)
```

The answer result retains all retrieved sources even if the model omits citations. Invalid citation labels are reported separately. An empty retrieval result produces a deterministic insufficient-context answer without a model call. Neither valid citation labels nor successful generation establish factual correctness.

## Evaluate against a JSONL dataset

Each non-empty line must be a JSON object containing a unique `query_id`, a `query`, and at least one ground-truth field: `relevant_chunk_ids` (chunk-level retrieval targets), `reference_answer`, or both. Optional per-query `metadata` is preserved. For example:

```json
{"query_id":"q-001","query":"What color is the sky?","relevant_chunk_ids":["doc_a_c0000"],"reference_answer":"The sky is blue.","metadata":{"topic":"weather"}}
```

Keep project-specific ground truth authored and reviewed; the repository's `eval/golden/questions.jsonl` remains empty until those records are supplied. A dataset digest identifies the exact bytes that were evaluated.

```python
from eval.datasets import load_jsonl_dataset
from eval.runners import EvaluationRunConfig, EvaluationRunner

dataset = load_jsonl_dataset("eval/golden/questions.jsonl")
runner = EvaluationRunner(EvaluationRunConfig(cutoffs=(1, 3, 5)))
report = runner.run(dataset, retriever, answerer)
runner.write_report(report, "eval/reports/baseline.json")
print(report.metrics)
```

Retrieval metrics use relevant chunk IDs. Precision@k divides by k even when fewer results are returned; average precision@k divides by `min(number of relevant chunks, k)`. Aggregate values are macro averages over queries with the required ground truth, and `metric_counts` reports each denominator. Answer exact match ignores case, punctuation, and citation labels after Unicode normalization; token F1 uses multiset token overlap. Both are deterministic lexical metrics, not semantic similarity or proof of faithfulness. If answer references exist but no answerer is supplied, the report is marked incomplete rather than treating those metrics as zero.

## Compare pipeline variants

Build each runtime variant explicitly, giving it a unique name and a JSON-safe configuration that describes the actual components. Then run all variants on the same loaded dataset and cutoffs:

```python
from eval.runners import (
    EvaluationRunConfig,
    ExperimentConfig,
    ExperimentRunner,
    ExperimentVariant,
)

experiment = ExperimentRunner(
    ExperimentConfig(
        experiment_id="chunking-trial-001",
        baseline_variant="baseline",
        evaluation=EvaluationRunConfig(cutoffs=(1, 3, 5)),
        bootstrap_resamples=2000,
        confidence_level=0.95,
        random_seed=17,
    )
)
report = experiment.run(
    dataset,
    [
        ExperimentVariant("baseline", {"chunk_size": 500}, baseline_retriever, answerer),
        ExperimentVariant(
            "smaller-chunks", {"chunk_size": 300}, candidate_retriever, candidate_answerer
        ),
    ],
)
experiment.write_report(report, "eval/reports/chunking-trial-001.json")
```

Each candidate comparison contains paired per-query metric means, candidate-minus-baseline deltas, and seeded percentile bootstrap intervals. Latency is measured per query; positive latency deltas mean slower execution. The report makes no automatic improvement claim. The variant configuration is descriptive metadata and must match the constructed runtime objects. Meaningful real scores require reviewed ground truth in the dataset.

## Diagnose an evaluation report

Diagnosis uses recorded errors, retrieval metrics, answer token F1, and citation-resolution results. It does not call a model or change the pipeline:

```python
from eval.diagnostics import DiagnosisConfig, EvaluationDiagnoser

diagnoser = EvaluationDiagnoser(DiagnosisConfig(low_token_f1_threshold=0.5))
diagnosis = diagnoser.diagnose(report)
diagnoser.write_report(diagnosis, "eval/reports/baseline-diagnosis.json")
for finding in diagnosis.findings:
    print(finding.category, finding.affected_query_ids, finding.evidence)
```

The low token-overlap finding is a configurable review heuristic; it is not a factuality judgment. Citation checks only verify that labels resolve to retrieved chunks. Recommendations identify follow-up inspection or controlled comparisons, not proven improvements.

## Verify

Run the deterministic local test suite without Ollama or model downloads:

```powershell
.venv\Scripts\python.exe -m pytest -q
```
