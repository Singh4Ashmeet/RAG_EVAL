# M4.2 Chunk-Size Ablation — Implementation Plan

## 1. Existing Architecture Summary

### Chunking System
- **Interface**: `core.chunking.base.DocumentChunker` (Protocol)
- **Implementation**: `core.chunking.chunker.ElementAwareChunker`
- **Parameters**: `chunk_size` (default 1000), `chunk_overlap` (default 200) — **measured in characters**
- **Deterministic chunk IDs**: `{document_id}_c{chunk_index:04d}`
- **Overlap architecture**: Single-overlap at oversized element splitter level only; packing is forward-only, no packing-level overlap

### Ingestion Pipeline
- `core.ingest.pipeline.IngestionPipeline` orchestrates parsing + chunking
- `create_default_ingestion_pipeline()` creates pipeline with `ElementAwareChunker()` (defaults)
- Pipeline is used in:
  - `_build_corpus_index()` — builds corpus index for validation
  - `_ingest_and_index()` — ingests corpus and indexes into vector store

### Baseline Execution
- `eval.runners.baseline.run_baseline()` orchestrates full evaluation
- Uses `create_default_ingestion_pipeline()` internally (hardcoded defaults)
- Stores artifacts in `eval/runs/baseline/{track}/{run_id}/`

### Evaluation
- `eval.runners.evaluate.EvaluationRunner` runs retrieval + optional generation
- Metrics: Recall@K, Precision@K, MRR@K, MAP@K, HitRate@K, SourceElement variants
- Outputs: `evaluation_report.json`, `aggregate_report.json`, `per_query_results.jsonl`, `run_manifest.json`, etc.

### Configuration
- `BaselineConfig`: track, dataset_path, corpus_dir, embed_model, chat_model, cutoffs, etc.
- No chunking parameters exposed in BaselineConfig

### Tests
- `tests/unit/test_chunker.py` — comprehensive ElementAwareChunker tests
- `tests/unit/test_ingestion_pipeline.py` — IngestionPipeline tests
- `tests/unit/test_baseline_execution.py` — baseline orchestration tests (mocked)
- `tests/unit/test_golden_corpora.py` — corpus/dataset integrity tests

---

## 2. Files to Modify

### A. `core/chunking/__init__.py`
- Export `ElementAwareChunker` (already done)

### B. `core/ingest/pipeline.py`
- Modify `create_default_ingestion_pipeline()` to accept optional `chunk_size` and `chunk_overlap` parameters
- Default to current values (1000, 200) to preserve backward compatibility

### C. `eval/runners/baseline.py`
- Add `chunk_size` and `chunk_overlap` to `BaselineConfig` (optional, with defaults)
- Modify `_ingest_and_index()` to accept and use chunking parameters
- Modify `_build_corpus_index()` to accept and use chunking parameters
- Pass chunking parameters through the pipeline creation

### D. `scripts/run_baseline.py`
- Add `--chunk-size` and `--chunk-overlap` CLI arguments
- Pass them through `BaselineConfig`

---

## 3. Files to Create

### A. Experimental Runner Script
- `scripts/run_chunk_ablation.py` — runs all 4 configurations sequentially
- Creates isolated run directories under `eval/runs/experiments/m4_2_chunk_ablation/`
- Organization: `{config_name}/{track}/{run_id}/`

### B. Evaluation Comparison Script
- `scripts/compare_chunk_ablation.py` — loads all experiment results and generates comparison tables

### C. Test Files
- `tests/unit/test_chunk_ablation.py` — tests for new experimental functionality

---

## 4. Configuration Strategy

### BaselineConfig Extensions
```python
class BaselineConfig(BaseModel):
    # ... existing fields ...
    chunk_size: int = 1000
    chunk_overlap: int = 200
```

### Experimental Configurations
| Name | chunk_size | chunk_overlap |
|------|-----------:|--------------:|
| baseline | 1000 | 200 |
| exp_500_100 | 500 | 100 |
| exp_500_150 | 500 | 150 |
| exp_750_150 | 750 | 150 |

---

## 5. Experimental Isolation Strategy

### Directory Structure
```
eval/runs/experiments/m4_2_chunk_ablation/
├── baseline/
│   ├── track_a/
│   │   └── baseline_track_a_20261104T.../
│   └── track_b/
│       └── baseline_track_b_20261104T.../
├── exp_500_100/
│   ├── track_a/
│   └── track_b/
├── exp_500_150/
│   ├── track_a/
│   └── track_b/
└── exp_750_150/
    ├── track_a/
    └── track_b/
```

### Isolation Guarantees
- Each configuration gets unique timestamped run directory
- No shared vector stores or artifacts
- Manifest records exact configuration used
- M3 baseline artifacts in `eval/runs/baseline/` untouched

---

## 6. Test Plan

### New Tests (`tests/unit/test_chunk_ablation.py`)
1. **Configuration parsing** — CLI args correctly populate BaselineConfig
2. **Pipeline creation** — `create_default_ingestion_pipeline(chunk_size, chunk_overlap)` produces correct chunker
3. **Deterministic chunking** — Same config + same document = same chunks
4. **Configuration isolation** — Different configs produce different chunk counts/boundaries
5. **Experimental runner** — All 4 configs execute without error
6. **Artifact structure** — Experimental runs create expected directory layout
7. **Manifest accuracy** — Manifest records actual chunk_size/chunk_overlap used

### Existing Tests Must Pass
- `test_chunker.py` — all 100+ tests
- `test_ingestion_pipeline.py` — all tests
- `test_baseline_execution.py` — all 40 tests
- `test_golden_corpora.py` — all tests

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OLLAMA unavailable during experiment | Report partial results; preserve artifacts; provide recovery instructions |
| Generation timeout (30-60s/query) | Run retrieval-only mode first to validate; run generation in separate pass |
| Resource contention (multiple concurrent runs) | Run configurations sequentially; one at a time |
| Non-deterministic chunk IDs | Chunk IDs use sequential index; verify deterministic across runs |
| Breaking existing default behavior | Default params preserve 1000/200; existing tests validate |
| Large chunk count increase (500/100 → ~2× chunks) | Monitor disk usage; SQLite scan time increases linearly |

---

## 8. Implementation Order

1. **Modify core ingestion** — `create_default_ingestion_pipeline()` accepts chunk params
2. **Extend BaselineConfig** — add chunk_size/chunk_overlap with defaults
3. **Modify baseline.py** — thread chunk params through `_build_corpus_index` and `_ingest_and_index`
4. **Update CLI** — add `--chunk-size` and `--chunk-overlap` args
5. **Create experimental runner** — `scripts/run_chunk_ablation.py`
6. **Create comparison script** — `scripts/compare_chunk_ablation.py`
7. **Write tests** — `tests/unit/test_chunk_ablation.py`
8. **Run experiments** — Execute all 4 configs for both tracks
9. **Analyze and report** — Generate comparison tables and final report

---

## 9. Reproducibility Requirements

- All experimental runs use fixed golden datasets (M3 immutable)
- Same embedding model (`nomic-embed-text`) and generation model (`llama3.2:3b`)
- Same cutoffs (1, 3, 5), same retrieval implementation
- Only chunk_size and chunk_overlap vary
- Manifests record exact configuration for auditability
- Existing M3 baseline artifacts never modified