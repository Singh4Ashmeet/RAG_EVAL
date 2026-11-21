# Phase 2 Evaluation Suite Architecture and Methodology

## 1. Overview and Design Principles

The Phase 2 Evaluation Suite transforms `RAG_EVAL` from a single-track test into an independent, multi-track, scientifically rigorous measurement instrument.

In accordance with Phase 2 principles:
- **Zero RAG System Modifications**: No retrieval optimization, chunk size adjustments, BM25 tuning, reranking, or prompt manipulation was introduced. The RAG system remains an unchanged subject of evaluation.
- **Strict Control Preservation**: The historical Track A control dataset (14 queries) is preserved byte-for-byte in its original state to provide an immutable benchmark control.
- **Scientific Measurement Layer**: New evaluation tracks (B, C, D) were created with deterministic corpus ground truth to measure standard RAG queries, multi-format and adversarial challenges, and abstention/hallucination resistance.

---

## 2. Multi-Track Dataset Architecture

The evaluation framework consists of four independent tracks located in `eval/golden/` and grounded in corresponding corpus directories in `eval/corpora/`:

| Track | Dataset Path | Corpus Directory | Corpus Formats | Query Count | Purpose / Characteristics |
|---|---|---|---|---|---|
| **Track A** | `eval/golden/track_a.jsonl` | `eval/corpora/track_a/` | Markdown (`.md`) | 14 | Historical control baseline; regression control across all phases |
| **Track B** | `eval/golden/track_b.jsonl` | `eval/corpora/track_b/` | Plain text (`rfc8259.txt`) | 52 | Expanded standard RAG questions covering all 30 chunks and sections of IETF RFC 8259 |
| **Track C** | `eval/golden/track_c.jsonl` | `eval/corpora/track_c/` | `.md`, `.pdf`, `.docx`, `.txt` | 30 | Multi-format, multi-page PDF provenance, markdown & docx tables, near-duplicate incident log discrimination, and cross-document synthesis |
| **Track D** | `eval/golden/track_d.jsonl` | `eval/corpora/track_d/` | Markdown (`reference_guide.md`) | 25 | Unanswerable / insufficient-evidence queries to measure hallucination resistance and abstention accuracy |

### Track Details

#### Track A (Historical Control)
- **14 queries** grounded in `architecture.md`, `hosted-beta.md`, and `runbook.md`.
- Unaltered from Phase 0 and Phase 1.
- Establishes control stability: Hit@1 = 0.5714, Hit@3 = 0.8571, Hit@5 = 0.9286, MRR@5 = 0.7202, MAP@5 = 0.7202, Source Element Recall@5 = 0.9286, Token F1 = 0.3017–0.3406.

#### Track B (Expanded Standard RAG)
- **52 queries** spanning direct factual lookup, specific detail extraction, multi-paragraph synthesis, and ABNF grammar rules.
- Grounded in `eval/corpora/track_b/rfc8259.txt` (30 chunks, 221 elements).
- Provides a realistic, statistically meaningful test of dense vector retrieval over longer single-document technical specifications.

#### Track C (Multi-Format & Adversarial RAG)
- **30 queries** grounded in a diverse 4-document multi-format corpus:
  1. `system_spec.md`: Multi-column markdown tables for operational tiers, SLAs, and error codes (`ERR-1001` through `ERR-3001`).
  2. `operations_guide.pdf`: 3-page synthetic PDF testing page-level provenance (node topologies on page 1, HA timeouts on page 2, DR RPO/RTO on page 3).
  3. `product_evaluation.docx`: Word document containing performance benchmark tables (throughput, P95/P99 latency, F1) and deployment recommendations.
  4. `incident_log.txt`: Near-duplicate incident reports (`INC-401` vs `INC-402`, `INC-501` vs `INC-502`) designed to challenge embedding discriminability.
- Queries explicitly test `requires_table`, `requires_multi_chunk`, `category="table_lookup"`, `category="provenance"`, `category="adversarial"`, and `category="cross_document"`.

#### Track D (Abstention & Hallucination Resistance)
- **25 unanswerable queries** grounded against `eval/corpora/track_d/reference_guide.md`.
- All cases have `answerable=False`, `answerability="unanswerable"`, `relevant_chunk_ids=[]`, `relevant_element_ids=[]`, and `reference_answer=None`.
- Tests the model's refusal behavior when presented with queries outside the corpus (e.g. Cassandra, Kafka 0.8, DynamoDB, Couchbase, Mesos, Vault mTLS).

---

## 3. Schema Extensions

The dataset schema (`eval/datasets/schema.py`) was extended while preserving backward compatibility:

1. **`CorpusTrack`**:
   - Added `TRACK_C = "track_c"`
   - Added `TRACK_D = "track_d"`
2. **`QuestionCategory`**:
   - Added `TABLE_LOOKUP = "table_lookup"`
   - Added `NUMERICAL_COMPARISON = "numerical_comparison"`
   - Added `ADVERSARIAL = "adversarial"`
   - Added `PROVENANCE = "provenance"`
3. **`EvaluationCase`**:
   - Added `answerable: bool | None = None` with before-validation synchronization (`sync_answerable`) mapping `answerability == "answerable"` to `True` and `"unanswerable"` to `False`.
   - Added `requires_multi_chunk: bool = False`.
   - Added `requires_table: bool = False`.
4. **`EvaluationDataset`**:
   - Added `version: str = Field(default="1.0.0")`.

---

## 4. Evaluation Metrics and Abstention Scoring

### Retrieval Metrics
- `hit_rate@k` ($k \in \{1, 3, 5\}$): Proportion of queries where at least one relevant chunk was retrieved in the top $k$.
- `recall@k`: Fraction of all declared relevant chunks retrieved in the top $k$.
- `precision@k`: Fraction of top-$k$ retrieved chunks that are relevant.
- `mrr@k`: Mean Reciprocal Rank of the first relevant chunk in the top $k$.
- `map@k`: Mean Average Precision across relevant chunks in the top $k$.
- `source_element_recall@k`: Recall evaluated against fine-grained document elements.

### Answer Quality Metrics
- `token_f1`: Unigram token overlap between generated answer and ground-truth reference answer.
- `exact_match`: Binary indicator of normalized string equivalence.

### Abstention and Hallucination Metrics (`eval/metrics/abstention.py`)
- **`is_abstention(text: str | None) -> bool`**: Regex-backed refusal detector normalizing text and matching natural language refusal idioms:
  - `"cannot answer"`, `"cannot be answered"`, `"cannot determine"`, `"cannot provide"`
  - `"could not find"`, `"couldn't find"`, `"unable to find"`
  - `"insufficient information"`, `"not enough information"`, `"insufficient evidence"`
  - `"does not contain"`, `"do not contain"`, `"does not mention"`, `"do not mention"`
  - `"not supported by the provided source"`
  - Empty strings or `None` are classified as abstentions.
- **`abstention_accuracy(is_abstained, is_unanswerable) -> float`**: Accuracy of the system's refusal decision relative to ground-truth answerability.
- **`false_answer_rate(is_abstained, is_unanswerable) -> float`**: Proportion of unanswerable queries where the model hallucinated or attempted an answer ($1.0 - \text{abstention\_accuracy}$).

---

## 5. Failure Taxonomy Classification

Implemented in `eval/failure_taxonomy.py` via `FailureType(StrEnum)`:

```
FailureType
├── NONE ("none"): Query succeeded; relevant evidence retrieved and answer generated with high fidelity.
├── RETRIEVAL_FAILURE ("retrieval_failure"): Zero relevant chunks retrieved in top-k (hit_rate@k == 0).
├── RANKING_FAILURE ("ranking_failure"): Relevant evidence retrieved in top-k, but missed rank 1, causing degraded downstream synthesis.
├── GENERATION_FAILURE ("generation_failure"): Evidence retrieved at rank 1, but generated answer had low token F1 (< 0.3) or model abstained on an answerable question.
├── ABSTENTION_FAILURE ("abstention_failure"): System failed to abstain on an unanswerable query (hallucination).
├── ANNOTATION_FAILURE ("annotation_failure"): Query marked answerable but missing ground-truth chunks or reference answer.
└── SYSTEM_FAILURE ("system_failure"): Unhandled runtime exception or provider communication failure during execution.
```

The classifier deterministically assigns each query a `failure_type`, stored in `per_query_results.jsonl` and aggregated in baseline reports.

---

## 6. Corpus Ground-Truth and Provenance Verification

- All corpora are accompanied by `manifest.json` and `PROVENANCE.json` tracking document identity (SHA-256), title, source URI, and retrieval timestamps.
- Automated validation via `DatasetValidator` verifies:
  1. Dataset schema and JSONL parsing integrity.
  2. Document existence in corpus.
  3. Chunk existence and element containment (`CHUNK_ELEMENT_MISMATCH` check).
  4. Evidence reference existence (`PROVENANCE_EVIDENCE_MISSING` check).
  5. Answerability consistency (`ANNOTATION_STATE_INCONSISTENT` check).
- Validation across all 4 tracks passes with **0 errors and 0 warnings**.

---

## 7. How to Reproduce Baseline Executions

Execute baseline runs using the unified CLI runner:

```bash
# 1. Historical Control Regression (Track A - with generation)
uv run python scripts/run_baseline.py --track track_a

# 2. Expanded Standard RAG (Track B - retrieval only)
uv run python scripts/run_baseline.py --track track_b --no-generation

# 3. Multi-Format & Adversarial RAG (Track C - retrieval only)
uv run python scripts/run_baseline.py --track track_c --no-generation

# 4. Abstention & Hallucination Evaluation (Track D - with generation)
uv run python scripts/run_baseline.py --track track_d

# 5. Automated Golden Dataset Validation
uv run python scripts/generate_golden_datasets.py

# 6. Full Test Suite Execution
uv run pytest
```
