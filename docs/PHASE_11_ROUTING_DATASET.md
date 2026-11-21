# Phase 11 — Independent Routing Dataset Construction & Validation

## 1. Executive Summary

Phase 10 of RAG_EVAL was intentionally **BLOCKED** because the repository contained **0 independent training queries** and **0 independent validation queries**, with all existing 121 queries reserved as protected evaluation benchmarks across Tracks A, B, C, and D.

In Phase 11, we designed, implemented, and verified a **leakage-safe, reproducible, independently generated routing dataset** constructed strictly from repository documents. The dataset establishes ground truth for training and evaluating learned retrieval routers (BM25 vs. Dense vs. Hybrid) in Phase 12, without contaminating or modifying the protected evaluation tracks.

> [!IMPORTANT]
> **Production Baseline Retained Unchanged:**
> - Chunker: `ElementAwareChunker` (chunk_size=1000, overlap=200)
> - Embedding: `nomic-embed-text:latest` (768 dimensions)
> - Retrieval: Dense + BM25 RRF (`candidate_k=20`, `rrf_k=100`, `final_k=5`)
> - Generation: `llama3.2:3b`
> - Abstention & citations active; no experimental routing activated in production.

---

## 2. Dataset Partitioning & Document-Level Separation

To avoid data leakage across splits, partitioning is enforced strictly by **SOURCE DOCUMENT** rather than randomly by query. No source document appears in more than one partition.

```
Total Corpus: 17 Source Documents (114 Chunks, 586 Elements)
  │
  ├── TRAIN PARTITION (8 Documents, 70 Chunks, 361 Elements) — 319 Queries (61.0%)
  │     ├── rfc8259.txt (RFC 8259 JSON Specification)
  │     ├── architecture.md (System Architecture)
  │     ├── render-deployment.md (Render Cloud Deployment)
  │     ├── hosted-beta.md (Hosted Beta Platform)
  │     ├── benchmarks.md (Evaluation Benchmarks and SLA)
  │     ├── incident_log.txt (Production Incident Log)
  │     ├── hosted-terms.md (Terms of Service)
  │     └── product_evaluation.docx (Product Evaluation Assessment)
  │
  ├── VALIDATION PARTITION (5 Documents, 23 Chunks, 105 Elements) — 109 Queries (20.8%)
  │     ├── runbook.md (Operational Runbook)
  │     ├── findings.md (Architectural Findings)
  │     ├── hosted-privacy.md (Hosted Privacy Policy)
  │     ├── operations_guide.pdf (Operations Guide)
  │     └── reference_guide.md (API Reference Guide)
  │
  └── TEST PARTITION (4 Documents, 21 Chunks, 120 Elements) — 95 Queries (18.2%)
        ├── README.md (System Overview and Quickstart)
        ├── system_spec.md (System Infrastructure Specification)
        ├── ci-cd.md (CI/CD Pipeline Architecture)
        └── privacy.md (General Privacy Policy)
```

### Document Separation Verification
- Train $\cap$ Validation: $\emptyset$ (0 overlapping documents)
- Train $\cap$ Test: $\emptyset$ (0 overlapping documents)
- Validation $\cap$ Test: $\emptyset$ (0 overlapping documents)
- **Source Separation Status: VERIFIED (PASS)**

---

## 3. Dataset Characteristics & Taxonomy

A total of **523 independent query examples** were deterministically synthesized from document chunks, exceeding the Phase 11 requirement of at least 500 examples.

### 3.1 Taxonomy Distribution

| Query Category | Description | Count | Percentage |
| :--- | :--- | ---:| ---:|
| `specific_detail` | Nuanced facts, rules, constraints, edge cases | 289 | 55.3% |
| `technical_identifier` | Code spans, protocol terms, flags, error codes | 77 | 14.7% |
| `direct_factual` | Core definitions, purpose, capabilities | 75 | 14.3% |
| `numerical` | Quantitative limits, SLAs, durations, ports | 41 | 7.8% |
| `multi_step` | Interaction and sequential flow across concepts | 17 | 3.3% |
| `configuration` | Environment variables, defaults, parameters | 12 | 2.3% |
| `procedural` | Runbook steps, deployment actions, commands | 12 | 2.3% |
| **Total** | | **523** | **100.0%** |

### 3.2 Partition Distribution

| Partition | Document Count | Chunk Count | Query Count | Ratio |
| :--- | ---:| ---:| ---:| ---:|
| **Train** | 8 | 70 | 319 | 61.0% |
| **Validation** | 5 | 23 | 109 | 20.8% |
| **Test** | 4 | 21 | 95 | 18.2% |
| **Total** | **17** | **114** | **523** | **100.0%** |

---

## 4. Oracle Labeling & Retrieval Analysis

Every generated query was evaluated across the three candidate retrieval strategies using the repository's native retrievers:
1. **BM25Retriever**: Exact token and lexical matching
2. **DenseRetriever**: `nomic-embed-text:latest` 768-dim embeddings with cosine similarity
3. **HybridRetriever**: Production Dense + BM25 RRF baseline (`candidate_k=20`, `rrf_k=100`, `final_k=5`)

### 4.1 Oracle Decision Logic
- Metric score: $\text{score} = \text{MRR@5} + 0.5 \times \text{Hit@1}$
- If BM25 score strictly exceeds Dense and Hybrid: `BM25` win
- If Dense score strictly exceeds BM25 and Hybrid: `DENSE` win
- If Hybrid score strictly exceeds BM25 and Dense: `HYBRID` win
- If BM25 and Hybrid tie beating Dense: `BM25` (preferred for computational efficiency)
- If Dense and Hybrid tie beating BM25: `DENSE` (preferred for efficiency)
- If all three tie with non-zero hits: `AMBIGUOUS`
- If all three achieve zero hits: `NO_CLEAR_WINNER`

### 4.2 Observed Strategy Distribution

| Oracle Strategy | Description | Query Count | Percentage |
| :--- | :--- | ---:| ---:|
| `bm25` | Lexical/identifier query where BM25 matched best or tied Hybrid without Dense | 216 | 41.3% |
| `ambiguous` | Equal retrieval effectiveness across all three strategies (tie) | 124 | 23.7% |
| `dense` | Semantic query where Dense matched best or tied Hybrid without BM25 | 92 | 17.6% |
| `hybrid` | Cross-concept or hybrid query where RRF fusion strictly outperformed solo | 48 | 9.2% |
| `no_clear_winner` | Difficult/unmatched query where all strategies missed top-5 | 43 | 8.2% |
| **Total** | | **523** | **100.0%** |

---

## 5. Protected Evaluation Leakage Audit

A dedicated automated leakage audit was conducted comparing all 523 independent routing examples against the 121 protected evaluation queries (Track A: 14, Track B: 52, Track C: 30, Track D: 25) and M4 query analysis records.

| Leakage Metric | Criterion | Measured Result | Audit Status |
| :--- | :--- | :--- | :--- |
| **Exact Overlap** | $\text{count} = 0$ | **0 matches** | **PASS** |
| **Normalized Signature Overlap** | $\text{count} = 0$ | **0 matches** | **PASS** |
| **Near-Duplicate Fuzzy Similarity** | $\text{ratio} < 0.70$ | **0 matches $\ge 0.70$ (max = 0.0000)** | **PASS** |
| **Protected Dataset Isolation** | Tracks A-D untouched | **100% isolated** | **PASS** |
| **Split Document Separation** | Disjoint sets | **100% disjoint** | **PASS** |
| **Overall Leakage Verdict** | | | **PASS** |

---

## 6. Artifact Inventory & Reproducibility

All Phase 11 artifacts are stored under `data/routing/phase11/`:

```text
data/
  routing/
    phase11/
      routing_dataset.jsonl          # Complete independent dataset (523 JSON records)
      routing_dataset_manifest.json  # Comprehensive manifest with SHA-256 integrity hashes
      train.jsonl                    # Train partition (319 records)
      validation.jsonl               # Validation partition (109 records)
      test.jsonl                     # Test partition (95 records)
      oracle_labels.jsonl            # Detailed retrieval metrics & winning strategies
      leakage_report.json            # Machine-readable leakage audit verification
```

### Reproducible Commands
```bash
# 1. Regenerate complete dataset with oracle labels
uv run python scripts/build_routing_dataset.py --seed 42

# 2. Audit leakage against protected evaluation benchmarks
uv run python scripts/audit_routing_leakage.py

# 3. Validate structural integrity, schema, and manifest hashes
uv run python scripts/validate_routing_dataset.py

# 4. Run automated Phase 11 test suite
uv run pytest tests/unit/test_phase11_routing_dataset.py
```
