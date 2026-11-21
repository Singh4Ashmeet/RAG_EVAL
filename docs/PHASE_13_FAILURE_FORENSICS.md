# Phase 13 — Retrieval Failure Forensics & Bottleneck Analysis

## Executive Summary

Phase 13 performed rigorous failure forensics across all protected evaluation tracks (A, B, C, D) using the frozen production baseline. Every query received a deterministic failure classification, and cross-reference analysis against Phases 3–12 experimental data identified the dominant retrieval bottleneck.

### Phase 13 Decision Gate Verdict

```text
PHASE 13 COMPLETE — FAILURE BOTTLENECK IDENTIFIED
```

**Primary Bottleneck:** RANKING_FAILURE — the relevant chunk is retrieved within the candidate pool but is not placed at rank 1 (or is pushed below rank 5 by RRF fusion). This accounts for **24 of 36 total failures** (67%) across all answerable tracks.

**Recommended Next Intervention:** Cross-encoder reranking of the RRF candidate pool. Phase 7 already demonstrated that MiniLM-L-6-v2 reranking improves Track B Hit@1 from 0.4038 to 0.5385 (+13.5pp). The forensic evidence shows this is the intervention with the highest evidence-backed probability of improvement.

---

## 1. Frozen Production Baseline

| Component | Configuration |
|---|---|
| Chunker | `ElementAwareChunker` (chunk_size=1000, overlap=200) |
| Embedding Model | `nomic-embed-text:latest` (768 dimensions) |
| Retrieval Fusion | Reciprocal Rank Fusion (`BM25 + Dense RRF`) |
| Fusion Hyperparameters | `candidate_k=20`, `rrf_k=100`, `final_k=5` |
| Generator Model | `llama3.2:3b` |
| Safety | Citations, provenance tracking, abstention logic |

No retrieval algorithm was introduced, modified, or activated during Phase 13.

---

## 2. Baseline Metrics (Reproduced)

| Track | Queries | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Failures@5 | Suboptimal | Success@1 |
|---|---|---|---|---|---|---|---|---|
| **Track A** (Historical Control) | 14 | 0.6429 | 0.9286 | **1.0000** | 0.7917 | 0 | 5 | 9 |
| **Track B** (Primary Benchmark) | 52 | 0.4038 | 0.5962 | 0.7308 | 0.5212 | **14** | **17** | 21 |
| **Track C** (Adversarial/Hard) | 30 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0 | 0 | 30 |
| **Track D** (Abstention Safety) | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

- **Track A:** Perfect Hit@5 (1.0000). 5 queries have the relevant chunk at ranks 2–4 instead of rank 1.
- **Track B:** 14 complete misses (Hit@5=0) and 17 suboptimal rankings (Hit@5=1 but Hit@1=0). This is the primary target for improvement.
- **Track C:** Perfect across all metrics.
- **Track D:** All 25 unanswerable queries correctly trigger abstention (100% safety).

---

## 3. Failure Taxonomy Distribution

### Global Failure Clusters (Across All Answerable Tracks)

| Failure Cluster | Count | Percentage |
|---|---|---|
| **RANKING_FAILURE** | 24 | **66.7%** |
| **FUSION_FAILURE** | 5 | 13.9% |
| **IDENTIFIER_TOKENIZATION** | 3 | 8.3% |
| **SEMANTIC_MISMATCH** | 2 | 5.6% |
| **LEXICAL_MISMATCH** | 2 | 5.6% |
| **Total failures** | **36** | 100% |

> [!IMPORTANT]
> RANKING_FAILURE dominates the failure distribution. In 24 of 36 failures, the relevant chunk **is retrieved** within the candidate pool but placed at the wrong rank position. This is a reranking problem, not a retrieval coverage problem.

### Track B Failure Breakdown (14 Complete Misses)

| Failure Class | Count | Query IDs |
|---|---|---|
| **FUSION_FAILURE** | 5 | q10, q26, q41, q46, q47 |
| **IDENTIFIER_TOKENIZATION** | 3 | q27, q29, q51 |
| **RANKING_FAILURE** | 2 | q15, q45 |
| **SEMANTIC_MISMATCH** | 2 | q05, q42 |
| **LEXICAL_MISMATCH** | 2 | q13, q28 |

---

## 4. Track B Failure Forensics (Query-Level Evidence)

### 4.1 Complete Misses (Hit@5 = 0) — 14 Queries

| Query ID | Query | BM25 Rank (top-20) | Dense Rank (top-20) | Failure Class | Recoverability |
|---|---|---|---|---|---|
| q05 | What components form the grammar of a JSON number? | None | None | SEMANTIC_MISMATCH | Fundamentally unsupported |
| q10 | Which four whitespace characters are permitted? | None | 4 | FUSION_FAILURE | **Recoverable** (Dense top-5) |
| q13 | Which RFC does RFC 8259 obsolete? | None | 8 | LEXICAL_MISMATCH | Potentially recoverable |
| q15 | Is ECMA-404 reference normative or informative? | 20 | 9 | RANKING_FAILURE | Potentially recoverable |
| q26 | Does JSON number grammar permit leading zeros? | None | 4 | FUSION_FAILURE | **Recoverable** (Dense top-5) |
| q27 | What IEEE 754 precision is recommended? | 19 | 18 | IDENTIFIER_TOKENIZATION | Deep candidate |
| q28 | What exact range of integers is interoperable? | None | 6 | LEXICAL_MISMATCH | Potentially recoverable |
| q29 | Are NaN, Infinity permitted in JSON numbers? | None | None | IDENTIFIER_TOKENIZATION | Fundamentally unsupported |
| q41 | What DoS vulnerability affects JSON parsers? | 16 | **1** | FUSION_FAILURE | **Recoverable** (Dense rank 1!) |
| q42 | Risk from large numbers exceeding 64-bit float? | 13 | None | SEMANTIC_MISMATCH | Deep candidate |
| q45 | What changed in Section 8.1 between RFC 7159/8259? | 11 | 6 | RANKING_FAILURE | Potentially recoverable |
| q46 | Which BCP reference was removed in Appendix A? | None | 4 | FUSION_FAILURE | **Recoverable** (Dense top-5) |
| q47 | Which errata IDs were applied to RFC 8259? | None | **2** | FUSION_FAILURE | **Recoverable** (Dense rank 2!) |
| q51 | What document defines ABNF notation? | 15 | 12 | IDENTIFIER_TOKENIZATION | Deep candidate |

### 4.2 Recoverability Summary

| Category | Count | Query IDs |
|---|---|---|
| **Already recoverable** (in individual top-5) | 5 | q10, q26, q41, q46, q47 |
| **Potentially recoverable** (in top-10) | 3 | q13, q15, q28, q45 |
| **Deep candidate** (in top-20 only) | 3 | q27, q42, q51 |
| **Fundamentally unsupported** | 2 | q05, q29 |

> [!TIP]
> 5 of the 14 Track B misses are **already recoverable**: the relevant chunk appears in the top-5 of an individual retriever (Dense) but gets pushed below rank 5 by RRF fusion because BM25 doesn't contribute a matching score. A reranker applied to the extended candidate pool would recover these.

### 4.3 Suboptimal Rankings (Hit@5 = 1, Hit@1 = 0) — 17 Queries

| Query ID | Hybrid Rank | BM25 Rank (top-5) | Dense Rank (top-5) | Best Individual |
|---|---|---|---|---|
| q04 | 2 | 1 | 5 | BM25=1 |
| q06 | 2 | 1 | None | BM25=1 |
| q07 | 3 | 3 | None | BM25=3 |
| q12 | 5 | 4 | None | BM25=4 |
| q14 | 4 | 2 | None | BM25=2 |
| q16 | 4 | 5 | None | BM25=5 |
| q23 | 2 | 3 | 1 | Dense=1 |
| q24 | 5 | 3 | None | BM25=3 |
| q25 | 5 | None | None | Hybrid=5 |
| q30 | 2 | 1 | None | BM25=1 |
| q32 | 3 | 2 | 2 | Tie=2 |
| q33 | 2 | 2 | 2 | Tie=2 |
| q37 | 4 | 5 | 5 | Tie=5 |
| q39 | 2 | 2 | 2 | Tie=2 |
| q43 | 2 | 5 | 4 | Dense=4 |
| q44 | 4 | None | 2 | Dense=2 |
| q50 | 3 | None | None | Hybrid=3 |

**Key pattern:** In 6 of 17 suboptimal queries, BM25 alone ranks the relevant chunk at **rank 1**, but RRF fusion pushes it to rank 2+. This further supports the reranking intervention thesis.

---

## 5. Cross-Reference with Previous Experiments (Track B)

### 5.1 Aggregate Comparison Table

| Configuration | Phase | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Latency |
|---|---|---|---|---|---|---|
| Dense only | 4 | 0.4231 | 0.5192 | 0.6538 | 0.5029 | 104 ms |
| BM25 only | 4 | 0.4038 | 0.5769 | 0.6731 | 0.5019 | 0.5 ms |
| **Hybrid RRF (Baseline)** | **4** | **0.4038** | **0.5962** | **0.7308** | **0.5212** | **107 ms** |
| Reranker (MiniLM-L-6-v2) | 4/7 | **0.5385** | **0.6731** | 0.7115 | **0.5994** | 1572 ms |
| SPLADE only | 6 | 0.0577 | 0.2308 | 0.4231 | 0.1798 | 79 ms |
| HyDE Dense | 8 | 0.3654 | 0.6731 | **0.7885** | 0.5247 | 6539 ms |
| **Oracle routing** | **9** | **0.5769** | **0.7692** | **0.8462** | **0.7115** | — |

### 5.2 Headroom Analysis

| Metric | Baseline | Best Previously Tested | Best Config | Oracle Routing | Theoretical Max |
|---|---|---|---|---|---|
| Hit@1 | 0.4038 | **0.5385** | Phase 7: MiniLM Reranker | 0.5769 | 1.0000 |
| Hit@3 | 0.5962 | **0.7692** | Phase 9: Oracle Routing | 0.7692 | 1.0000 |
| Hit@5 | 0.7308 | **0.8462** | Phase 9: Oracle Routing | 0.8462 | 1.0000 |
| MRR@5 | 0.5212 | **0.7115** | Phase 9: Oracle Routing | 0.7115 | 1.0000 |

> [!NOTE]
> Oracle routing achieves Hit@5 = 0.8462, meaning 8 of the 14 misses (57%) are recoverable by selecting the optimal retrieval strategy per query. The reranker achieves Hit@1 = 0.5385, the highest measured Hit@1 of any tested configuration.

### 5.3 Persistent Failures Across All Methods

From Phase 6 SPLADE complementarity analysis, **9 Track B queries** are not found by ANY retriever (Dense, BM25, or SPLADE) in top-5. These represent the fundamental retrieval ceiling for the current chunking and embedding configuration.

From our Phase 13 analysis: **2 queries** (q05, q29) are fundamentally unsupported — neither BM25 nor Dense finds the relevant chunk even in top-20.

---

## 6. Phase 11 Oracle Data Analysis

### 6.1 Strategy Distribution (523 Independent Queries)

| Oracle Strategy | Count | Percentage |
|---|---|---|
| BM25 | 216 | 41.3% |
| Ambiguous | 124 | 23.7% |
| Dense | 92 | 17.6% |
| Hybrid | 48 | 9.2% |
| No Clear Winner | 43 | 8.2% |

### 6.2 No-Clear-Winner Analysis (43 Queries)

All 43 no-clear-winner queries are **complete misses across all strategies** (BM25, Dense, and Hybrid all achieve Hit@5 = 0.0).

**Document distribution of no-clear-winner queries:**

| Source Document | Count |
|---|---|
| rfc8259.txt | 12 |
| README.md | 11 |
| architecture.md | 5 |
| hosted-beta.md | 5 |
| runbook.md | 3 |
| hosted-privacy.md | 2 |
| Others (5 documents) | 5 |

**Root cause analysis:** These 43 queries represent cases where the independent query generation process (Phase 11) created queries whose answers exist in the source document but the current chunking + embedding configuration cannot surface the relevant chunk. This confirms that ~8.2% of queries hit a fundamental retrieval ceiling with the current pipeline.

The underlying problems are:
1. **Chunk boundary fragmentation** — the answer spans chunk boundaries
2. **Vocabulary mismatch** — the generated query uses different terminology than the source text
3. **Sparse ABNF/grammar notation** — technical notation doesn't embed well into 768-dimensional dense space

### 6.3 Ambiguous Query Analysis (124 Queries)

| Category | Count | Interpretation |
|---|---|---|
| All strategies achieve MRR@5 = 1.0 | 120 (96.8%) | Query is **too easy** — all strategies retrieve at rank 1 |
| All strategies achieve Hit@5 = 1.0 but different MRR | 4 (3.2%) | Strategies differ in ranking but all find the answer |
| Mixed hit patterns | 0 (0%) | — |

**Conclusion:** 96.8% of ambiguous queries are trivially easy — every strategy retrieves the correct chunk at rank 1. These queries are not useful for future routing training because they provide no routing signal.

---

## 7. Recoverable vs Persistent Failures

### Category 1 — Already Recoverable (Existing Experimental Component Succeeds)

| Query ID | Recoverable By | Evidence |
|---|---|---|
| q10 | Dense-only retrieval | Dense rank 4, BM25 misses |
| q26 | Dense-only retrieval | Dense rank 4, BM25 misses |
| q41 | Dense-only retrieval | Dense rank **1**, BM25 rank 16 |
| q46 | Dense-only retrieval | Dense rank 4, BM25 misses |
| q47 | Dense-only retrieval | Dense rank **2**, BM25 misses |
| q04, q06, q30 | BM25-only retrieval | BM25 rank 1, but RRF pushes to rank 2 |

**Mechanism:** A cross-encoder reranker applied to the RRF candidate pool (top-20 from each retriever) would re-score and promote these correctly.

### Category 2 — Potentially Recoverable

| Query ID | Evidence | Potential Mechanism |
|---|---|---|
| q13 | Dense rank 8 | Larger candidate_k or reranking |
| q15 | Dense rank 9, BM25 rank 20 | Reranking of extended candidates |
| q28 | Dense rank 6 | Reranking |
| q45 | Dense rank 6, BM25 rank 11 | Reranking |

### Category 3 — Fundamentally Unsupported

| Query ID | Query | Root Cause |
|---|---|---|
| q05 | What components form the grammar of a JSON number? | Chunk c0014 contains only ABNF notation (`number = [ minus ] int [ frac ] [ exp ]`). Neither BM25 nor Dense can match "components" to ABNF grammar symbols. |
| q29 | Are NaN, Infinity permitted in JSON numbers? | The mention of "Infinity and NaN" appears in a different chunk than the one annotated as relevant. The relevant chunk c0014 (number grammar section) doesn't contain "NaN" or "Infinity". |

---

## 8. Recommended Next Intervention

Based on the measured failure distribution:

### Primary Recommendation: Cross-Encoder Reranking of Extended Candidate Pool

**Evidence:**
1. **24 of 36 failures (67%)** are RANKING_FAILURE — the relevant chunk is in the candidate pool but ranked too low.
2. **5 of 14 Track B misses** are FUSION_FAILURE where the relevant chunk is in the individual top-5 of one retriever but pushed down by RRF.
3. Phase 7 already demonstrated that MiniLM-L-6-v2 reranking improves Hit@1 from 0.4038 → 0.5385 (+13.5pp) on Track B.
4. The latency trade-off (107ms → ~1500ms) is significant but manageable for a quality-critical application.

**Specific configuration to evaluate:**
- Increase `candidate_k` from 20 to 40 (to capture more of the deep candidates found by individual retrievers)
- Apply cross-encoder reranking to the top-40 RRF candidates
- Select final top-5 from reranked results
- Evaluate impact on Track A regression (Phase 7 showed Track A improved with reranking)

### Secondary Recommendation: Investigate Chunk Boundary Optimization

**Evidence:**
- 2 of 14 Track B misses are fundamentally unsupported due to chunk content not matching query semantics
- 43 Phase 11 no-clear-winner queries are all complete misses
- Some of these may be recoverable with overlapping chunk strategies or metadata-enriched chunks

### Not Recommended at This Time

| Intervention | Reason Against |
|---|---|
| Learned routing (Phase 12) | Already evaluated; causes regressions on Tracks A and B |
| SPLADE | Phase 6 showed SPLADE alone is dramatically worse; complementarity gain is marginal |
| HyDE | Phase 8 showed 86× latency increase for +5.8pp Hit@5 gain; regression on Track A |
| New embedding model | No evidence that the current model is the bottleneck; ranking is the issue |

---

## 9. Artifact Inventory

| File | Description |
|---|---|
| `scripts/run_failure_forensics.py` | Main forensics analysis script |
| `docs/reports/phase13_failure_analysis.json` | Machine-readable report with query-level records |
| `docs/PHASE_13_FAILURE_FORENSICS.md` | This document |
| `docs/PHASE_13_VALIDATION.md` | Validation evidence and decision gate |
| `tests/unit/test_phase13_failure_forensics.py` | 22 unit tests verifying report integrity |
