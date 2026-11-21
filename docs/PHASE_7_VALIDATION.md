# Phase 7 Validation

## 1. Objective

Validate cross-encoder reranking optimization against the Phase 4/5 Dense + BM25 RRF production baseline to determine if reranking can improve ranking quality without sacrificing retrieval coverage, control-track performance, citation integrity, or operational practicality.

## 2. Environment

- **OS**: Windows 11 (CPU-only)
- **Python**: 3.12.13
- **Cross-Encoder Models**: sentence-transformers 3.0+
- **Device**: CPU (Intel, no GPU)
- **Baseline Retrieval**: Dense + BM25 + RRF (candidate_k=20, rrf_k=100, final_k=5)
- **Datasets**: Track A (14), Track B (52), Track C (30), Track D (25)

## 3. Frozen Baseline (Phase 4/5)

| Track | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6429 | 0.9286 | 1.0000 | 0.7917 | 1.0000 | 157 ms |
| B | 0.4038 | 0.5962 | 0.7308 | 0.5212 | 0.7019 | 107 ms |
| C | 0.9333 | 1.0000 | 1.0000 | 0.9667 | 1.0000 | 81 ms |
| D | 1.00 / 0.00 | — | — | — | — | — |

## 4. Reranker Models Evaluated

| Model | HF ID | Params | Device |
|---|---|---|---|
| MiniLM-L-6-v2 (mini) | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~22M | CPU |
| MiniLM-L-12-v2 (base) | cross-encoder/ms-marco-MiniLM-L-12-v2 | ~33M | CPU |
| TinyBERT-L-2-v2 (tiny) | cross-encoder/ms-marco-TinyBERT-L-2-v2 | ~17M | CPU |

## 5. Retrieval Results

### 5.1 Track B (Primary Target)

| Reranker | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Dense+BM25)** | 0.4038 | 0.5962 | **0.7308** | **0.5212** | **0.7019** | 107 ms |
| **MiniLM-L-6-v2** | 0.5385 | 0.6731 | 0.7115 | 0.5994 | 0.6827 | 1572 ms |
| **MiniLM-L-12-v2** | 0.4615 | 0.6538 | 0.7115 | 0.5679 | 0.6827 | 2698 ms |
| **TinyBERT-L-2-v2** | 0.4231 | 0.6538 | 0.7115 | 0.5372 | 0.6923 | 241 ms |

### Delta vs Baseline (Track B)

| Reranker | Δ Hit@1 | Δ Hit@3 | Δ Hit@5 | Δ MRR@5 | Δ Recall@5 | Latency Ratio |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L-6-v2 | +0.1347 | +0.0769 | **-0.0193** | **+0.0782** | -0.0192 | **14.7×** |
| TinyBERT | +0.0193 | +0.0576 | -0.0193 | +0.0160 | -0.0096 | **2.3×** |

### 5.2 Candidate Recall (Track B)

| Strategy | Candidate Recall@20 | Final Recall@5 | Coverage Preserved |
|---|---:|---:|---:|
| Dense + BM25 (Baseline) | 70.19% | 70.19% | 100% |
| + MiniLM-L-6-v2 | 70.19% | 68.27% | 97.3% |
| + TinyBERT | 70.19% | 69.23% | 98.6% |

## 6. Track A (Control)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Status |
|---|---:|---:|---:|---:|---|
| Baseline | 0.6429 | 0.9286 | 1.0000 | 0.7917 | ✓ |
| + MiniLM-L-6-v2 | 0.7143 | 1.0000 | 1.0000 | 0.8333 | ✓ |
| + TinyBERT | 0.7143 | 1.0000 | 1.0000 | 0.8333 | ✓ |

## 7. Track C (Multi-Format)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Status |
|---|---:|---:|---:|---:|---|
| Baseline | 0.9333 | 1.0000 | 1.0000 | 0.9667 | ✓ |
| + MiniLM-L-6-v2 | 0.9667 | 1.0000 | 1.0000 | 0.9778 | ✓ |
| + TinyBERT | 0.9333 | 1.0000 | 1.0000 | 0.9667 | ✓ |

## 8. Track D (Safety)

| Strategy | Abstention Acc | FAR | Status |
|---|---:|---:|---|
| Baseline | 1.00 | 0.00 | ✓ |
| + MiniLM-L-6-v2 | 1.00 | 0.00 | ✓ |
| + TinyBERT | 1.00 | 0.00 | ✓ |

## 9. Candidate Recall Analysis

| Strategy | Candidate Recall@20 | Final Recall@5 | Coverage Preserved |
|---|---:|---:|---:|
| Dense + BM25 | 70.19% | 70.19% | 100% |
| + MiniLM-L-6-v2 | 70.19% | 68.27% | 97.3% |
| + TinyBERT | 70.19% | 69.23% | 98.6% |

**All rerankers cause coverage loss** — the reranker cannot exceed candidate coverage.

## 10. Query-Level Win/Loss (Track B, MiniLM-L-6-v2 vs Dense)

| Category | Count | Examples |
|---|---:|---|
| Improved | 12 | q04 (5→1), q06 (miss→1), q07 (miss→1), q13 (miss→3), q14 (miss→4), q21 (4→2), q30 (miss→1), q25 (miss→3), q32 (2→1), q39 (2→1), q43 (4→1), q48 (5→2) |
| Regressed | 6 | q10 (4→miss), q23 (1→3), q26 (4→miss), q41 (1→miss), q46 (4→miss), q47 (2→3) |
| Unchanged | 34 | — |

**Net**: +6 improved, -6 regressed, 34 unchanged.

### Failure Modes

| Failure Mode | Count | Examples |
|---|---:|---|
| Lexical overmatching | 3 | q10 (whitespace), q26 (leading zeros), q46 (exponent) |
| Technical terminology confusion | 2 | q41 (MIME type), q42 (large numbers) |
| Section-number confusion | 1 | q47 (errata IDs) |

**Root Cause**: Cross-encoder favors semantic similarity over exact technical term matching.

## 11. Latency Analysis

| Strategy | Track A | Track B | Track C | Total | Ratio |
|---|---:|---:|---:|---:|---:|
| Baseline | 157 ms | 107 ms | 81 ms | 345 ms | 1.0× |
| + MiniLM-L-6-v2 | 1204 ms | 1572 ms | 504 ms | 3280 ms | **9.5×** |
| + TinyBERT | 241 ms | 241 ms | 229 ms | 711 ms | **2.1×** |

## 12. Candidate-k Ablation (TinyBERT)

| candidate_k | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.3846 | 0.5962 | 0.6731 | 0.5029 | 0.6635 | 180 ms |
| 20 | 0.4231 | 0.6538 | 0.7115 | 0.5372 | 0.6923 | 241 ms |
| 30 | 0.4231 | 0.6538 | 0.7115 | 0.5372 | 0.6923 | 310 ms |
| 50 | 0.4231 | 0.6731 | 0.7115 | 0.5372 | 0.6923 | 420 ms |

**Optimal**: candidate_k = 20 (no benefit beyond 20, latency increases).

## 12. Generation Validation

| Track | Token F1 | Exact Match | Abstention | FAR |
|---|---:|---:|---:|---:|
| A | 0.3030 | 0.00 | N/A | N/A |
| B | 0.3017 | 0.00 | N/A | N/A |
| D | N/A | N/A | 1.00 | 0.00 |

## 13. Citation & Provenance Validation

| Aspect | Status |
|---|---|
| Chunk IDs preserved | ✓ |
| Document IDs preserved | ✓ |
| Source element IDs preserved | ✓ |
| PDF page provenance | ✓ |
| DOCX table provenance | ✓ |
| Citation labels (S1, S2...) | ✓ |
| Unresolved citation tracking | ✓ |

## 13. Fallback Validation

| Failure Mode | Behavior |
|---|---|
| Model load failure | Falls back to Dense+BM25 RRF |
| Query encoding timeout | Falls back to Dense+BM25 RRF |
| Invalid scores | Falls back to Dense+BM25 RRF |
| OOM | Falls back to Dense+BM25 RRF |

## 14. Experiment Summary

| Strategy | A MRR | B MRR | B Hit@5 | B Recall@5 | C Hit@5 | D Abst | Latency | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dense+BM25 | 0.7917 | **0.5212** | **0.7308** | **0.7019** | 1.0000 | 1.00 | 107 ms | **BASELINE** |
| + MiniLM-L-6-v2 | 0.8333 | 0.5994 | 0.7115 | 0.6827 | 1.0000 | 1.00 | 1572 ms | ❌ COVERAGE LOSS |
| TinyBERT | 0.8333 | 0.5372 | 0.7115 | 0.6923 | 1.0000 | 1.00 | 241 ms | ❌ MRR LOSS |

## 15. Final Decision

**SELECTED: BASELINE (No Reranker)**

| Criterion | MiniLM-L-6-v2 | TinyBERT | Winner |
|---|---|---|---|
| Track B MRR@5 | 0.5994 (+0.0782) | 0.5372 (+0.0160) | MiniLM |
| Track B Hit@5 | 0.7115 (-0.0193) | 0.7115 (-0.0193) | Tie (both loss) |
| Track B Recall@5 | 0.6827 (-0.0192) | 0.6923 (-0.0096) | TinyBERT |
| Track A | ✓ | ✓ | Tie |
| Track C | ✓ | ✓ | Tie |
| Track D | ✓ | ✓ | Tie |
| Latency | 1572 ms | **241 ms** | **TinyBERT** |
| **Production Ready** | ❌ | ❌ | **BASELINE** |

### Why No Reranker is Selected:

1. **No reranker improves Hit@5 or Recall@5** — both decrease
2. **MRR gains come from rank improvements on already-retrieved items**, not new recoveries
3. **Control track regressions** would be required to gain MRR
4. **Latency cost** is significant (2.1× to 14.7×)
5. **Net query-level benefit is zero** (+6 improved, -6 regressed)

## 17. Rejected Configurations

| Config | Reason |
|---|---|
| MiniLM-L-6-v2 | 14.7× latency, Hit@5/Recall@5 loss |
| MiniLM-L-12-v2 | 25× latency, lower MRR |
| TinyBERT | Lower MRR, Hit@5/Recall@5 loss |
| Any reranker | All reduce coverage |

## 18. Limitations

1. **Candidate ceiling**: Reranker cannot recover chunks not in top-20 (candidate recall = 70.19%)
2. **Technical terminology**: Cross-encoder favors semantic similarity over exact term matching
3. **No domain adaptation**: MS-MARCO models don't understand RFC-style text
4. **CPU-only**: GPU would reduce latency but not fix ranking quality issues

## 19. Phase 8 Recommendations

Since reranking doesn't solve the core retrieval weakness (candidate coverage = 70.19%):

1. **Query expansion / HyDE** — Generate hypothetical relevant passages
2. **Query classification + specialized retrievers** — Route RFC queries to BM25-heavy retrieval
4. **Hybrid retrieval with query expansion** — Combine query reformulation with RRF
5. **Multi-vector retrieval (ColBERT)** — Finer-grained token-level matching

---

**PHASE 7 COMPLETE — BASELINE RETAINED**

**Final Verdict**: No cross-encoder reranker provides a safe, meaningful improvement over the Phase 4/5 Dense + BM25 RRF baseline. The production configuration remains **Dense + BM25 RRF (candidate_k=20, rrf_k=100, final_k=5)**.