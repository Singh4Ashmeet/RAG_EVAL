# Phase 6 SPLADE Learned Sparse Retrieval

## 1. Objective

Evaluate whether learned sparse retrieval (SPLADE) can improve retrieval performance over the Phase 4/5 production baseline (Dense + BM25 with RRF) on Track B, while preserving Track A, Track C, and Track D controls.

## 2. Frozen Baseline (Phase 4/5)

| Component | Value |
|---|---|
| Chunking | ElementAwareChunker, 1000/200 |
| Embedding | nomic-embed-text:latest (dim=768) |
| Generation | llama3.2:3b |
| Retrieval | Dense + BM25 + RRF (candidate_k=20, rrf_k=100, final_k=5) |
| Query Strategy | Identity (no transformation) |

### Phase 4/5 Production Baseline (Dense + BM25 + RRF)

| Track | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6429 | 0.9286 | 1.0000 | 0.7917 | 1.0000 | 85 ms |
| B | 0.4038 | 0.5962 | 0.7308 | 0.5212 | 0.7019 | 107 ms |
| C | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 1.0000 | 67 ms |
| D | 1.00 / 0.00 | — | — | — | — | — |

## 3. SPLADE Model Selection

**Model**: `naver/splade-cocondenser-ensembledistil` (SentenceTransformers compatible)

**Rationale**:
- Well-maintained Hugging Face model
- CPU compatible (runs on CPU in ~79ms/query)
- Distilled from SPLADE ensemble for efficiency
- Produces sparse vectors compatible with exact dot-product scoring
- 373 average non-zero dimensions per vector (vs 30,522 vocab size)

**Configuration**:
- Max sequence length: 512
- Sparse threshold: 0.0 (keep all positive weights)
- Top-k tokens: None (keep all positive activations)
- Batch size: 32
- ReLU activation applied to embeddings

## 4. Ablation Matrix Results (Track B)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| **Dense** | 0.4231 | 0.5192 | 0.6538 | 0.5029 | 0.6250 | 142 ms |
| **BM25** | 0.4038 | 0.5769 | 0.6731 | 0.5019 | 0.6346 | 0.7 ms |
| **SPLADE** | 0.0577 | 0.2308 | 0.4231 | 0.1798 | 0.4135 | 79 ms |
| **Dense + BM25 (Baseline)** | 0.3846 | 0.5962 | **0.7308** | **0.5212** | **0.7019** | 107 ms |
| Dense + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 299 ms |
| BM25 + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 300 ms |
| Dense + BM25 + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 273 ms |

## 5. Complementarity Analysis (Track B)

| Category | Count | Description |
|---|---:|---|
| All three retrievers | 12 | Dense + BM25 + SPLADE all found relevant chunk |
| Dense + BM25 only | 16 | SPLADE missed, but Dense+BM25 found |
| Dense + SPLADE only | 4 | BM25 missed, but Dense+SPLADE found |
| BM25 + SPLADE only | 4 | Dense missed, but BM25+SPLADE found |
| Dense only | 2 | Only Dense found |
| BM25 only | 3 | Only BM25 found |
| SPLADE only | 2 | Only SPLADE found |
| None | 9 | No retriever found relevant chunk |

### Key Complementarity Findings

- **SPLADE recovered 2 queries that neither Dense nor BM25 found**: q28, q45
- **SPLADE + BM25 recovered 4 queries that Dense missed**
- **SPLADE + Dense recovered 4 queries that BM25 missed**

**SPLADE-only queries**: q28 (escape characters), q45 (errata IDs)
**BM25+SPLADE-only queries**: q06, q07, q16, q30

## 6. Track A Control (Regression Check)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense + BM25 (Baseline) | 0.6429 | 0.9286 | **1.0000** | **0.7917** |
| Dense + BM25 + SPLADE | 0.6429 | 0.8571 | 0.9286 | 0.7524 |

**REGRESSION**: Hit@5 drops from 1.0000 → 0.9286 (-7.1 pp), MRR@5 drops from 0.7917 → 0.7524 (-0.0393)

## 6. Track C Control (Regression Check)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense + BM25 (Baseline) | 0.9667 | 1.0000 | **1.0000** | **0.9833** |
| Dense + BM25 + SPLADE | 0.7667 | 1.0000 | 1.0000 | 0.8722 |

**REGRESSION**: Hit@1 drops from 0.9667 → 0.7667 (-20 pp), MRR@5 drops from 0.9833 → 0.8722 (-0.1111)

## 7. Track D Safety Validation

| Strategy | Abstention Accuracy | False Answer Rate |
|---|---:|---:|
| Baseline (Dense+BM25) | 1.00 | 0.00 |
| Dense + BM25 + SPLADE | 1.00 | 0.00 |

**Track D remains safe** - all 25 unanswerable queries correctly abstained.

## 7. Query-Level Diagnostics (Track B)

### SPLADE-only Recoveries (2 queries)
| Query | Topic | Relevant Chunk |
|---|---|---|
| q28 | UTF-16 surrogate pairs | doc_61a5378f4255c720_c0017 |
| q45 | Errata IDs applied | doc_61a5378f4255c720_c0027 |

### BM25 + SPLADE Only (Dense Missed) - 4 queries
| Query | Topic |
|---|---|
| q06 | UTF-8 encoding |
| q07 | JSON string escaping |
| q16 | Comma in arrays |
| q30 | Escape characters |

### Dense + SPLADE Only (BM25 Missed) - 4 queries
| Query | Topic |
|---|---|
| q10 | Whitespace characters |
| q44 | Forward slash escaping |
| q46 | Case sensitivity of literals |
| q47 | Exponent case sensitivity |

## 8. Latency Analysis

| Strategy | Track A | Track B | Track C | Total |
|---|---:|---:|---:|---:|
| Dense | 142 ms | 142 ms | 67 ms | ~351 ms |
| BM25 | 85 ms | 0.7 ms | 65 ms | ~151 ms |
| SPLADE | 79 ms | 79 ms | 76 ms | ~234 ms |
| Dense + BM25 | 85 ms | 107 ms | 67 ms | ~259 ms |
| Dense + BM25 + SPLADE | 387 ms | 273 ms | 229 ms | ~889 ms |

**SPLADE adds ~3.5x latency** over Dense+BM25 baseline.

## 9. Index Statistics

| Metric | BM25 | SPLADE |
|---|---:|---:|
| Chunks indexed | 30 | 30 |
| Total non-zero entries | N/A (term freq) | 11,190 |
| Avg non-zero per chunk | N/A | 373 |
| Max non-zero per chunk | N/A | ~600 |
| Index build time | <1 ms | ~2.5 s |
| Query encoding time | <1 ms | ~79 ms |

## 9. Track A, B, C, D Summary

| Track | Baseline (Dense+BM25) | +SPLADE | Delta | Status |
|---|---|---|---|---|
| A Hit@5 | 1.0000 | 0.9286 | -7.1 pp | ❌ REGRESSION |
| A MRR@5 | 0.7917 | 0.7524 | -0.0393 | ❌ REGRESSION |
| B Hit@5 | 0.7308 | 0.7308 | 0.0000 | ✓ NEUTRAL |
| B MRR@5 | 0.5212 | 0.4462 | -0.0750 | ❌ REGRESSION |
| C Hit@5 | 1.0000 | 1.0000 | 0.0000 | ✓ |
| C Hit@1 | 0.9667 | 0.7667 | -20 pp | ❌ REGRESSION |
| C MRR@5 | 0.9833 | 0.8722 | -0.1111 | ❌ REGRESSION |
| D Safety | 1.00/0.00 | 1.00/0.00 | — | ✓ |

## 10. SPLADE Index Statistics

| Metric | Value |
|---|---:|
| Model | naver/splade-cocondenser-ensembledistil |
| Embedding dim | 30,522 (vocab size) |
| Avg non-zero per query | 373 |
| Avg non-zero per chunk | 373 |
| Total non-zero entries | 11,190 |
| Sparsity | ~98.8% |
| Index build time | ~2.5 s (30 chunks) |
| Query encode time | ~79 ms |
| Retrieval time | ~0.5 ms |

## 11. Complementarity Summary

| SPLADE Contribution | Count | Significance |
|---|---:|---|
| SPLADE-only (unique) | 2 | Low - but genuine complementarity |
| SPLADE+BM25 (Dense missed) | 4 | Moderate - recovers Dense failures |
| SPLADE+Dense (BM25 missed) | 4 | Moderate - recovers BM25 failures |
| All three agree | 12 | High confidence retrievals |

## 12. Conclusion

**SPLADE does not improve the production baseline.**

### Key Findings:
1. **No improvement on Track B primary metrics**: Hit@5 unchanged (0.7308), MRR@5 regresses (-0.0750)
2. **Regressions on control tracks**: Track A (-7.1 pp Hit@5), Track C (-20 pp Hit@1, -0.1111 MRR@5)
3. **Limited unique value**: Only 2 queries recovered exclusively by SPLADE
3. **Latency cost**: 3.5x slower than Dense+BM25 baseline
4. **Control track regressions**: Significant degradation on Track A and Track C

### Why SPLADE Underperforms:
- **Single-document corpus**: RFC 8259 is a single 28KB document - SPLADE excels at multi-document semantic matching
- **Vocabulary mismatch**: SPLADE's vocabulary expansion doesn't help when the exact terminology is already in the query
- **Noisy activations**: SPLADE adds noise that hurts RRF fusion quality
- **RRF dilution**: Adding a third weak ranker dilutes the strong Dense+BM25 signals

## 13. COIL Decision

**COIL is NOT justified** for Phase 6B because:
- SPLADE showed no meaningful improvement
- Complementarity was minimal (2 unique recoveries)
- Control track regressions outweigh benefits
- Corpus characteristics (single technical document) don't match COIL's strengths

## 14. Final Recommendation

**RETAIN Phase 5 Production Baseline**: Dense + BM25 + RRF (candidate_k=20, rrf_k=100)

**SPLADE Branch Status**: COMPLETE - NO IMPROVEMENT

The learned sparse retrieval experiment demonstrated that SPLADE does not provide sufficient complementary signal to justify its latency cost and control track regressions on this corpus. The Phase 4/5 Dense+BM25 RRF baseline remains the production configuration.

## 14. Future Directions (Phase 7)

If further retrieval improvement is needed:
1. **Query classification + specialized retrievers** (route RFC questions differently)
2. **HyDE / query expansion** with LLM-generated hypothetical documents
3. **Cross-encoder reranking** with better models (if latency budget allows)
4. **Multi-vector representations** (ColBERT-style)
5. **Learned sparse retrieval on multi-document corpora** (where SPLADE excels)

---

**PHASE 6 COMPLETE — SPLADE BRANCH COMPLETE — BASELINE RETAINED**