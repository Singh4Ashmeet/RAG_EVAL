# Phase 6 Validation

## 1. Objective

Evaluate learned sparse retrieval (SPLADE) against the Phase 4/5 production baseline (Dense + BM25 with RRF) to determine if learned sparse retrieval provides meaningful complementary signal on Track B.

## 2. Environment

- **OS**: Windows 11 (CPU-only)
- **Python**: 3.12.13
- **Model**: naver/splade-cocondenser-ensembledistil (SentenceTransformers)
- **Device**: CPU (Intel, no GPU)
- **Corpus**: RFC 8259 single document (28KB, 30 chunks @ 1000/200)

## 3. Frozen Baseline (Phase 4/5)

| Variable | Value |
|---|---|
| Chunking | ElementAwareChunker, 1000/200 |
| Embedding | nomic-embed-text:latest (dim=768) |
| Retrieval | Dense + BM25 + RRF (candidate_k=20, rrf_k=100) |
| Query Strategy | Identity |
| Track B Baseline | Hit@1=0.4038, Hit@3=0.5962, Hit@5=0.7308, MRR@5=0.5212, Recall@5=0.7019 |

## 4. SPLADE Configuration

| Parameter | Value |
|---|---|
| Model | naver/splade-cocondenser-ensembledistil |
| Framework | SentenceTransformers 3.0+ |
| Device | CPU |
| Max Length | 512 |
| Sparse Threshold | 0.0 |
| Top-K Tokens | None (all positive) |
| Batch Size | 32 |
| Activation | ReLU |
| Vocab Size | 30,522 |

## 5. Retrieval Results

### 5.1 Standalone Retrievers (Track B)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.4231 | 0.5192 | 0.6538 | 0.5029 | 0.6250 | 142 ms |
| BM25 | 0.4038 | 0.5769 | 0.6731 | 0.5019 | 0.6346 | 0.7 ms |
| SPLADE | 0.0577 | 0.2308 | 0.4231 | 0.1798 | 0.4135 | 79 ms |

### 5.2 Fusion Strategies (Track B)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense + BM25 (Baseline) | 0.3846 | 0.5962 | **0.7308** | **0.5212** | **0.7019** | 107 ms |
| Dense + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 299 ms |
| BM25 + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 300 ms |
| Dense + BM25 + SPLADE | 0.2692 | 0.6346 | 0.7308 | 0.4462 | 0.7019 | 273 ms |

### 5.3 Track A Results

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense + BM25 (Baseline) | 0.6429 | 0.9286 | **1.0000** | **0.7917** |
| Dense + BM25 + SPLADE | 0.6429 | 0.8571 | 0.9286 | 0.7524 |

**REGRESSION**: Hit@5 -7.1 pp, MRR@5 -0.0393

### 5.4 Track C Results

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense + BM25 (Baseline) | 0.9667 | 1.0000 | **1.0000** | **0.9833** |
| Dense + BM25 + SPLADE | 0.7667 | 1.0000 | 1.0000 | 0.8722 |

**REGRESSION**: Hit@1 -20 pp, MRR@5 -0.1111

### 5.5 Track D Safety

| Strategy | Abstention Accuracy | False Answer Rate |
|---|---:|---:|
| Baseline | 1.00 | 0.00 |
| Dense + BM25 + SPLADE | 1.00 | 0.00 |

## 6. Ablation Results

### 6.1 Complementarity Analysis (Track B, 52 queries)

| Category | Count | % |
|---|---:|---:|
| All three retrievers | 12 | 23% |
| Dense + BM25 only | 16 | 31% |
| Dense + SPLADE only | 4 | 8% |
| BM25 + SPLADE only | 4 | 8% |
| Dense only | 2 | 4% |
| BM25 only | 3 | 6% |
| SPLADE only | 2 | 4% |
| None | 9 | 17% |

### 6.2 SPLADE Unique Contributions

| Contribution | Count | Queries |
|---|---:|---|
| SPLADE only (unique) | 2 | q28 (UTF-16 surrogates), q45 (errata IDs) |
| SPLADE + BM25 (Dense missed) | 4 | q06, q07, q16, q30 |
| SPLADE + Dense (BM25 missed) | 4 | q10, q44, q46, q47 |
| All three agree | 12 | High confidence |

## 7. Query-Level Diagnostics (Track B)

### SPLADE Unique Recoveries (2 queries)
| Query | Topic | Relevant Chunk |
|---|---|---|
| q28 | UTF-16 surrogate pairs | doc_61a5378f4255c720_c0017 |
| q45 | Errata IDs applied | doc_61a5378f4255c720_c0027 |

### BM25 + SPLADE Only (4 queries Dense missed)
| Query | Topic |
|---|---|
| q06 | UTF-8 encoding MUST requirement |
| q07 | JSON string escape rules |
| q16 | Comma separator in arrays |
| q30 | Escape characters in strings |

### Dense + SPLADE Only (4 queries BM25 missed)
| Query | Topic |
|---|---|
| q10 | Permitted whitespace characters |
| q44 | Solidus (/) escaping optional |
| q46 | Case sensitivity of literals |
| q47 | Exponent case (e/E) |

## 8. Latency Analysis

| Strategy | Track A | Track B | Track C | Total |
|---|---:|---:|---:|---:|
| Dense | 142 ms | 142 ms | 67 ms | 351 ms |
| BM25 | 85 ms | 0.7 ms | 65 ms | 151 ms |
| SPLADE | 79 ms | 79 ms | 76 ms | 234 ms |
| Dense + BM25 | 85 ms | 107 ms | 67 ms | 259 ms |
| Dense + BM25 + SPLADE | 387 ms | 273 ms | 229 ms | 889 ms |

**SPLADE adds 3.4x latency** over Dense+BM25 baseline.

## 9. SPLADE Index Statistics

| Metric | Value |
|---|---:|
| Model | naver/splade-cocondenser-ensembledistil |
| Vocab Size | 30,522 |
| Chunks Indexed | 30 |
| Total Non-Zero Entries | 11,190 |
| Avg Non-Zero/Chunk | 373 |
| Max Non-Zero/Chunk | ~600 |
| Sparsity | 98.8% |
| Index Build Time | ~2.5 s |
| Query Encode Time | ~79 ms |
| Retrieval Time | ~0.5 ms |

## 10. Control Track Validation

| Track | Baseline Hit@5 | +SPLADE Hit@5 | Delta | Status |
|---|---:|---:|---:|---|
| A | 1.0000 | 0.9286 | -7.1 pp | ❌ REGRESSION |
| B | 0.7308 | 0.7308 | 0.0 | NEUTRAL |
| C | 1.0000 | 1.0000 | 0.0 | ✓ |
| C Hit@1 | 0.9667 | 0.7667 | -20 pp | ❌ REGRESSION |

| Track | Baseline MRR@5 | +SPLADE MRR@5 | Delta | Status |
|---|---:|---:|---:|---|
| A | 0.7917 | 0.7524 | -0.0393 | ❌ REGRESSION |
| B | 0.5212 | 0.4462 | -0.0750 | ❌ REGRESSION |
| C | 0.9833 | 0.8722 | -0.1111 | ❌ REGRESSION |

## 11. Track D Safety

| Strategy | Abstention Accuracy | False Answer Rate |
|---|---:|---:|
| Baseline | 1.00 | 0.00 |
| Dense + BM25 + SPLADE | 1.00 | 0.00 |

**Track D remains safe.**

## 11. Citation & Provenance Validation

| Aspect | Status |
|---|---|
| Chunk IDs preserved | ✓ |
| Document IDs preserved | ✓ |
| Source element IDs preserved | ✓ |
| PDF page provenance | ✓ |
| DOCX table provenance | ✓ |
| Citation labels (S1, S2...) | ✓ |
| Unresolved citation tracking | ✓ |

## 12. Generation Validation

| Track | Token F1 | Status |
|---|---:|---|
| A | ~0.30 | Comparable |
| B | ~0.30 | Comparable |
| D | N/A | 1.00 abstention, 0.00 FAR |

## 12. Failure Handling

| Failure Mode | Behavior |
|---|---|
| Model load failure | Falls back to Dense+BM25 |
| Query encoding timeout | Falls back to Dense+BM25 |
| Invalid sparse output | Falls back to Dense+BM25 |
| OOM | Falls back to Dense+BM25 |

## 13. Final Recommendation

**RETAIN PHASE 4/5 BASELINE: Dense + BM25 + RRF**

### Why SPLADE Was Not Selected:

| Criterion | Result |
|---|---|
| Track B MRR@5 improvement | ❌ -0.0750 (regression) |
| Track B Hit@5 improvement | ❌ No change |
| Track B Recall@5 improvement | ❌ No change |
| Track A regression tolerance | ❌ -7.1 pp Hit@5 |
| Track C regression tolerance | ❌ -20 pp Hit@1, -0.11 MRR@5 |
| Track D safety | ✓ Maintained |
| Latency cost | ❌ 3.4x slower |
| Unique SPLADE recoveries | 2/52 queries |
| COIL justification | ❌ Not warranted |

## 14. COIL Decision

**COIL NOT WARRANTED** because:
1. SPLADE showed no meaningful improvement on primary metrics
2. Complementarity was minimal (2 unique recoveries out of 52)
3. Control track regressions are significant
3. Single-document corpus doesn't match COIL's design strengths
4. Latency cost is prohibitive for production

## 15. Limitations

1. **Single-document corpus**: RFC 8259 is a single 28KB technical document - SPLADE excels at multi-document semantic matching
2. **Clean technical text**: RFC 8259 has precise terminology already captured by BM25
3. **CPU-only**: GPU would reduce SPLADE latency to ~20ms but not fix relevance issues
4. **No LLM reformulation**: Could expand queries for SPLADE but adds latency
5. **Single SPLADE model**: Only tested ensembledistil variant

## 16. Phase 7 Recommendations

1. **Query classification + routing** - Route RFC questions to BM25-heavy retrieval
2. **HyDE / hypothetical documents** - Generate synthetic relevant passages for expansion
3. **Cross-encoder reranking** - With better models if latency budget increases
4. **Multi-vector (ColBERT-style)** - For finer-grained matching
5. **Learned sparse on multi-doc corpora** - SPLADE may excel where BM25 fails

---

**PHASE 6 COMPLETE — SPLADE BRANCH COMPLETE — BASELINE RETAINED**

**Final Verdict**: Learned sparse retrieval (SPLADE) does not provide sufficient complementary signal to justify its latency cost and control track regressions on this corpus. The Phase 4/5 Dense+BM25 RRF baseline remains the production configuration.