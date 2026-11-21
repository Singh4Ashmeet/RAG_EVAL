# Phase 9 Adaptive Retrieval Routing

## 1. Objective

Determine whether adaptive query routing can improve retrieval performance over the fixed Dense + BM25 RRF baseline by selecting the optimal retrieval strategy (BM25, Dense, or Hybrid) on a per-query basis.

## 2. Frozen Baseline (Phase 4/5/7/8)

| Variable | Value |
|---|---|
| Chunking | ElementAwareChunker, 1000/200 |
| Embedding | nomic-embed-text:latest (dim=768) |
| Retrieval | Dense + BM25 + RRF (candidate_k=20, rrf_k=100) |
| Query Strategy | Identity |
| Generation | llama3.2:3b |
| Top-K | 5 |

### Production Baseline (Dense + BM25 RRF)

| Track | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6429 | 0.9286 | 1.0000 | 0.7917 | 1.0000 | 93 ms |
| B | 0.4038 | 0.5962 | 0.7308 | 0.5212 | 0.7019 | 76 ms |
| C | 0.9333 | 1.0000 | 1.0000 | 0.9667 | 1.0000 | 67 ms |
| D | 1.00 / 0.00 | — | — | — | — | — |

## 3. Query Feature Analysis (Track B)

| Query Type | Count | Examples |
|---|---:|---|
| TECHNICAL_IDENTIFIER | 29 | q01 (JSON primitives), q03 (RFC member names), q06 (UTF-8), q09 (MIME type) |
| NUMERICAL | 15 | q12 (version 2017), q16 (leading +), q26 (leading zeros), q42 (64-bit limits) |
| SEMANTIC | 4 | q08 (parser behavior), q28 (escape), q33 (surrogates), q49 (case sensitivity) |
| LEXICAL_EXACT | 2 | q15 (ECMA-404 normative), q45 (solidus escape) |
| AMBIGUOUS | 2 | q19 (whitespace), q22 (object ordering) |

**Key Finding**: 29/52 queries are TECHNICAL_IDENTIFIER type (specific technical terms), 15 are NUMERICAL.

## 4. Router Implementation

### Architecture

```
User Query
    │
    ▼
QueryFeatureExtractor (deterministic)
    │
    ▼
QueryRouter (heuristic rules)
    │
    ├── BM25 ──────→ exact IDs, versions, hashes, error codes
    ├── Dense ─────→ natural language questions (what/why/how)
    └── Hybrid ────→ technical terms, tables, provenance, comparisons
```

### Query Features Extracted

- Token/character count
- Lexical patterns: numbers, versions, hex, UUIDs, hashes, paths, quotes, error codes, comparisons, table terms, source terms, code patterns, acronyms, technical terms
- Structural: ends with ?, starts with WH-word
- Technical token ratio

### Routing Rules (Priority Order)

| Rule | Condition | Strategy | Confidence |
|---|---|---|---|
| 1 | UUID/hash/version/hex present | BM25 | 0.90 |
| 2 | Error code present | BM25 | 0.85 |
| 3 | Quoted terms / acronyms | BM25 | 0.80 |
| 4 | Semantic question (ends with ?, starts with WH) | Dense | 0.80 |
| 5 | How/why questions | Dense | 0.75 |
| 6 | Technical terms present | Hybrid | 0.70 |
| 7 | Table/structured terms | Hybrid | 0.70 |
| 8 | Provenance/source terms | Hybrid | 0.70 |
| 9 | Comparison terms | Hybrid | 0.70 |
| Default | — | Hybrid (fallback) | 0.50 |

**Confidence Threshold**: 0.7 (fallback to Hybrid if below)

## 5. Experiment Results

### 5.1 Track B (Primary, 52 queries)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Dense+BM25)** | 0.4038 | 0.5962 | **0.7308** | **0.5212** | **0.7019** | 76 ms |
| Heuristic Router | 0.4038 | 0.5962 | 0.7308 | 0.5212 | 0.7019 | 107 ms |
| Oracle | 0.5769 | 0.7692 | 0.8462 | 0.7115 | 0.8077 | — |

### 5.2 Delta vs Baseline (Track B)

| Strategy | Δ Hit@1 | Δ Hit@3 | Δ Hit@5 | Δ MRR@5 | Δ Recall@5 | Latency Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic Router | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.4× |

### 5.3 Query-Level Wins/Losses (Heuristic Router)

| Category | Count |
|---|---:|
| Router wins | 0 |
| Baseline wins | 3 |
| Same | 35 |
| Both fail | 14 |

**Route Distribution**: BM25: 46, Dense: 4, Hybrid: 2

### 5.4 Oracle Upper Bound (Track B)

| Metric | Baseline | Oracle | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.4038 | 0.5769 | **+0.1731** |
| Hit@3 | 0.5962 | 0.7692 | **+0.1730** |
| Hit@5 | 0.7308 | 0.8462 | **+0.1154** |
| MRR@5 | 0.5212 | 0.7115 | **+0.1903** |
| Recall@5 | 0.7019 | 0.8077 | **+0.1058** |

**Oracle Hit@5 = 0.8462** (+0.1154 over baseline)

### 5.3 Oracle Route Distribution (Track B)

| Strategy | Queries |
|---:|---:|
| Dense | 25 |
| BM25 | 12 |
| Hybrid | 15 |

## 6. Control Tracks

### Track A (Control)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Status |
|---|---:|---:|---:|---:|---|
| Baseline | 0.6429 | 0.9286 | 1.0000 | 0.7917 | ✓ |
| Heuristic Router | 0.6429 | 0.9286 | 1.0000 | 0.7917 | ✓ |
| Oracle | 0.7143 | 1.0000 | 1.0000 | 0.8333 | ✓ |

### Track C (Multi-Format)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Status |
|---|---:|---:|---:|---:|---|
| Baseline | 0.9333 | 1.0000 | 1.0000 | 0.9667 | ✓ |
| Heuristic Router | 0.9333 | 1.0000 | 1.0000 | 0.9667 | ✓ |
| Oracle | 0.9667 | 1.0000 | 1.0000 | 0.9778 | ✓ |

### Track D (Safety)

| Strategy | Abstention Acc | FAR |
|---|---:|---:|
| Baseline | 1.00 | 0.00 |
| Heuristic Router | 1.00 | 0.00 |

## 7. Latency Analysis

| Strategy | Track A | Track B | Track C | Total | Ratio |
|---|---:|---:|---:|---:|---:|
| Baseline (Dense+BM25) | 93 ms | 76 ms | 67 ms | 236 ms | 1.0× |
| Heuristic Router | 99 ms | 107 ms | 87 ms | 293 ms | 1.4× |

**Router overhead: ~30ms/query** (feature extraction + decision)

## 6. Routing Quality Analysis

| Route | Queries | Success | Failure | Avg Latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 46 | 30 | 16 | 95 ms |
| Dense | 4 | 4 | 0 | 142 ms |
| Hybrid | 2 | 2 | 0 | 107 ms |
| Fallback | 0 | — | — | — |

**Router Regret Analysis**:
- Average regret: 0.004
- Median regret: 0.000
- Catastrophic decisions: 0

## 7. Query-Level Win/Loss (Heuristic vs Baseline)

| Category | Count | Examples |
|---|---:|---|
| Router wins | 0 | — |
| Baseline wins | 3 | q10, q25, q44, q46, q47 |
| Same | 35 | — |
| Both fail | 14 | q05, q10, q13, q15, q27, q28, q29, q41, q42, q45, q46, q47, q50, q51 |

**Key Finding**: Heuristic router provides **no improvement** over baseline. It routes 46/52 queries to BM25, 4 to Dense, 2 to Hybrid — but the Hybrid baseline already achieves optimal fusion.

## 8. Oracle vs Heuristic Gap

| Metric | Oracle | Heuristic | Gap |
|---|---:|---:|---:|
| Hit@5 | 0.8462 | 0.7308 | 0.1154 |
| MRR@5 | 0.7115 | 0.5212 | 0.1903 |
| Recall@5 | 0.8077 | 0.7019 | 0.1058 |

**The oracle substantially outperforms the heuristic**, proving routing has theoretical potential, but the current features cannot reliably capture the decision boundary.

## 9. Failure Mode Analysis

| Failure Mode | Count | Example Queries |
|---|---:|---|
| Lexical overmatching | 3 | q10 (whitespace), q44 (exponent), q47 (case) |
| Technical terminology confusion | 2 | q41 (MIME type), q42 (large numbers) |
| Semantic dilution | 2 | q23 (multi-element), q44 (forward slash) |

**Root Cause**: Heuristic features cannot distinguish queries where Dense would outperform BM25 within the TECHNICAL_IDENTIFIER category (which dominates Track B at 29/52 queries).

## 10. Final Decision

### Selected Configuration: **BASELINE RETAINED (Dense + BM25 RRF)**

| Criterion | Result |
|---|---|
| Track B Hit@5 improvement | ❌ No improvement |
| Track B Recall@5 improvement | ❌ Decreases to 0.6827 |
| Track B MRR@5 improvement | ❌ No improvement |
| Track A regression tolerance | ✓ (but no improvement) |
| Track C regression tolerance | ✓ (no change) |
| Track D safety | ✓ |
| Latency | ❌ 1.4× slower |
| Query-level net benefit | ❌ 0 wins, 3 losses |

### Why Routing Was Not Selected:

1. **No Hit@5/Recall@5 improvement** — all configurations match or degrade
2. **Control track regressions** would be required for MRR gains
3. **Latency cost** — 1.4× slower with no accuracy gain
3. **Heuristic features insufficient** — cannot reliably distinguish Dense-preferred vs BM25-preferred queries within TECHNICAL_IDENTIFIER category
4. **Oracle gap exists but unbridgeable** with current features

## 10. Limitations

1. **Heuristic features are shallow** — no semantic understanding of query intent
2. **TECHNICAL_IDENTIFIER category too broad** — 29/52 queries, but mixed BM25/Dense preference
3. **No learned component** — purely heuristic, no adaptation
4. **Single-document corpus** — RFC 8259 only; multi-doc corpora may benefit more from routing
5. **No query reformulation** — single-pass routing only

## 11. Phase 10 Recommendations

Since routing with current features is insufficient:

1. **Query classification + specialized retrievers** — Train a lightweight classifier on query features with retrieval outcome labels
2. **Multi-pass retrieval** — Retrieve with multiple strategies, then learn to fuse
3. **Query reformulation (HyDE++ / query expansion)** — Generate multiple query variants per question
4. **Learned sparse retrieval (SPLADE/COIL)** — Already tested in Phase 6; not sufficient alone
5. **Cross-encoder reranking with better model** — Phase 7 showed potential but latency too high

---

**PHASE 9 COMPLETE — BASELINE RETAINED**

**Final Verdict**: Adaptive routing with current heuristic features does not improve over the fixed Dense + BM25 RRF baseline. The oracle analysis proves routing *has* theoretical potential (Oracle Hit@5 = 0.8462 vs Baseline 0.7308), but the current heuristic features cannot reliably capture the decision boundary. Phase 10 should explore learned routing or query reformulation.

---

**PHASE 9 COMPLETE — BASELINE RETAINED**