# PHASE 14 — TARGETED RANKING OPTIMIZATION & RRF FORENSICS

## 1. Executive Summary

Phase 13 identified the primary retrieval bottleneck in RAG_EVAL as **RANKING_FAILURE**, accounting for **66.7%** (24 of 36) of all evaluated failures across protected tracks. Specifically:
- **Track A:** 5 ranking failures (relevant chunks present in candidate pool at ranks 2–4, Hit@5=1.0000).
- **Track B:** 19 ranking failures, 5 fusion failures, 3 identifier tokenization issues, 2 semantic mismatches, 2 lexical mismatches (Hit@5=0.7308, 14 complete misses).
- **Track C:** 0 failures (Hit@1=1.0000 across all 30 queries).
- **Track D:** 100% abstention safe (25/25 queries).

Phase 14 performed a controlled, leakage-free empirical investigation of the **Reciprocal Rank Fusion (RRF) ranking layer** to determine whether parameter tuning ($rrf\_k$, $candidate\_k$) or weighted fusion could eliminate ranking failures without regressing safety or retrieval quality.

**Final Phase 14 Decision:**
```text
PHASE 14 COMPLETE — BASELINE RETAINED
```

The frozen production baseline remains active and unchanged:
- **Chunker:** `ElementAwareChunker` (`chunk_size=1000`, `overlap=200`)
- **Embedding:** `nomic-embed-text:latest` (768 dimensions)
- **Retrieval:** Dense + BM25 RRF (`candidate_k=20`, `rrf_k=100`, `final_k=5`)
- **Generation:** `llama3.2:3b`
- **Safety:** Citations, provenance, and Track D abstention enabled.

---

## 2. Frozen Production Baseline Metrics

Before experimentation, the baseline was verified across all 121 protected queries (14 Track A + 52 Track B + 30 Track C + 25 Track D):

| Track | Queries | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency (Mean) | Safety Status |
|---|---|---|---|---|---|---|---|---|
| **Track A** | 14 | 0.6429 | 0.9286 | 1.0000 | 0.7917 | 1.0000 | 22.7 ms | 100% Hit@5 |
| **Track B** | 52 | 0.3846 | 0.5962 | 0.7308 | 0.5115 | 0.7019 | 25.6 ms | RFC 8259 Standard |
| **Track C** | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11.7 ms | Robust (Adversarial) |
| **Track D** | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A | 25/25 Safe Abstention |

*Note on Track B Hit@1:* When ties are broken deterministically by chunk identifier, Track B baseline achieves Hit@1=0.3846 (20/52) vs 0.4038 (21/52) under arbitrary set hash ordering on `track_b_q20` and `track_b_q38` where Dense and BM25 produced identical reciprocal sums.

---

## 3. RRF Mechanics and Forensics

The RRF ranking formula is defined as:
$$\text{RRF}(d) = \sum_{i \in \text{retrievers}} \frac{w_i}{k + \text{rank}_i(d)}$$
where $\text{rank}_i(d)$ is 1-indexed position in retriever $i$, $k$ is the smoothing constant, and $w_i$ is retriever weight.

### 3.1 Failure Pattern Breakdown (121 Queries)

Every evaluated query was mapped into one of five RRF forensic categories:

| Pattern | Category | Count | Percentage | Description |
|---|---|---|---|---|
| **BOTH_OK** | Retained Top-5 | 82 | 67.8% | Relevant chunk successfully retrieved in top-5 |
| **E_ABSTENTION** | Safe Abstention | 25 | 20.7% | Track D unanswerable queries (0 false answers) |
| **A_DENSE_WINS_RRF_LOSES** | Single-Source Suppression | 6 | 5.0% | Dense ranks relevant chunk in top 2–8, but BM25 misses. Single-source RRF score cannot beat weak dual-source hits. |
| **C_BOTH_RETRIEVE_MISORDERS** | Dual-Source Deep Ranking | 5 | 4.1% | Both retrievers retrieve relevant chunk (e.g. ranks 9 & 20), but competitors with ranks 1–5 in one retriever push it below top-5. |
| **E_ABSENT_BOTH** | Coverage Limit | 2 | 1.7% | Fundamental tokenization/representation misses (`q05`, `q29`). |
| **B_BM25_WINS_RRF_LOSES** | Single-Source Suppression | 1 | 0.8% | BM25 ranks relevant chunk at rank 13, Dense misses entirely. |

### 3.2 Key Forensic Findings
1. **The Single-Source Score Ceiling:** For $k=100$, a chunk retrieved at rank 1 by Dense alone scores $\frac{1}{101} \approx 0.0099$. A competitor chunk retrieved at rank 15 in Dense and rank 18 in BM25 scores $\frac{1}{115} + \frac{1}{118} \approx 0.0087 + 0.0085 = 0.0172$, completely dominating the single-source hit. This explains why `track_b_q10`, `q13`, `q26`, `q28`, `q46`, and `q47` fail to enter top-5 despite strong Dense retrieval.
2. **Rank Reordering without Cross-Attention:** RRF treats rank 1 in BM25 (e.g., token overlap on generic words) with the exact same weight as rank 1 in Dense (semantic match). No scalar adjustment of $k$ or static weights can dynamically differentiate lexical noise from semantic relevance.

---

## 4. Controlled Experiment Matrix

A total of 16 distinct configurations were evaluated across all 4 tracks.

### 4.1 Stage 1: $rrf\_k$ Sweep ($candidate\_k=20$, equal weights)

| $rrf\_k$ | Track B Hit@1 | Track B Hit@5 | Track B MRR@5 | Wins | Losses | Net Wins | P13 Fixed | P13 Regressed | Mean Latency |
|---|---|---|---|---|---|---|---|---|---|
| **20** | 0.3846 | **0.7500** | **0.5128** | 1 | 1 | 0 | 1 (`q41`) | 0 | 25.8 ms |
| **40** | 0.3846 | 0.7308 | 0.5099 | 0 | 1 | -1 | 0 | 0 | 25.4 ms |
| **60** | 0.3846 | 0.7308 | 0.5115 | 0 | 0 | 0 | 0 | 0 | 25.3 ms |
| **80** | 0.3846 | 0.7308 | 0.5115 | 0 | 0 | 0 | 0 | 0 | 25.5 ms |
| **100 (Base)** | 0.3846 | 0.7308 | 0.5115 | 0 | 0 | 0 | 0 | 0 | 25.6 ms |
| **150** | 0.3846 | 0.7308 | 0.5115 | 0 | 0 | 0 | 0 | 0 | 25.4 ms |
| **200** | 0.3846 | 0.7308 | 0.5115 | 0 | 0 | 0 | 0 | 0 | 25.5 ms |

**Observation:** $rrf\_k=20$ yielded a nominal increase in Hit@5 (+0.0192) by promoting `track_b_q41` from unranked to rank 5. However:
- `track_b_q50` was simultaneously demoted from rank 3 to rank 5, causing Track B Hit@3 to degrade from 0.5962 to 0.5769 (-1 query).
- Net wins across all 121 queries was **0**.
- Out of 36 Phase 13 failures, only **1 was fixed (2.8%)**, failing the meaningful progress threshold.

### 4.2 Stage 2: Candidate Depth Sweep ($rrf\_k=20$, equal weights)

| $candidate\_k$ | Track B Hit@1 | Track B Hit@5 | Track B MRR@5 | Wins | Losses | Net Wins | P13 Fixed | P13 Regressed | Latency |
|---|---|---|---|---|---|---|---|---|---|
| **10** | 0.3846 | 0.6923 | 0.5029 | 0 | 2 | -2 | 0 | 2 | 21.3 ms |
| **20** | 0.3846 | 0.7500 | 0.5128 | 1 | 1 | 0 | 1 | 0 | 25.8 ms |
| **30** | 0.3846 | 0.7500 | 0.5128 | 2 | 2 | 0 | 2 | 1 | 30.1 ms |
| **50** | 0.3846 | 0.7500 | 0.5128 | 2 | 2 | 0 | 2 | 1 | 38.4 ms |

**Observation:** Expanding candidate depth to 30 or 50 introduces regressions on Phase 13 queries (e.g. `track_b_q45` pushed down) while failing to increase net wins above 0. Truncating to 10 causes severe degradation (Hit@5 drops to 0.6923).

### 4.3 Stage 3: Weighted Dense / BM25 Fusion ($rrf\_k=20, candidate\_k=20$)

| Dense / BM25 Weight | Track B Hit@1 | Track B Hit@5 | Track B MRR@5 | Wins | Losses | Net Wins | P13 Fixed | P13 Regressed |
|---|---|---|---|---|---|---|---|---|
| **0.25 / 0.75** | 0.3846 | 0.7115 | 0.5119 | 9 | 8 | +1 | 2 | 1 |
| **0.40 / 0.60** | 0.3654 | 0.7308 | 0.5061 | 3 | 3 | 0 | 0 | 0 |
| **0.50 / 0.50** | 0.3846 | 0.7500 | 0.5128 | 1 | 1 | 0 | 1 | 0 |
| **0.60 / 0.40** | 0.4038 | 0.6731 | 0.5029 | 4 | 10 | -6 | 1 | 4 |
| **0.75 / 0.25** | 0.3846 | 0.6731 | 0.4865 | 6 | 15 | -9 | 3 | 5 |

**Observation:** Shifting weight toward BM25 or Dense creates sharp trade-offs. Weighting Dense higher (0.60 or 0.75) causes catastrophic drops in Hit@5 (-5.77pp) and 4–5 query regressions because BM25's exact token grounding is lost.

---

## 5. Query-Level Win / Loss Analysis (Best Candidate: $rrf\_k=20, ck=20$)

| Query ID | Baseline Rank | Candidate Rank | Baseline Hit@1/3/5 | Candidate Hit@1/3/5 | Classification | Forensic Explanation |
|---|---|---|---|---|---|---|
| `track_b_q41` | None | **5** | 0 / 0 / 0 | 0 / 0 / **1** | **WIN (Fixed)** | Dense rank 1 + BM25 rank 16. Lower $k$ boosted dense top rank enough to enter top-5. |
| `track_b_q50` | **3** | 5 | 0 / **1** / 1 | 0 / 0 / 1 | **LOSS (Regressed)** | Dense rank 3 + BM25 rank 4. Reordered behind competing single-source candidates. |
| All 119 others | - | - | - | - | **UNCHANGED** | Zero rank change affecting top-5 retrieval status. |

**Totals:**
- **Wins:** 1
- **Losses:** 1
- **Unchanged:** 119
- **Net Wins:** 0
- **Regression on Hit@3:** -1 query (0.5962 → 0.5769)

---

## 6. Decision Gate Analysis

Under Step 11 and Step 12 criteria, a candidate configuration is production-eligible only if:
1. **Improves Track B ranking metrics:** $rrf\_k=20$ nominally raises Hit@5 by 1 query but degrades Hit@3 by 1 query.
2. **Fixes a meaningful portion of Phase 13 failures:** 1 query fixed out of 36 failures (2.8%) is not a meaningful improvement.
3. **Zero regression on control tracks:** Track A Hit@5 (1.0000), Track C Hit@1 (1.0000), Track D (25/25) preserved.
4. **Net positive query wins:** Net wins was exactly 0 (1 win, 1 loss).
5. **Deterministic ordering:** Implemented in `core/retrieval/hybrid.py`.

Because no configuration achieved a meaningful net improvement over the baseline, the baseline configuration is retained.

---

## 7. Production Configuration & Wiring

The production retrieval pipeline remains:

```python
HybridRetriever(
    dense_retriever=DenseRetriever(embedder, vector_store),
    bm25_retriever=BM25Retriever(all_chunks),
    fusion_method="rrf",
    rrf_k=100,
    candidate_k=20,
    weights=None,  # standard symmetric RRF
)
```

Tie-handling is now hardened and deterministic across all environments:
`rrf_scores.sort(key=lambda x: (-x[1], x[0]))`

---

## 8. Recommendation for Phase 15

Phase 14 conclusively demonstrates that **RRF parameter optimization cannot overcome the RANKING_FAILURE bottleneck**. The mathematical ceiling of rank-sum fusion cannot replace cross-attention score calibration.

**Recommended Phase 15 Intervention:**
Implement **Cross-Encoder Reranking** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the candidate pool ($candidate\_k=20$ or $40$).
- Phase 7 empirical evidence demonstrated +13.5pp Hit@1 improvement (0.4038 → 0.5385).
- Reranking directly inspects full query-chunk cross-attention, solving both single-source suppression (`A_DENSE_WINS_RRF_LOSES`) and dual-source misordering (`C_BOTH_RETRIEVE_MISORDERS`).
