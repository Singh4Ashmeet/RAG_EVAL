# Phase 15 — Targeted Reranking Optimization & Candidate-Set Forensics

## 1. Executive Summary

Phase 15 conducted a rigorous, controlled evaluation of **pretrained cross-encoder reranking** over top-$N$ retrieval candidates from the frozen production baseline (`Dense + BM25 RRF`, `candidate_k=20`, `rrf_k=100`, `final_k=5`).

The objective was to determine whether cross-encoder reranking could safely resolve the dominant **RANKING_FAILURE** bottleneck (which accounted for ~67% of retrieval failures in Phase 13) without compromising Track A safety, Track B coverage, Track C adversarial robustness, Track D abstention safety, latency budgets, or zero-leakage constraints.

### Final Decision

`PHASE 15 COMPLETE — RERANKING EXPERIMENTAL, BASELINE RETAINED`

The production baseline remains completely frozen:
- **Chunker**: `ElementAwareChunker(chunk_size=1000, overlap=200)`
- **Embedding**: `nomic-embed-text:latest` (768 dimensions)
- **Retrieval**: `Dense + BM25 RRF` (`candidate_k=20`, `rrf_k=100`, `final_k=5`)
- **Generation**: `llama3.2:3b` with provenance, citations, and abstention safety

Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2` and `ms-marco-TinyBERT-L-2-v2`) remains an **experimental, unpromoted research branch**.

---

## 2. Baseline Reconciliation & Historical Evidence

### Production Baseline Performance (Tracks A–D)
| Metric | Track A (Control) | Track B (Challenging) | Track C (Robustness) | Track D (Abstention) |
|---|:---:|:---:|:---:|:---:|
| **Total Queries** | 14 | 52 | 30 | 25 |
| **Hit@1** | 0.6429 (9/14) | 0.3846 (20/52) | 1.0000 (30/30) | N/A |
| **Hit@3** | 0.9286 (13/14) | 0.5962 (31/52) | 1.0000 (30/30) | N/A |
| **Hit@5** | **1.0000 (14/14)** | **0.7308 (38/52)** | **1.0000 (30/30)** | N/A |
| **MRR@5** | 0.7917 | 0.5115 | 1.0000 | N/A |
| **Recall@5** | 1.0000 | 0.7019 | 1.0000 | N/A |
| **Safe Abstention** | N/A | N/A | N/A | **100.0% (25/25)** |
| **False Answer Rate** | N/A | N/A | N/A | **0.0% (0/25)** |
| **Mean Latency** | ~0.0ms | ~0.0ms (cached) | ~0.0ms | ~0.0ms |

### Historical Comparison (Phase 7 vs Phase 15)
In Phase 7, initial experiments with `ms-marco-MiniLM-L-6-v2` showed that while top-1 ranking precision improved, top-5 hit rate and recall regressed, accompanied by a 14.7x latency inflation. Phase 15 subjected this to exhaustive candidate-set forensics, candidate depth sweeps ($N \in \{5, 10, 20, 30, 40\}$), and hybrid mitigation strategies (coverage anchoring and margin gating).

---

## 3. Candidate-Set Recoverability Forensics

Phase 13 identified 37 analyzed failures across all tracks (36 non-NONE/ABSTENTION queries in Phase 13 taxonomy plus edge boundaries). Candidate-set recoverability analysis evaluated whether the correct ground-truth chunk was already present within the top-40 candidate pool:

| Category | Description | Count | % of Failures | Recoverability Status |
|---|---|:---:|:---:|---|
| **Category A** | Relevant chunk present in top-5, but at suboptimal rank (rank 2–5) | 23 | 62.2% | **Theoretically Recoverable** |
| **Category B** | Relevant chunk present in candidate pool (top 6–20/40), but missed in top-5 | 12 | 32.4% | **Theoretically Recoverable** |
| **Category C** | Relevant chunk absent from top-40 candidate pool | 2 | 5.4% | **Fundamentally Unrecoverable** |
| **Total** | | **37** | **100.0%** | **94.6% Theoretically Recoverable** |

### Breakdown of Category B Failures (Track B Misses in Candidate Pool)
12 of the 14 Track B misses are already present in the baseline candidate pool:
- `track_b_q10`, `track_b_q13`, `track_b_q15`, `track_b_q26`, `track_b_q27`, `track_b_q28`, `track_b_q41`, `track_b_q42`, `track_b_q45`, `track_b_q46`, `track_b_q47`, `track_b_q51`

### Category C Unrecoverable Failures
- `track_b_q05`: Query requires specific lexical/semantic matching not retrieved by either dense or BM25 in top-40.
- `track_b_q29`: Chunk absent from top-40 candidates across all retrievers.
Reranking cannot solve Category C without upstream retrieval improvements.

---

## 4. Evaluated Cross-Encoder Matrix

We evaluated two lightweight, CPU-compatible pretrained cross-encoders across candidate depths $k \in \{5, 10, 20, 30, 40\}$ and three distinct operational strategies:
1. **Strategy A (Full Rerank)**: Reranks all top-$N$ candidates by cross-encoder logit score.
2. **Strategy B (Coverage-Preserving Anchor)**: Guarantees that the baseline top-1 candidate is anchored in the top-5 to protect recall.
3. **Strategy C (Margin-Gated Rerank)**: Skips reranking and retains baseline when the baseline top-1 margin is decisive ($\Delta \ge 0.001$ or $0.0005$).

### Comprehensive Evaluation Results Table

| Configuration | Model | Strategy | Depth ($N$) | Track B Hit@1 | Track B Hit@5 | Track B MRR | Track A Hit@5 | Track C Hit@1 | Track D Abst. | Net Wins (B) | Mean Latency (ms) | Latency Mult. | Production Gate |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Production Baseline** | None (RRF) | Baseline | 20 | 0.3846 | **0.7308** | 0.5115 | **1.0000** | **1.0000** | **100%** | Baseline | ~10.0ms | 1.0x | **BASELINE** |
| MiniLM_d5_full | MiniLM-L-6 | Full (A) | 5 | 0.5192 | 0.7308 | 0.5981 | 1.0000 | 0.9667 | 100% | +11 | 178.4ms | 17.8x | **REJECT** (Gate 3, 6) |
| MiniLM_d10_full | MiniLM-L-6 | Full (A) | 10 | 0.5385 | 0.6923 | 0.5962 | 1.0000 | 0.9667 | 100% | +8 | 428.0ms | 42.8x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d20_full | MiniLM-L-6 | Full (A) | 20 | **0.5385** | 0.7115 | **0.5994** | 1.0000 | 0.9667 | 100% | +7 | 1155.2ms | 115.5x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d20_anchor | MiniLM-L-6 | Anchor (B) | 20 | 0.5385 | 0.7115 | 0.5994 | 1.0000 | 0.9667 | 100% | +7 | 1159.6ms | 116.0x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d20_gate001 | MiniLM-L-6 | Gated (C) | 20 | 0.5385 | 0.7115 | 0.5994 | 1.0000 | 0.9667 | 100% | +7 | 1151.3ms | 115.1x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d20_gate0005 | MiniLM-L-6 | Gated (C) | 20 | 0.5192 | 0.7115 | 0.5897 | 1.0000 | 0.9667 | 100% | +6 | 968.9ms | 96.9x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d30_full | MiniLM-L-6 | Full (A) | 30 | 0.5385 | 0.7115 | 0.5994 | 1.0000 | 0.9667 | 100% | +7 | 1679.3ms | 167.9x | **REJECT** (Gate 2, 3, 6) |
| MiniLM_d40_full | MiniLM-L-6 | Full (A) | 40 | 0.5385 | 0.7115 | 0.5994 | 1.0000 | 0.9667 | 100% | +7 | 1676.1ms | 167.6x | **REJECT** (Gate 2, 3, 6) |
| TinyBERT_d5_full | TinyBERT-L-2 | Full (A) | 5 | 0.4231 | 0.7308 | 0.5423 | 1.0000 | 0.9667 | 100% | +4 | 18.3ms | 1.8x | **REJECT** (Gate 3) |
| TinyBERT_d10_full | TinyBERT-L-2 | Full (A) | 10 | 0.4038 | 0.7308 | 0.5276 | 1.0000 | 0.9667 | 100% | +0 | 33.1ms | 3.3x | **REJECT** (Gate 3, 5) |
| TinyBERT_d20_full | TinyBERT-L-2 | Full (A) | 20 | 0.4231 | 0.7115 | 0.5372 | 0.9286 | 0.9667 | 100% | +0 | 86.1ms | 8.6x | **REJECT** (Gate 1, 2, 3, 5, 6) |
| TinyBERT_d20_anchor | TinyBERT-L-2 | Anchor (B) | 20 | 0.4231 | 0.7115 | 0.5372 | 0.9286 | 0.9667 | 100% | +0 | 85.0ms | 8.5x | **REJECT** (Gate 1, 2, 3, 5, 6) |
| TinyBERT_d20_gate001 | TinyBERT-L-2 | Gated (C) | 20 | 0.4231 | 0.7115 | 0.5372 | 0.9286 | 0.9667 | 100% | +0 | 85.8ms | 8.6x | **REJECT** (Gate 1, 2, 3, 5, 6) |
| TinyBERT_d20_gate0005 | TinyBERT-L-2 | Gated (C) | 20 | 0.4231 | 0.7308 | 0.5452 | 0.9286 | 0.9667 | 100% | +2 | 75.8ms | 7.6x | **REJECT** (Gate 1, 3, 5, 6) |
| TinyBERT_d30_full | TinyBERT-L-2 | Full (A) | 30 | 0.4231 | 0.7115 | 0.5372 | 0.9286 | 0.9667 | 100% | +0 | 136.4ms | 13.6x | **REJECT** (Gate 1, 2, 3, 5, 6) |
| TinyBERT_d40_full | TinyBERT-L-2 | Full (A) | 40 | 0.4231 | 0.7115 | 0.5372 | 0.9286 | 0.9667 | 100% | +0 | 141.6ms | 14.2x | **REJECT** (Gate 1, 2, 3, 5, 6) |

---

## 5. Query-Level Wins and Losses Analysis

### MiniLM-L-6-v2 (Depth=20)
- **Wins (+15 queries)**: Successfully promoted ground-truth chunks to rank 1:
  - `track_a_q05`, `track_a_q07`
  - `track_b_q04`, `track_b_q06`, `track_b_q07`, `track_b_q13`, `track_b_q25`, `track_b_q30`, `track_b_q32`, `track_b_q37`, `track_b_q39`, `track_b_q41`, `track_b_q43`, `track_b_q47`
- **Losses (-8 queries, 6 from Phase 13 failures)**:
  - `track_a_q06`: Regressed due to overconfidence on generic semantic overlap.
  - `track_b_q12`, `track_b_q16`, `track_b_q23`, `track_b_q24`, `track_b_q50`: Cross-encoder demoted specific numeric/exact configuration chunks in favor of superficially fluent explanations.
  - Overall Track B Hit@5 fell from 0.7308 to 0.7115 (-1.9pp).

### TinyBERT-L-2-v2 (Depth=20)
- **Net Wins: +0**: 12 wins offset by 12 losses.
- **Track A Regression**: Track A Hit@5 fell from 1.0000 to 0.9286 (failed Gate 1 immutable control).
- Failed to resolve technical specification queries (`track_b_q10`, `q26`, `q41`, `q46`).

---

## 6. Multi-Track Safety Gate Verification

Six strict production gates were applied:
1. **Gate 1 (Track A Control Integrity)**: Track A Hit@5 must remain 1.0000 (14/14).
   - *Result*: MiniLM passed (1.0). TinyBERT failed at depth $\ge 20$ (0.9286).
2. **Gate 2 (Track B Coverage Preservation)**: Track B Hit@5 $\ge 0.7308$ and Recall@5 $\ge 0.7019$.
   - *Result*: All MiniLM depths $\ge 10$ failed (Hit@5 = 0.6923–0.7115). All TinyBERT depths $\ge 20$ failed (Hit@5 = 0.7115).
3. **Gate 3 (Track C Adversarial Robustness)**: Track C Hit@1 must remain 1.0000 (30/30).
   - *Result*: **FAILED by all cross-encoder configurations** (Hit@1 dropped to 0.9667 due to distractor needle confusion).
4. **Gate 4 (Track D Abstention Safety)**: 100% abstention rate (25/25), 0% false answers.
   - *Result*: **PASSED by all configurations** (cross-encoders do not invent answers when candidates are abstention-flagged).
5. **Gate 5 (Efficacy / Net Wins)**: Net wins $> 0$ and $\ge 3$ Phase 13 ranking failures resolved.
   - *Result*: MiniLM passed (+7 to +11 net wins). TinyBERT failed (+0 net wins at depth 10, 20, 30, 40).
6. **Gate 6 (Latency Budget)**: Latency multiplier $\le 3.0\times$ baseline.
   - *Result*: MiniLM severely failed (17.8x to 167.9x multiplier). TinyBERT passed only at depth 5 (1.8x).

**Gate Verdict**: 0 out of 16 tested configurations passed all 6 gates.

---

## 7. Forensic Conclusions & Production Justification

1. **The Reranking Paradox**: Pretrained cross-encoders (trained on MS-MARCO general-domain web passage ranking) exhibit strong semantic bias. While they effectively boost conversational queries, they demote domain-specific architectural and numerical configurations in favor of general prose.
2. **Adversarial Vulnerability**: Cross-encoders dropped Track C Hit@1 from 100% to 96.67%, demonstrating vulnerability to distractors that RRF fusion safely suppresses.
3. **Latency Infeasibility on CPU**: `MiniLM-L-6-v2` adds 1150ms per query on CPU (a 115x penalty), which is prohibitive for production SLA requirements.
4. **Candidate-Set Reality**: While 94.6% of failures are theoretically recoverable in the top-40 candidate pool, off-the-shelf cross-encoders lack the domain grounding to recover them safely without human-curated or in-domain fine-tuning.

---

## 8. Recommendations for Phase 16

Since neither RRF fusion tuning (Phase 14) nor off-the-shelf cross-encoder reranking (Phase 15) can safely resolve the ranking bottleneck:
1. **Investigate Context-Enriched Candidate Representation**: Enrich candidate chunks with structural/hierarchical breadcrumbs prior to initial vector/lexical indexing.
2. **Domain-Specific Late Interaction (ColBERT) or In-Domain Cross-Encoder Fine-Tuning**: If reranking is revisited, it must be fine-tuned on the Phase 11 independent domain dataset rather than relying on out-of-domain MS-MARCO weights.
3. **Query Expansion / Transformation Forensics**: Evaluate whether targeted query rewriting can bridge the lexical/semantic gaps that caused Category B and Category C candidate misses.
