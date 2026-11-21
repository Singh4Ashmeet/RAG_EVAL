# Phase 12 — Learned Query Routing Report

## Executive Summary

Phase 12 developed, validated, and evaluated a lightweight learned retrieval router designed to adaptively select among BM25, Dense, and Hybrid retrieval strategies based on query representations.

The goal of this phase was to determine whether a lightweight learned router could outperform the frozen production baseline (`Dense + BM25 RRF`, `candidate_k=20`, `rrf_k=100`, `final_k=5`, `nomic-embed-text:latest`, `llama3.2:3b`) without regressing protected evaluation benchmarks (Tracks A–D), safety, reproducibility, or latency.

### Phase 12 Decision Gate Verdict

```text
PHASE 12 COMPLETE — LEARNED ROUTER EXPERIMENTAL, BASELINE RETAINED
```

- **Production Baseline Status:** Completely frozen and retained as default (`HybridRetriever` with RRF).
- **Learned Router Status:** Retained in codebase under `core/retrieval/router/learned.py` as an optional/experimental component for latency-sensitive non-production workflows.
- **Production Gate Reason:** The raw learned router produced net negative retrieval changes (29 losses vs 10 wins on held-out test; 8 losses vs 6 wins on Track B). Even with confidence gating ($\tau = 0.70$), Hit@5 regressed on Track A (1.0000 $\to$ 0.9286) and Track B (0.7308 $\to$ 0.6923). Consequently, learned routing cannot replace the production hybrid baseline.

---

## 1. Frozen Production Baseline

The production retrieval pipeline established in Phases 0–10 remains intact:

| Component | Configuration |
|---|---|
| Chunker | `ElementAwareChunker` (chunk_size=1000, overlap=200) |
| Embedding Model | `nomic-embed-text:latest` (768 dimensions) |
| Retrieval Fusion | Reciprocal Rank Fusion (`BM25 + Dense RRF`) |
| Fusion Hyperparameters | `candidate_k=20`, `rrf_k=100`, `final_k=5` |
| Generator Model | `llama3.2:3b` |
| Verification / Safety | Citations, provenance tracking, abstention logic |

The learned router was evaluated against this exact configuration across all benchmarks.

---

## 2. Dataset Audit & Leakage Boundary Enforcement

Training and validation utilized the leakage-safe independent routing dataset generated in Phase 11 (`data/routing/phase11/`):

- **Total queries generated:** 523 across 17 independent documents.
- **Data partitions:** 319 train (10 documents), 109 validation (3 documents), 95 test (4 documents).
- **Leakage check:** Zero overlap with protected Tracks A, B, C, D (121 queries) or M4 analysis queries.
- **Oracle labels:** Determined via oracle multi-strategy evaluation against BM25, Dense, and Hybrid.

### Exclusion of Ambiguous / No-Clear-Winner Queries

To prevent injecting arbitrary noise into training:
- **Train partition:** Out of 319 queries, 99 ambiguous/no-clear-winner queries were excluded. **220 confident queries** were used for training (`bm25`: 140, `dense`: 53, `hybrid`: 27).
- **Validation partition:** Out of 109 queries, 30 ambiguous/no-clear-winner queries were excluded. **79 confident queries** were used for hyperparameter validation (`bm25`: 49, `dense`: 17, `hybrid`: 13).
- **Held-out test partition:** Evaluated on all **95 held-out test queries** regardless of ambiguity.

---

## 3. Router Architecture & Training

The learned router (`core/retrieval/router/learned.py`) is implemented as a scikit-learn pipeline optimized for sub-millisecond inference:

1. **Feature Extraction:**
   - `TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000, sublinear_tf=True)`
2. **Classifier:**
   - `LogisticRegression(C=5.0, max_iter=300, solver='lbfgs', random_state=42)`
3. **Confidence Gating & Fallback:**
   - Prediction probabilities $P(s \mid q)$ are computed via softmax.
   - If $\max_s P(s \mid q) < \tau$ (default $\tau = 0.70$), the router falls back to `RoutingStrategy.HYBRID`.
   - If confidence meets or exceeds threshold $\tau$, the model routes directly to the predicted strategy (`BM25`, `DENSE`, or `HYBRID`).

### Validation Performance (Tuned on 79 Confident Validation Queries)

- **Overall Accuracy:** 67.09% (53 / 79 correct)
- **Macro F1:** 0.4342
- **Per-Class Breakdown:**
  - `bm25`: Precision = 0.7231, Recall = 0.9592, **F1 = 0.8246** (Support = 49)
  - `dense`: Precision = 0.4167, Recall = 0.2941, **F1 = 0.3448** (Support = 17)
  - `hybrid`: Precision = 0.5000, Recall = 0.0769, **F1 = 0.1333** (Support = 13)
- **Confidence Distribution:**
  - Mean: 0.7589
  - Median: 0.8080
  - Min: 0.4199, Max: 0.9280

### Confidence Threshold Analysis on Validation Set

| Threshold $\tau$ | Fallback Rate (%) | Predicted BM25 | Predicted Dense | Fallback to Hybrid |
|---|---|---|---|---|
| 0.50 | 3.8% | 63 | 13 | 3 |
| 0.60 | 17.7% | 55 | 10 | 14 |
| **0.70** | **26.6%** | **48** | **10** | **21** |
| 0.80 | 49.4% | 34 | 6 | 39 |

A confidence threshold of **$\tau = 0.70$** was chosen as the operational operating point for evaluation, balancing routing risk with baseline safety.

---

## 4. Comprehensive Evaluation Across Benchmarks

The router was evaluated across 5 targets: Held-out Test (95 queries), Track A (14 queries), Track B (52 queries), Track C (30 queries), and Track D (25 queries).

### Summary Comparison Table

| Target | Model Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Mean Latency | Wins / Losses / Same | Fallback Rate |
|---|---|---|---|---|---|---|---|---|
| **Phase 11 Test** (95 queries) | Baseline (Hybrid) | 0.5789 | 0.8316 | 0.9263 | 0.7142 | 37.81 ms | — | — |
| | Learned (Raw) | 0.5474 | 0.7368 | 0.8632 | 0.6640 | 20.10 ms | 10 / 29 / 56 | 0.0% |
| | **Learned (Gated $\tau=0.7$)** | **0.6105** | **0.8000** | **0.9368** | **0.7249** | **29.98 ms** | 4 / 4 / 87 | 73.7% |
| **Track A** (14 queries) | Baseline (Hybrid) | 0.6429 | 0.9286 | **1.0000** | 0.7917 | 49.21 ms | — | — |
| | Learned (Raw) | 0.7143 | 0.8571 | 0.9286 | 0.8036 | 7.54 ms | 2 / 3 / 9 | 0.0% |
| | **Learned (Gated $\tau=0.7$)** | 0.6429 | 0.9286 | **0.9286** | 0.7738 | **25.79 ms** | 0 / **1** / 13 | 42.9% |
| **Track B** (52 queries) | Baseline (Hybrid) | 0.3846 | 0.5962 | **0.7308** | 0.5115 | 53.64 ms | — | — |
| | Learned (Raw) | 0.4038 | 0.5769 | 0.6731 | 0.5019 | 10.28 ms | 6 / 8 / 38 | 0.0% |
| | **Learned (Gated $\tau=0.7$)** | 0.3846 | 0.5577 | **0.6923** | 0.4933 | **23.43 ms** | 4 / **6** / 42 | 38.5% |
| **Track C** (30 queries) | Baseline (Hybrid) | 0.9000 | 0.9667 | 1.0000 | 0.9500 | 22.07 ms | — | — |
| | Learned (Raw) | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 2.97 ms | 2 / 0 / 28 | 0.0% |
| | **Learned (Gated $\tau=0.7$)** | 0.9000 | 0.9667 | 1.0000 | 0.9500 | **12.36 ms** | 0 / 0 / 30 | 46.7% |
| **Track D** (25 queries) | Baseline (Hybrid) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.68 ms | 100% abstention | — |
| | Learned (Raw) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.89 ms | 100% abstention | 0.0% |
| | **Learned (Gated $\tau=0.7$)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **15.22 ms** | 100% abstention | 40.0% |

---

## 5. Detailed Analysis & Key Findings

### 1. Robustness of the Frozen Baseline
The fixed `Dense + BM25 RRF` baseline demonstrates remarkable resilience. Because RRF merges rankings from both dense semantic and BM25 lexical retrievers, it acts as an implicit ensemble that compensates for query-type ambiguity without requiring upfront classification.

### 2. Failure Mode of Ungated Routing
Without confidence gating, routing queries directly to a single retriever (predominantly BM25) caused severe regressions:
- On the held-out test set, raw routing incurred **29 losses** against only 10 wins (Hit@5 dropped by 6.31 points).
- On Track B (RFC 8259 domain), Hit@5 dropped from 0.7308 to 0.6731.
- Many queries containing technical terms appeared to be BM25 candidates, but actually required semantic dense matching to disambiguate synonyms and conceptual relationships.

### 3. Gating Trade-Offs
Confidence gating with fallback to Hybrid significantly reduced the regression rate:
- On held-out test, gated routing yielded 4 wins, 4 losses, and 87 unchanged queries, while increasing Hit@1 from 0.5789 to 0.6105.
- However, on protected benchmarks, gating was insufficient to prevent regressions:
  - **Track A:** 1 query regressed, dropping Hit@5 from 1.0000 to 0.9286.
  - **Track B:** 6 queries regressed vs 4 improvements, dropping Hit@5 from 0.7308 to 0.6923.
- To eliminate all regressions, the confidence threshold would have to be set so high that the fallback rate approaches 100%, rendering the router redundant.

### 4. Latency Efficiency
The primary advantage demonstrated by the learned router is execution latency:
- BM25 routing bypasses Ollama embedding generation entirely, reducing latency from ~50 ms down to ~3–10 ms.
- Across Track B, average latency dropped from **53.64 ms** to **23.43 ms** with gated routing, representing a **56% latency reduction**.
- However, per the project charter, retrieval effectiveness and protected benchmark integrity take precedence over latency.

### 5. Abstention & Safety Preservation
On Track D (25 unanswerable adversarial queries), all queries preserved 100% abstention safety across both baseline and router variants.

---

## 6. Artifact Inventory

- `core/retrieval/router/learned.py`: Complete implementation of `LearnedQueryRouter` and `LearnedRouteDecision`.
- `models/routing/phase12/model.joblib`: Serialized scikit-learn pipeline (TF-IDF + LogisticRegression).
- `models/routing/phase12/metadata.json`: Model configuration, parameters, and metadata.
- `scripts/train_learned_router.py`: Deterministic training script with class weighting, feature tuning, and validation report.
- `scripts/evaluate_learned_router.py`: Multi-target benchmarking script evaluating raw and gated modes against baseline.
- `docs/reports/phase12_evaluation_report.json`: Machine-readable benchmark report with full metrics.
- `tests/unit/test_phase12_learned_router.py`: Comprehensive test suite (14 unit tests, all passing).

---

## 7. Conclusion & Next Phase Readiness

The experimental learned query router is fully functional, reproducible, and tested. However, because it causes measurable regressions on protected benchmarks Track A and Track B, it fails the promotion gate.

**Final Status:**
- `PHASE 12 COMPLETE — LEARNED ROUTER EXPERIMENTAL, BASELINE RETAINED`
- Production baseline remains `Dense + BM25 RRF`.
- Repository is ready for subsequent phases without regression.
