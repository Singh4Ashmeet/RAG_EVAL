# Phase 12 — Validation & Decision Gate Report

## 1. Decision Gate Verdict

> ### **PHASE 12 COMPLETE — LEARNED ROUTER EXPERIMENTAL, BASELINE RETAINED**
>
> All required Phase 12 criteria and verification gates have been satisfied:
> 1. ✅ **Baseline Frozen & Preserved**: The production retrieval pipeline remains 100% frozen on `ElementAwareChunker(1000/200)` $\to$ `nomic-embed-text:latest` $\to$ `Dense + BM25 RRF` (`candidate_k=20`, `rrf_k=100`, `final_k=5`) $\to$ `llama3.2:3b`.
> 2. ✅ **Leakage Boundary Strictly Enforced**: Training and hyperparameter selection strictly utilized only the disjoint Phase 11 independent dataset (`train.jsonl` and `validation.jsonl`). Zero protected queries from Tracks A, B, C, D (121 queries) or M4 were used in training.
> 3. ✅ **Ambiguous / No-Clear-Winner Handling**: Excluded 99 ambiguous queries from the 319 train set and 30 ambiguous queries from the 109 validation set to prevent training on arbitrary noise.
> 4. ✅ **Learned Classifier Implementation**: Implemented `LearnedQueryRouter` (`core/retrieval/router/learned.py`) with TF-IDF + Logistic Regression and confidence-gated fallback to Hybrid.
> 5. ✅ **Validation Tuning & Calibration**: Tuned confidence threshold $\tau = 0.70$ on 79 confident validation queries, achieving 67.09% accuracy and 26.6% fallback rate.
> 6. ✅ **Evidence-Based Production Gate Evaluation**:
>    - Evaluated across Held-out Test (95 queries), Track A (14 queries), Track B (52 queries), Track C (30 queries), and Track D (25 queries).
>    - Raw learned routing caused 29 losses on Test and 8 losses on Track B.
>    - Gated routing ($\tau = 0.70$) reduced regressions but regressed Track A Hit@5 (1.0000 $\to$ 0.9286) and Track B Hit@5 (0.7308 $\to$ 0.6923).
>    - Therefore, the learned router **fails promotion criteria** and is **classified as experimental only**.
> 7. ✅ **Abstention & Safety Maintained**: 100% abstention preserved on Track D (25/25 queries).
> 8. ✅ **Full Test Suite & Quality Compliance**: All 904 tests passing, zero linter errors, clean formatting, zero type check errors, and passing production smoke test.

---

## 2. Benchmark Scorecard

| Target | Metric | Frozen Baseline | Learned Router (Raw) | Learned Router (Gated $\tau=0.7$) | Verdict vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 11 Test** (95 queries) | Hit@1 | 0.5789 | 0.5474 | **0.6105** | Gated +3.16% |
| | Hit@3 | 0.8316 | 0.7368 | 0.8000 | -3.16% |
| | Hit@5 | 0.9263 | 0.8632 | **0.9368** | Gated +1.05% |
| | MRR@5 | 0.7142 | 0.6640 | **0.7249** | Gated +1.07% |
| | Mean Latency | 37.81 ms | **20.10 ms** | 29.98 ms | -20.7% latency |
| | Wins / Losses / Same | — | 10 / 29 / 56 | 4 / 4 / 87 | Neutral (4W / 4L) |
| **Track A** (14 queries) | Hit@1 | 0.6429 | 0.7143 | 0.6429 | Parity |
| | Hit@3 | 0.9286 | 0.8571 | 0.9286 | Parity |
| | Hit@5 | **1.0000** | 0.9286 | **0.9286** | **REGRESSION (-7.14%)** |
| | MRR@5 | **0.7917** | 0.8036 | 0.7738 | -1.79% |
| | Mean Latency | 49.21 ms | **7.54 ms** | 25.79 ms | -47.6% latency |
| | Wins / Losses / Same | — | 2 / 3 / 9 | 0 / **1** / 13 | Net Loss (1 Loss) |
| **Track B** (52 queries) | Hit@1 | 0.3846 | 0.4038 | 0.3846 | Parity |
| | Hit@3 | 0.5962 | 0.5769 | 0.5577 | -3.85% |
| | Hit@5 | **0.7308** | 0.6731 | **0.6923** | **REGRESSION (-3.85%)** |
| | MRR@5 | **0.5115** | 0.5019 | 0.4933 | -1.82% |
| | Mean Latency | 53.64 ms | **10.28 ms** | 23.43 ms | -56.3% latency |
| | Wins / Losses / Same | — | 6 / 8 / 38 | 4 / **6** / 42 | Net Loss (4W / 6L) |
| **Track C** (30 queries) | Hit@1 | 0.9000 | 0.9667 | 0.9000 | Parity |
| | Hit@5 | 1.0000 | 1.0000 | 1.0000 | Parity (100%) |
| | MRR@5 | 0.9500 | 0.9833 | 0.9500 | Parity |
| | Mean Latency | 22.07 ms | **2.97 ms** | 12.36 ms | -44.0% latency |
| | Wins / Losses / Same | — | 2 / 0 / 28 | 0 / 0 / 30 | Neutral (0W / 0L) |
| **Track D** (25 queries) | Abstention Rate | 100% | 100% | 100% | **100% SAFE** |
| | Mean Latency | 26.68 ms | **4.89 ms** | 15.22 ms | -43.0% latency |

---

## 3. Decision Gate Criteria Check

| Criterion | Requirement | Result | Evaluation |
| :--- | :--- | :--- | :--- |
| **Query-level Wins vs Losses** | Net positive on test & benchmarks | Test: 4W / 4L, Track B: 4W / 6L | ❌ **FAIL** (Net negative on Track B) |
| **Protected Track Regression** | Zero regression on Tracks A, B, C | Track A Hit@5: -7.14%, Track B Hit@5: -3.85% | ❌ **FAIL** (Regressions observed) |
| **Safety & Abstention** | 100% abstention on Track D | 100% preserved (25/25) | ✅ **PASS** |
| **Latency Advantage** | Reduced latency vs hybrid | 20% to 56% reduction across tracks | ✅ **PASS** |
| **Deterministic Behavior** | Reproducible with tests | 14 new tests passing deterministically | ✅ **PASS** |
| **Option A: Promote to Production** | Must beat baseline without regressions | Regressions present on Tracks A and B | ❌ **REJECTED** |
| **Option B: Experimental Only** | Router viable, fast, but baseline retained | Preserved in codebase for experimental use | ✅ **ACCEPTED** |
| **Option C: Outright Reject** | Only if model fails completely or degrades safety | Safety preserved, latency advantageous | ❌ **NOT APPLICABLE** |

---

## 4. Full Regression Suite Validation

| Verification Check | Target / Command | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Full Pytest Suite** | `uv run pytest -q` | **904 passed, 1 skipped** | All 14 Phase 12 unit tests passing |
| **Ruff Linter** | `uv run ruff check .` | **All checks passed** | Zero linting errors |
| **Ruff Formatter** | `uv run ruff format --check .` | **233 files checked, 0 modified** | 100% compliant |
| **MyPy Static Types** | `uv run mypy` | **Success: 0 issues in 100 source files** | Fully typed |
| **Production Smoke Test** | `uv run python main.py --smoke-test` | **PASS** | Frozen baseline verified in runtime |
| **Track A Quality & Corpora** | `test_golden_quality.py::[track_a]` | **PASS** | Protected benchmark integrity verified |
| **Track B Quality & Corpora** | `test_golden_quality.py::[track_b]` | **PASS** | Protected benchmark integrity verified |
| **Track C Quality & Corpora** | `test_golden_quality.py::[track_c]` | **PASS** | Protected benchmark integrity verified |
| **Track D Quality & Corpora** | `test_golden_quality.py::[track_d]` | **PASS** | Protected benchmark integrity verified |
| **Track D Abstention Safety** | `test_abstention_metrics.py` | **PASS** | Unanswerable queries abstained |

---

## 5. Phase-by-Phase Verification Matrix (Phases 0–12)

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 0** | Production Foundation & Architecture | ✅ **PASS** |
| **Phase 1** | Document Ingestion & Element-Aware Normalization | ✅ **PASS** |
| **Phase 2** | Golden Datasets & Multi-Track Evaluation (Tracks A–D) | ✅ **PASS** |
| **Phase 3** | Chunking Ablation & Element-Aware Optimal Configuration | ✅ **PASS** |
| **Phase 4** | Dense + BM25 RRF Production Retrieval Validation | ✅ **PASS** |
| **Phase 5** | Query Optimization & Identity Preservation | ✅ **PASS** |
| **Phase 6** | SPLADE Sparse Retrieval Evaluation (Isolated/Rejected) | ✅ **PASS** |
| **Phase 7** | Cross-Encoder Reranking Evaluation (Isolated/Rejected) | ✅ **PASS** |
| **Phase 8** | HyDE Hypothetical Embeddings Evaluation (Isolated/Rejected) | ✅ **PASS** |
| **Phase 9** | Heuristic Query Routing (Baseline Preservation) | ✅ **PASS** |
| **Phase 10** | Learned Routing Audit & Data Block Documentation | ✅ **PASS** |
| **Phase 11** | Independent Routing Dataset Construction & Validation | ✅ **PASS** |
| **Phase 12** | **Learned Query Routing (Experimental, Baseline Retained)** | ✅ **PASS** |

---

## 6. Files Created and Modified in Phase 12

### Created Files
- `core/retrieval/router/learned.py`: Complete implementation of `LearnedQueryRouter` and `LearnedRouteDecision` with confidence gating.
- `models/routing/phase12/model.joblib`: Serialized TF-IDF + LogisticRegression pipeline.
- `models/routing/phase12/metadata.json`: Model configuration and training metadata.
- `scripts/train_learned_router.py`: Deterministic model training and validation evaluation script.
- `scripts/evaluate_learned_router.py`: Multi-target benchmarking script evaluating raw and gated modes against baseline.
- `tests/unit/test_phase12_learned_router.py`: Unit test suite (14 test cases) covering training, inference, confidence gating, serialization, and dataset isolation.
- `docs/reports/phase12_evaluation_report.json`: Machine-readable evaluation metrics across all 5 benchmark targets.
- `docs/PHASE_12_LEARNED_ROUTING.md`: Comprehensive technical architecture, evaluation, and latency report.
- `docs/PHASE_12_VALIDATION.md`: Validation evidence, benchmark scorecard, and decision gate verdict.

### Modified Files
- `core/retrieval/router/__init__.py`: Exported `LearnedQueryRouter` and `LearnedRouteDecision`.
