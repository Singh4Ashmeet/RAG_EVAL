# Phase 17 — Validation Report

## Validation Overview

**PHASE 17 COMPLETE — STRUCTURAL METADATA FUSION EXPERIMENTAL, BASELINE RETAINED**

This report documents the verification, safety, reproducibility, provenance, and data-integrity checks performed during Phase 17.

---

## 1. Leakage Verification

### Partition Isolation Audit

| Partition | Purpose | Document Set | Query Set | Used for Development? | Used for Final Eval? |
|---|---|---|---|:---:|:---:|
| **Phase 11 Train** | Training | 8 documents | 319 queries | ❌ (Not used) | ❌ |
| **Phase 11 Validation** | Development Sweep | 5 documents | 109 queries | ✅ Yes | ❌ |
| **Phase 11 Test** | Held-Out Dev Test | 4 documents | 95 queries | ❌ | ❌ |
| **Protected Track A** | Control Benchmark | 3 documents | 14 queries | ❌ | ✅ Yes |
| **Protected Track B** | RFC Primary Benchmark | 1 document | 52 queries | ❌ | ✅ Yes |
| **Protected Track C** | Adversarial Multi-Format | 4 documents | 30 queries | ❌ | ✅ Yes |
| **Protected Track D** | Abstention Safety | 1 document | 25 queries | ❌ | ✅ Yes |

### Leakage Audit Findings
1. **Zero Protected Query Exposure:** No query from Track A, B, C, or D was inspected, mined, or used to tune signal weights, alpha grids, or margin thresholds.
2. **Phase 11 Document Disjointness:** Phase 11 validation documents are strictly partitioned; development decisions were isolated to Phase 11 validation queries.
3. **No M4 Query Contamination:** Historical M4 evaluation queries remain external and unreferenced.
4. **Leakage Status: PASS / CLEAN.**

---

## 2. Provenance & Chunk Integrity

### Invariant Checks

| Invariant | Requirement | Status | Verification Method |
|---|---|:---:|---|
| **Chunk ID Immutability** | `chunk.chunk_id` unchanged | ✅ PASS | Verified in `test_provenance_preservation` |
| **Document ID Immutability** | `chunk.document_id` unchanged | ✅ PASS | Verified in `test_provenance_preservation` |
| **Text Verbatim Preservation** | `chunk.text` strictly unmodified | ✅ PASS | Verified in `test_provenance_preservation` |
| **Location Integrity** | `chunk.locations` preserved | ✅ PASS | Spatial coordinates unchanged |
| **Source Element Integrity** | `chunk.source_element_ids` intact | ✅ PASS | Element provenance links intact |
| **Non-Interference with Generation** | Prompts receive original text | ✅ PASS | Generator receives canonical chunk |

---

## 3. Determinism & Baseline Equivalence

### Determinism Check
All structural signals ($S_{\text{heading}}$, $S_{\text{section}}$, $S_{\text{element\_type}}$, $S_{\text{table\_schema}}$, $S_{\text{doc\_context}}$) use deterministic regex and token set containment. Identical inputs produce bit-for-bit identical floats across multiple invocations (verified in `test_deterministic_scoring`).

### Baseline Equivalence Invariant
When structural fusion is disabled (`strategy="none"` or `alpha=0.0`):
- Candidates returned are identical in ordering, chunk IDs, and RRF scores to the production baseline.
- `diagnostics["modifications"] == 0`.
- Verified in `test_baseline_equivalence_when_disabled`.

---

## 4. Test Suite Summary

### Phase 17 Unit Tests
- File: `tests/unit/test_phase17_structural_fusion.py`
- Tests: 14 passing
  - `test_tokenize_text` ✅
  - `test_heading_signal_matching` ✅
  - `test_section_signal_hierarchy` ✅
  - `test_element_type_handling` ✅
  - `test_table_schema_matching` ✅
  - `test_doc_context_matching` ✅
  - `test_deterministic_scoring` ✅
  - `test_score_normalization_bounded` ✅
  - `test_missing_document_and_empty_metadata` ✅
  - `test_baseline_equivalence_when_disabled` ✅
  - `test_additive_fusion_ordering` ✅
  - `test_rank_aware_margin_boost` ✅
  - `test_tie_break_strategy` ✅
  - `test_provenance_preservation` ✅

---

## 5. Repository Integrity & Safety

- **Production Baseline Intact:** The canonical retrieval pipeline in `main.py` and `core/retrieval/` remains strictly unchanged.
- **Experimental Code Isolated:** Structural scoring and fusion components reside in `core/retrieval/structural/` and are strictly opt-in.
- **No Tracked Temporary Artifacts:** No ephemeral SQLite databases or temporary files committed.
