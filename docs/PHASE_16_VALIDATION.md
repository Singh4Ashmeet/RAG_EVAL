# Phase 16 — Validation Report

## Decision Validation

**PHASE 16 COMPLETE — STRUCTURAL REPRESENTATION EXPERIMENTAL, BASELINE RETAINED**

This report validates the integrity, reproducibility, and safety of the Phase 16
experimental evaluation.

---

## 1. Leakage Verification

### Enrichment Data Sources

| Source | Origin | Evaluation Query Dependent? |
|--------|--------|-----------------------------|
| Document title | First heading or filename | ❌ No |
| Section hierarchy | Parsed element headers + RFC regex | ❌ No |
| Element types | Parser element classification | ❌ No |
| Table schemas | Parsed TableData headers/captions | ❌ No |
| Page numbers | Parser location metadata | ❌ No |

### Protected Data Isolation

| Dataset | Used in Enrichment Construction? | Used in Evaluation? |
|---------|----------------------------------|---------------------|
| Track A golden queries | ❌ No | ✅ Yes (evaluation only) |
| Track B golden queries | ❌ No | ✅ Yes (evaluation only) |
| Track C golden queries | ❌ No | ✅ Yes (evaluation only) |
| Track D golden queries | ❌ No | ✅ Yes (evaluation only) |
| Phase 11 routing dataset | ❌ No | ❌ No |
| M4 queries | ❌ No | ❌ No |

**Leakage status: CLEAN** ✅

---

## 2. Representation Integrity

### Chunk Boundary Preservation

For every track and every representation variant (baseline, heading_enriched, full_structural):

- ✅ Chunk count is identical to canonical count
- ✅ Chunk IDs are identical in the same order
- ✅ Document IDs are identical
- ✅ Source element IDs are preserved
- ✅ Original chunk.text is embedded within enriched text (never modified)

### Determinism Verification

Representation transforms are fully deterministic:
- No random components
- No LLM-generated content
- No external API calls
- Same input always produces identical output
- Verified via `test_representation_determinism` unit test

---

## 3. Provenance Verification

### Chunk-to-Source Traceability

| Property | Baseline | Heading Enriched | Full Structural |
|----------|----------|------------------|-----------------|
| chunk_id preserved | ✅ | ✅ | ✅ |
| document_id preserved | ✅ | ✅ | ✅ |
| source_element_ids preserved | ✅ | ✅ | ✅ |
| locations preserved | ✅ | ✅ | ✅ |
| metadata preserved | ✅ | ✅ | ✅ |
| Original text recoverable | ✅ | ✅ | ✅ |

Enrichment wraps the original text; it never modifies the canonical Chunk object.
The enriched text is used only for embedding and BM25 indexing. The generator
continues to receive the original `chunk.text`.

---

## 4. Test Results

### Phase 16 Unit Tests

```
tests/unit/test_phase16_representation.py — 10 passed

  test_extract_document_title_from_header          PASSED
  test_extract_document_title_fallback_filename     PASSED
  test_extract_section_hierarchies                  PASSED
  test_extract_table_metadata                       PASSED
  test_baseline_representation_preserves_text       PASSED
  test_heading_enriched_representation              PASSED
  test_full_structural_representation               PASSED
  test_representation_determinism                   PASSED
  test_missing_document_or_metadata                 PASSED
  test_factory_function                             PASSED
```

### Evaluation Script

- `scripts/run_phase16_representation.py` — exit code 0
- 7 configurations evaluated across 4 tracks
- All results recorded in `docs/reports/phase16_representation_evaluation.json`

---

## 5. Safety Gate Verification

### Gate Definitions

| Gate | Requirement | Rationale |
|------|-------------|-----------|
| G1 | Track A Hit@5 ≥ 1.0 | Internal docs retrieval must not regress |
| G2a | Track B Hit@5 ≥ 0.7308 | RFC retrieval coverage must not regress |
| G2b | Track B Recall@5 ≥ 0.7019 | RFC retrieval recall must not regress |
| G3 | Track C Hit@1 ≥ 1.0 | Multi-format adversarial precision must hold |
| G4 | Track D 100% abstention | Out-of-scope safety must hold |
| G5 | Net wins > 0 | Must demonstrate net improvement |
| G6 | Latency ≤ 2.0x | Must be production viable |

### Gate Results

All 6 non-baseline configurations failed at least 2 gates.
The most common failure pattern: Track B Hit@5/Recall@5 regression + Track C Hit@1 regression + negative net wins.

---

## 6. Baseline Metrics Verification

Baseline metrics from this evaluation match Phase 15 baseline:

| Track | Metric | Phase 15 | Phase 16 | Match |
|-------|--------|----------|----------|-------|
| A | Hit@1 | 0.6429 | 0.6429 | ✅ |
| A | Hit@5 | 1.0000 | 1.0000 | ✅ |
| A | MRR | 0.7917 | 0.7917 | ✅ |
| B | Hit@1 | 0.3846 | 0.3846 | ✅ |
| B | Hit@5 | 0.7308 | 0.7308 | ✅ |
| B | Recall@5 | 0.7019 | 0.7019 | ✅ |
| C | Hit@1 | 1.0000 | 1.0000 | ✅ |
| C | Hit@5 | 1.0000 | 1.0000 | ✅ |
| D | Abstention | 25/25 | 25/25 | ✅ |

Baseline is reproducible. ✅

---

## 7. Production Impact

**None.** No production code was modified. The representation module
(`core/retrieval/representation/`) is a new, isolated component not wired into
the production retrieval pipeline.

Existing production behavior is completely unchanged.
