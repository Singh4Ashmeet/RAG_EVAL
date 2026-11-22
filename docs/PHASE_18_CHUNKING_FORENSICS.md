# Phase 18 — Element-Centric & Boundary-Aware Retrieval Forensics

## Executive Summary & Final Decision

**PHASE 18 COMPLETE — CHUNK-BOUNDARY HYPOTHESIS NOT ACTIONABLE, BASELINE RETAINED**

Phase 18 directly tested the hypothesis that remaining retrieval failures in RAG_EVAL are caused by chunk-boundary alignment and retrieval-unit granularity (i.e. bundling the answer with unrelated surrounding elements inside 1000-character chunks).

The experimental results conclusively disproved this hypothesis:
1. **Severe Retrieval Collapse on Technical Prose (Track B):** Retrieving smaller, atomic semantic units (Candidate B: 115 chars avg) or boundary-aware grouped units (Candidate D: 337 chars avg) catastrophically degraded Track B retrieval quality:
   - Hit@5 dropped from **0.7308** (baseline) to **0.3654** (atomic) and **0.5962** (boundary-aware).
   - Recall@5 collapsed from **0.7019** (baseline) to **0.3173** (atomic) and **0.5769** (boundary-aware).
   - MRR@5 collapsed from **0.5115** to **0.2186** (atomic) and **0.3891** (boundary-aware).
2. **Track C Precision Degradation:** Track C Hit@1 dropped from a perfect **1.0000** (baseline) to **0.8667–0.9000** across all smaller-unit candidates.
3. **Negative Net Wins Across All Strategies:**
   - Candidate B (Element Atomic): 5 wins, 31 losses (**Net: -26**)
   - Candidate C (Heading Attached): 7 wins, 31 losses (**Net: -24**)
   - Candidate D (Boundary Aware 500): 10 wins, 25 losses (**Net: -15**)
4. **Latency & Index Inflation:** Atomic chunking increased the total index size by **5.5x** (358 units vs 65 baseline units) and increased retrieval latency by **4.22x** (161.8ms vs 38.3ms).
5. **Context Starvation Mechanism Discovered:** Small semantic units lack the lexical and semantic co-occurrence context needed for dense embeddings (`nomic-embed-text`) and BM25 term weighting to establish topic alignment with natural language queries.
6. **Production Baseline Retained:** 0 of 3 experimental chunking candidates passed the promotion gates. The production baseline remains immutable.

---

## 1. Audit of the Existing Chunking Pipeline

### Current Pipeline Flow
```text
Source Document (.md, .txt, .pdf, .docx)
       ↓
PlainTextParser / PDFParser / DOCXParser
       ↓
ExtractedElement Sequence (content, element_type, order_index, location)
       ↓
ElementAwareChunker (chunk_size=1000, chunk_overlap=200)
       ↓
Forward-Only Greedy Packing (concatenates with "\n\n" until candidate_len > 1000)
       ↓
Canonical Chunk Instances (chunk_id="{doc_id}_c{chunk_index:04d}")
       ↓
Dense Embedding (nomic-embed-text:latest) + BM25 Tokenization
       ↓
RRF Fusion (candidate_k=20, rrf_k=100, final_k=5)
```

### Ingestion Audit Findings
- **Element Definition:** Elements are paragraphs, list items, headings, table structures, and code blocks extracted by parsers.
- **Merge Behavior:** `ElementAwareChunker` packs adjacent elements greedily until 1000 characters is reached.
- **Boundary Obliviousness:** The baseline chunker does not respect section or element-type boundaries during packing. A heading, body text, a mermaid diagram, and an adjacent section can all be packed into one chunk if total characters $\le 1000$.

---

## 2. Failure-Driven Analysis (Phases 13–17 Cross-Reference)

Inspection of historical ranking failures revealed clear examples where 1000-character packing bundled target answers with unrelated text:
- **`track_a_q01` (`doc_95ccdbb830bc7585_c0000`):** Target element `e0004` (166 chars about evaluation retrieval) was packed with a 500-char mermaid diagram and `# Architecture` heading.
- **`track_b_q13` (`doc_61a5378f4255c720_c0013`):** Target element `e0094` (Array structure, Section 5) was packed with Section 6 (`6. Numbers`) and 4 number-definition elements.
- **`track_b_q10` (`doc_61a5378f4255c720_c0011`):** Target element `e0075` (Whitespace grammar, Section 2) was packed across 13 elements spanning Section 3 (`3. Values`) and Section 4 (`4. Objects`).

This evidence motivated the central hypothesis: Would isolating elements into atomic or boundary-aware chunks fix these failures?

---

## 3. Candidate Chunking Strategies

| Candidate | Strategy Name | Chunker Class | Target Granularity | Boundary Rules |
|---|---|---|---|---|
| **A** | **Baseline (Control)** | `ElementAwareChunker(1000, 200)` | ~860 chars avg | Forward greedy packing up to 1000 chars |
| **B** | **Element Atomic** | `ElementAtomicChunker(1000)` | ~150 chars avg | Exactly one semantic element per retrieval unit |
| **C** | **Heading Attached** | `ElementHeadingAttachedChunker(1200)` | ~170 chars avg | Atomic element with immediate section heading prepended |
| **D** | **Boundary Aware 500** | `BoundaryAwareChunker(500)` | ~337 chars avg | Compatible elements grouped up to 500 chars; strict heading/table isolation |

---

## 4. Stage 1: Phase 11 Development Evaluation

Evaluated against the independent Phase 11 validation split (109 queries, 5 documents):

| Strategy | Units | Avg Length | Hit@1 | Hit@5 | MRR@5 | Wins | Losses | Unchanged | Net Wins |
|---|---|---|---|---|---|---|---|---|---|
| **Candidate A (Baseline)** | **23** | **772.4** | **0.7982** | **0.9633** | **0.8700** | - | - | 109 | **0** |
| Candidate B (Element Atomic) | 105 | 167.6 | 0.7706 | 0.9358 | 0.8398 | 9 | 15 | 85 | **-6** |
| Candidate C (Heading Attached) | 105 | 189.7 | 0.7706 | 0.9358 | 0.8364 | 9 | 16 | 84 | **-7** |
| Candidate D (Boundary Aware 500) | 56 | 327.0 | 0.7523 | 0.9450 | 0.8329 | 9 | 18 | 82 | **-9** |

All three experimental strategies regressed on Phase 11 development data (Net wins -6 to -9).

---

## 5. Stages 2 & 3: Protected Tracks & Candidate-Set Forensics

### Track A (Internal Infrastructure Docs, 14 queries)

| Strategy | Units | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Candidate A (Baseline)** | **25** | 0.6429 | 0.9286 | **1.0000** | 0.7917 | **1.0000** | **22.5** |
| Candidate B (Element Atomic) | 93 | **0.8571** | 0.9286 | **1.0000** | **0.9071** | **1.0000** | 114.0 |
| Candidate C (Heading Attached) | 93 | 0.7857 | 0.9286 | 0.9286 | 0.8571 | 0.9286 | 100.5 |
| Candidate D (Boundary Aware 500) | 65 | 0.7857 | 0.8571 | **1.0000** | 0.8571 | **1.0000** | 68.6 |

*Observation:* On Track A, smaller units improved precision (Hit@1 0.6429 $\rightarrow$ 0.8571) because architecture elements are self-contained and descriptive.

### Track B (RFC 8259 Standard, 52 queries)

| Strategy | Units | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Candidate A (Baseline)** | **30** | **0.3846** | **0.5962** | **0.7308** | **0.5115** | **0.7019** | **38.3** |
| Candidate B (Element Atomic) | 221 | 0.1346 | 0.3077 | 0.3654 | 0.2186 | 0.3173 | 243.2 |
| Candidate C (Heading Attached) | 221 | 0.1346 | 0.2308 | 0.3462 | 0.2032 | 0.3077 | 227.1 |
| Candidate D (Boundary Aware 500) | 72 | 0.2692 | 0.4808 | 0.5962 | 0.3891 | 0.5769 | 76.6 |

*Observation:* Massive, catastrophic collapse on Track B across all metrics. Hit@5 dropped from 0.7308 to 0.3654 (-36.5pp).

### Track C (Adversarial Multi-Format, 30 queries)

| Strategy | Units | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Candidate A (Baseline)** | **10** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **11.2** |
| Candidate B (Element Atomic) | 35 | 0.8667 | 0.9333 | 1.0000 | 0.9094 | 0.9833 | 43.1 |
| Candidate C (Heading Attached) | 35 | 0.9000 | 0.9333 | 0.9667 | 0.9250 | 0.9500 | 45.7 |
| Candidate D (Boundary Aware 500) | 24 | 0.8667 | 0.9333 | 1.0000 | 0.9039 | 1.0000 | 22.4 |

### Track D (Abstention Safety, 25 queries)
All candidates achieved **25/25 safe abstention** with **0.0% false answer rate**.

---

## 6. Candidate-Set Recoverability Analysis

Evaluating candidate pool reach across the 121 queries in the benchmark:

| Metric / Category | Candidate A (Baseline) | Candidate B (Atomic) | Candidate C (Heading) | Candidate D (Boundary 500) |
|---|:---:|:---:|:---:|:---:|
| Total Units in Pool | 65 | 358 | 358 | 166 |
| Target in Top-5 | **88 / 121 (72.7%)** | 63 / 121 (52.1%) | 60 / 121 (49.6%) | 75 / 121 (62.0%) |
| Target in Top-20 | **99 / 121 (81.8%)** | 69 / 121 (57.0%) | 75 / 121 (62.0%) | 84 / 121 (69.4%) |
| Target in Top-40 | **104 / 121 (86.0%)** | 75 / 121 (62.0%) | 77 / 121 (63.6%) | 93 / 121 (76.9%) |
| Category A (In pool, suboptimal) | - | 12 | 17 | 18 |
| Category B (Pool outside top 5) | - | 6 | 15 | 9 |
| Category C (Absent from top 40) | - | 46 | 44 | 28 |

> [!CAUTION]
> In Candidate B, **46 out of 121 queries completely lost the target answer from the top-40 candidate pool**.
> Making chunks smaller did NOT improve candidate recoverability; it drastically shrank the candidate reach because small units fail to match query keywords.

---

## 7. Root Cause Analysis: The Context-Starvation Mechanism

Why does element-centric retrieval fail so severely despite intuitive appeal?

1. **Vocabulary Starvation for BM25:**
   - In RFC 8259, an atomic element is often a single line of ABNF grammar or a brief rule (e.g. `decimal-point = %x2E ; .`).
   - A typical query contains 10–20 words with high-level conceptual terms ("JSON grammar", "specification", "number formatting", "fractional representation").
   - The atomic element contains 3–6 words, none of which match the query's conceptual terms.
   - BM25 length normalization penalizes or ignores these tiny fragments.
2. **Dense Embedding Dispersion:**
   - Embedding models (`nomic-embed-text`) are trained on sentence-to-document and paragraph-to-document similarity.
   - A 100-character fragment has an extremely localized vector that clusters near specific syntax tokens rather than general technical concepts.
   - Surrounding paragraphs provide necessary semantic anchors that pull the embedding into the correct semantic neighborhood.
3. **The Chunking Dilemma:**
   - Large chunks (1000 chars): Risk sibling-chunk interference, but ensure that the chunk contains enough contextual vocabulary for retrieval.
   - Small chunks (100–300 chars): Isolate the answer, but starve the retriever of the lexical and semantic anchors required to locate the unit.

---

## 8. Generation & Context Validation

Because Candidates B, C, and D failed retrieval metrics, end-to-end promotion was not eligible.
However, context assembly validation demonstrated that:
- Atomic chunks (115 chars avg) fail to provide sufficient context for LLM generation. When `llama3.2:3b` was tested on isolated grammar lines, it hallucinates surrounding rules because the necessary explanatory context was trimmed away.
- Therefore, even if a parent-child context expansion architecture were built, it would only compensate for a retrieval step that is itself 50% worse than the baseline.

---

## 9. Latency and Index Cost

| Strategy | Total Units | Index Build Time (ms) | Mean Total Latency (ms) | Multiplier |
|---|:---:|:---:|:---:|:---:|
| **Candidate A (Baseline)** | **65** | **18.2** | **38.3** | **1.00x** |
| Candidate B (Element Atomic) | 358 | 92.4 | 161.8 | **4.22x** |
| Candidate C (Heading Attached) | 358 | 89.1 | 152.0 | **3.97x** |
| Candidate D (Boundary Aware 500) | 166 | 45.6 | 58.5 | **1.53x** |

Atomic chunking introduces a **4.2x latency penalty** due to searching over 5.5x more vectors and BM25 postings, violating production efficiency gates.

---

## 10. Promotion Gates Summary

| Gate | Requirement | Candidate B (Atomic) | Candidate C (Heading) | Candidate D (Boundary 500) |
|---|---|:---:|:---:|:---:|
| **G1** | Track A Hit@5 $\ge 1.0000$ | ✅ (1.0000) | ❌ (0.9286) | ✅ (1.0000) |
| **G2a** | Track B Hit@5 $\ge 0.7308$ | ❌ (0.3654) | ❌ (0.3462) | ❌ (0.5962) |
| **G2b** | Track B Recall@5 $\ge 0.7019$ | ❌ (0.3173) | ❌ (0.3077) | ❌ (0.5769) |
| **G3** | Track C Hit@1 $\ge 1.0000$ | ❌ (0.8667) | ❌ (0.9000) | ❌ (0.8667) |
| **G4** | Track D 100% Abstention / 0% FAR | ✅ (25/25) | ✅ (25/25) | ✅ (25/25) |
| **G5** | Net Wins $> 0$ | ❌ (-26) | ❌ (-24) | ❌ (-15) |
| **G6** | Latency Multiplier $\le 2.5x$ | ❌ (4.22x) | ❌ (3.97x) | ✅ (1.53x) |
| **G7** | No Excess Regressions ($Fixed \ge Regressed$) | ❌ (5 vs 12) | ❌ (7 vs 12) | ❌ (9 vs 10) |
| **RESULT** | **All Gates Passed** | **FAIL** | **FAIL** | **FAIL** |

**Conclusion:** 0 of 3 configurations passed all promotion gates.

---

## 11. Production Configuration (Unchanged)

```text
ElementAwareChunker(1000, 200)
        ↓
nomic-embed-text:latest (768 dimensions)
        ↓
Dense Retrieval + BM25 Retrieval
        ↓
Reciprocal Rank Fusion (RRF)
candidate_k = 20
rrf_k = 100
final_k = 5
        ↓
llama3.2:3b
        ↓
citations + provenance + Track D abstention safety
```
