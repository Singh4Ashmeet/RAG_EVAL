# Phase 16 — Document-Structure-Enriched Retrieval & Representation Forensics

## Decision

**PHASE 16 COMPLETE — STRUCTURAL REPRESENTATION EXPERIMENTAL, BASELINE RETAINED**

No enriched representation configuration passed all production safety gates.
The frozen production baseline is retained unchanged.

---

## 1. Representation Audit

### Information Flow Analysis

| Stage | Structural Context Available | Structural Context Used |
|-------|------------------------------|------------------------|
| Parsers | ✅ ElementType, headings, table_data, locations | N/A |
| Schema | ✅ source_element_ids, metadata, locations | N/A |
| Chunker | ⚠️ Headers packed as raw text, no hierarchy propagation | ❌ |
| Dense Retriever | ❌ Embeds `chunk.text` directly | ❌ |
| BM25 Retriever | ❌ Tokenizes `chunk.text` directly | ❌ |
| Generator | ❌ Sends `chunk.text` to LLM | ❌ |

**Key finding:** Document hierarchy, section context, table schemas, and element types
are present in parsed elements but lost during the chunking → retrieval path.

### RFC 8259 (Track B) Header Gap

The text parser's header regex (`^#{1,6}\s+`) does not match RFC-style numbered sections
(`1.  Introduction`). Track B has **zero** parser-detected headers, meaning all 30 chunks
have no structural context in the baseline representation.

---

## 2. Representation Variants Tested

### Rep A — Baseline (Production)
Raw `chunk.text` with no enrichment.

### Rep B — Heading Enriched
```
[DOCUMENT]
{document_title}

[SECTION]
{section_hierarchy}

[CONTENT]
{chunk.text}
```

### Rep C — Full Structural
```
[DOCUMENT]
{document_title}

[SECTION]
{section_hierarchy}

[CONTENT]
{chunk.text}

[STRUCTURE]
Type: {element_types} | Page: {pages} | Table Data: [{schema}]
```

All enrichment is deterministic, derived solely from parsed document metadata.
No LLM-generated content. No evaluation query data used.

---

## 3. Experiment Matrix

| Config | Dense Rep | BM25 Rep | Description |
|--------|-----------|----------|-------------|
| Config 1 | Baseline | Baseline | Production baseline |
| Config 2 | Heading | Baseline | Enriched dense only |
| Config 3 | Baseline | Heading | Enriched BM25 only |
| Config 4 | Heading | Heading | Both enriched (heading) |
| Config 5 | Full | Baseline | Full dense only |
| Config 6 | Baseline | Full | Full BM25 only |
| Config 7 | Full | Full | Both enriched (full) |

---

## 4. Results — Track A (Internal Documentation)

| Config | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 |
|--------|-------|-------|-------|-------|----------|
| **Baseline** | **0.6429** | **0.9286** | **1.0000** | **0.7917** | **1.0000** |
| RepB Dense + A BM25 | 0.6429 | 0.9286 | 1.0000 | 0.7798 | 1.0000 |
| A Dense + RepB BM25 | 0.7143 | 1.0000 | 1.0000 | 0.8452 | 1.0000 |
| RepB Dense + RepB BM25 | 0.5714 | 0.9286 | 1.0000 | 0.7560 | 1.0000 |
| RepC Dense + A BM25 | 0.6429 | 1.0000 | 1.0000 | 0.7976 | 1.0000 |
| A Dense + RepC BM25 | 0.7143 | 1.0000 | 1.0000 | 0.8452 | 1.0000 |
| RepC Dense + RepC BM25 | 0.6429 | 1.0000 | 1.0000 | 0.7976 | 1.0000 |

Track A Hit@5 maintained at 1.0 across all configurations. ✅

---

## 5. Results — Track B (RFC 8259)

| Config | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 |
|--------|-------|-------|-------|-------|----------|
| **Baseline** | **0.3846** | **0.5962** | **0.7308** | **0.5115** | **0.7019** |
| RepB Dense + A BM25 | 0.3654 | 0.5962 | 0.6923 | 0.4942 | 0.6731 |
| A Dense + RepB BM25 | 0.3654 | 0.5962 | 0.6923 | 0.4942 | 0.6635 |
| RepB Dense + RepB BM25 | 0.3654 | 0.6346 | 0.7115 | 0.5013 | 0.6923 |
| RepC Dense + A BM25 | 0.3269 | 0.6154 | 0.6923 | 0.4766 | 0.6635 |
| A Dense + RepC BM25 | 0.3654 | 0.5962 | 0.6923 | 0.4942 | 0.6635 |
| RepC Dense + RepC BM25 | 0.3077 | 0.6346 | 0.7308 | 0.4731 | 0.7019 |

> [!WARNING]
> Track B regressed on Hit@5 and Recall@5 for 5 of 6 enriched configurations.
> Only RepC_Dense_RepC_BM25 matched baseline Hit@5/Recall@5 but failed Track C.

---

## 6. Results — Track C (Adversarial / Multi-Format)

| Config | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 |
|--------|-------|-------|-------|-------|----------|
| **Baseline** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| RepB Dense + A BM25 | 0.9333 | 1.0000 | 1.0000 | 0.9667 | 1.0000 |
| A Dense + RepB BM25 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 1.0000 |
| RepB Dense + RepB BM25 | 0.9333 | 1.0000 | 1.0000 | 0.9667 | 1.0000 |
| RepC Dense + A BM25 | 0.9333 | 1.0000 | 1.0000 | 0.9611 | 1.0000 |
| A Dense + RepC BM25 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 1.0000 |
| RepC Dense + RepC BM25 | 0.9333 | 1.0000 | 1.0000 | 0.9611 | 1.0000 |

> [!CAUTION]
> Track C Hit@1 regressed from 1.0000 to 0.9333–0.9667 in all enriched configurations.
> The structural prefix dilutes the discriminative signal for small corpus retrieval.

---

## 7. Results — Track D (Abstention Safety)

All 7 configurations: **25/25 safe abstention, 0% false answers.** ✅

---

## 8. Win/Loss Analysis

| Config | Wins | Losses | Net | P13 Fixed | P13 Regressed |
|--------|------|--------|-----|-----------|---------------|
| RepB Dense + A BM25 | 4 | 11 | **-7** | 4 | 7 |
| A Dense + RepB BM25 | 2 | 4 | **-2** | 2 | 2 |
| RepB Dense + RepB BM25 | 6 | 10 | **-4** | 6 | 6 |
| RepC Dense + A BM25 | 5 | 11 | **-6** | 5 | 5 |
| A Dense + RepC BM25 | 2 | 4 | **-2** | 2 | 2 |
| RepC Dense + RepC BM25 | 7 | 12 | **-5** | 7 | 5 |

> [!IMPORTANT]
> Every enriched configuration has **negative net wins**. While enrichment fixes some
> Phase 13 failures, it introduces more regressions than it resolves.

---

## 9. Failure Analysis

### Why Enrichment Hurts Track B

The structural prefix (`[DOCUMENT]\n...\n[SECTION]\n...`) adds 50–150 tokens of metadata
to every chunk's embedding input. For the nomic-embed-text model (768 dimensions, 8192
token context), this:

1. **Dilutes semantic density** — the embedding now captures "RFC 8259 > 2. JSON Grammar"
   alongside the actual content, reducing the cosine similarity signal for precise queries.
2. **BM25 token pollution** — structural tokens like `DOCUMENT`, `SECTION`, `CONTENT`
   appear in every chunk, reducing IDF discrimination.
3. **Shuffles ranking** — some previously correct matches drop from top-5, while some
   previously-missed matches enter top-5. Net effect is negative.

### Why Enrichment Hurts Track C

Track C has only 10 chunks across 4 documents. The structural prefix creates a stronger
document-level signal that can override the content-level signal for queries targeting
specific in-document details (e.g., table values, incident specifics).

### Where Enrichment Helps

Enrichment does fix some Phase 13 failures — queries where the lack of section context
caused the wrong section's chunk to rank higher. RepC_Dense_RepC_BM25 fixed 7 P13 failures.
However, the regressions (12 losses) outweigh the gains.

---

## 10. Chunk Integrity Verification

| Track | Chunks | IDs Match | Doc IDs Match | Source Elements Preserved | Status |
|-------|--------|-----------|---------------|--------------------------|--------|
| Track A | 25 | ✅ | ✅ | ✅ | PASS |
| Track B | 30 | ✅ | ✅ | ✅ | PASS |
| Track C | 10 | ✅ | ✅ | ✅ | PASS |
| Track D | 2 | ✅ | ✅ | ✅ | PASS |

Enrichment is an additive text transformation. Original chunk boundaries, IDs,
document assignments, and source element references are fully preserved.

---

## 11. Transform Latency

| Representation | ms/chunk |
|---------------|----------|
| Baseline | 0.0007 |
| Heading Enriched | 0.0196 |
| Full Structural | 0.0248 |

Transform overhead is negligible (< 0.03 ms/chunk). Latency is not a concern.

---

## 12. Safety Gate Summary

| Gate | Config 2 | Config 3 | Config 4 | Config 5 | Config 6 | Config 7 |
|------|----------|----------|----------|----------|----------|----------|
| G1: Track A Hit@5 ≥ 1.0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| G2a: Track B Hit@5 ≥ 0.7308 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| G2b: Track B Recall@5 ≥ 0.7019 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| G3: Track C Hit@1 ≥ 1.0 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| G4: Track D 100% abstention | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| G5: Net wins > 0 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| G6: Latency ≤ 2.0x | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ALL GATES** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**0 of 6 enriched configurations passed all gates.**

---

## 13. Leakage Audit

- Enrichment derived from: document titles, section hierarchies, element types,
  table schemas, page numbers — all from parsed document metadata.
- No evaluation query data (Tracks A–D) used to construct representations.
- No LLM-generated content in enrichment.
- No protected M4 queries in training.
- **Leakage status: CLEAN** ✅

---

## 14. Production Configuration (Unchanged)

```text
ElementAwareChunker
    chunk_size=1000
    overlap=200
        ↓
nomic-embed-text:latest
    768 dimensions
        ↓
Dense + BM25 RRF
    candidate_k=20
    rrf_k=100
    final_k=5
        ↓
llama3.2:3b
        ↓
citations / provenance / abstention
```

---

## 15. Phase 17 Recommendation

The structural enrichment approach reveals a fundamental tension: adding document context
to chunk representations can help some queries but dilutes the embedding signal for others.

Future directions to investigate:

1. **Selective enrichment** — apply structural prefix only to chunks from documents
   with known section ambiguity (e.g., RFC with many similar sections)
2. **Late-interaction models** — ColBERT-style models that can attend to structural
   tokens without diluting content embeddings
3. **Chunk-level metadata search** — use structural metadata as a filter/boost signal
   rather than embedding it into the text representation
4. **Parser improvements** — teach the text parser to recognize RFC-style headers as
   `ElementType.HEADER`, which would naturally propagate section context into chunk text
5. **Hybrid metadata scoring** — add a lightweight structural similarity score to the
   RRF fusion alongside Dense and BM25

---

## 16. Test Results

- Phase 16 unit tests: **10 passed** (`tests/unit/test_phase16_representation.py`)
- Full test suite: verified separately
- Representation components: `core/retrieval/representation/`

---

## 17. Files Modified/Created

| File | Action |
|------|--------|
| `core/retrieval/representation/__init__.py` | Created — factory and exports |
| `core/retrieval/representation/base.py` | Created — ChunkRepresentation protocol |
| `core/retrieval/representation/baseline.py` | Created — BaselineRepresentation |
| `core/retrieval/representation/structural.py` | Created — HeadingEnriched + FullStructural |
| `tests/unit/test_phase16_representation.py` | Created — 10 unit tests |
| `scripts/run_phase16_representation.py` | Created — evaluation script |
| `docs/reports/phase16_representation_evaluation.json` | Created — machine-readable results |
| `docs/PHASE_16_REPRESENTATION.md` | Created — this report |
| `docs/PHASE_16_VALIDATION.md` | Created — validation report |
