# Phase 10 Validation

## 1. Objective

Validate whether a learned query classifier can route queries to the optimal retrieval strategy (BM25, Dense, or Hybrid) and improve retrieval performance over the Phase 4/5/7/8/9 production baseline (Dense + BM25 RRF).

## 2. Data Sufficiency Audit

### 2.1 Protected Evaluation Datasets (Cannot Use for Training)

| Dataset | Queries | Status |
|---|---:|---|
| Track A (eval/golden/track_a.jsonl) | 14 | PROTECTED |
| Track B (eval/golden/track_b.jsonl) | 52 | PROTECTED |
| Track C (eval/golden/track_c.jsonl) | 30 | PROTECTED |
| Track D (eval/golden/track_d.jsonl) | 25 | PROTECTED |
| **Total Protected** | **121** | **CANNOT USE** |

### 2.2 Potential Training Data Sources Examined

| Source | Queries | Overlap with Protected Sets | Usable for Training? |
|---|---:|---|---|
| docs/reports/m4_1_query_analysis.csv | ~40 (Track A+B) | **100% overlap** with Track A/B | ❌ NO |
| docs/reports/m4_2_per_query_results.csv | ~50+ (Track A+B) | **100% overlap** with Track A/B | ❌ NO |
| private_data/ws-alpha/documents/ | 87 document dirs | Documents, not queries | ❌ NO |
| Application logs | 0 | Not available | ❌ NO |
| Historical query logs | 0 | Not available | ❌ NO |
| Synthetic query datasets | 0 | Not available | ❌ NO |
| Benchmark query datasets | 0 | Not available | ❌ NO |

### 2.3 Critical Finding: M4 Data is NOT Independent

The M4_1 and M4_2 experiment data in `docs/reports/` contain the **exact same queries** as the protected Track A and Track B golden datasets:

- M4_1 queries: track_a_q01 through track_a_q14, track_b_q01 through track_b_q02+ (matching Track A/B)
- M4_2 queries: Same track_a_qXX and track_b_qXX identifiers, same query text
- Only difference: Different experimental configurations (baseline, exp_500_100, etc.)
- **Conclusion**: These are the SAME queries as protected evaluation sets, just from different experimental runs. They provide ZERO independent training signal.

---

## 3. Data Sufficiency Conclusion

### Independent Training Data Available: **0 queries**
### Independent Validation Data: **0 queries**
### Protected Evaluation Queries: **121** (14 + 52 + 30 + 25)

### Decision Per Phase 10 Decision Tree

> **Case C — No independent data**: Do NOT train a classifier. Do NOT label Track B and train on it. Do NOT claim learned routing validation. Instead perform an analysis-only study of what additional routing data would be required and document the blocker.

---

## 4. PHASE 10 BLOCKED — INSUFFICIENT INDEPENDENT ROUTING DATA

### Blocker Summary

| Blocker | Detail |
|---|---|
| **Zero independent training queries** | No queries exist outside the 121 protected evaluation queries |
| **M4 experiments are NOT independent** | Same queries as Track A/B, different configs only |
| **No application logs** | No historical user queries available |
| **No synthetic query generator** | No query generation pipeline exists |
| **No benchmark query datasets** | No public benchmark queries for this corpus |

### Required Data to Unblock

| Requirement | Minimum | Ideal |
|---|---:|---:|
| Independent training queries | 500+ | 2000+ |
| Independent validation queries | 100+ | 500+ |
| Query diversity (types) | 5+ categories | 10+ categories |
| Routing labels needed | Strategy per query (BM25/Dense/Hybrid) | With confidence scores |

---

## 5. Analysis: What Data Would Be Required to Unblock

### 5.1 Data Collection Requirements

To train a learned query router, we would need:

1. **Independent query corpus** (500+ queries) from a different distribution than Tracks A-D
2. **Routing labels** for each query: Best strategy (BM25/Dense/Hybrid) determined by oracle retrieval evaluation
4. **Validation set** (100+ queries) for threshold tuning
5. **No overlap** with Track A/B/C/D queries or M4 experiment queries

### 5.2 Potential Data Sources (Not Currently Available)

| Source | Effort | Notes |
|---|---|---|
| Synthetic query generation from corpus | Medium | Generate queries from private_data documents using LLM |
| Historical application logs | High | Requires production deployment |
| Public benchmark datasets (NQ, MS MARCO, etc.) | Low | Domain mismatch with RFC 8259 corpus |
| Crowdsourced query generation | High | Requires human annotation |
| LLM-generated synthetic queries | Medium | Need quality control for realism |

### 5.3 Minimum Viable Path Forward

If synthetic query generation is pursued:
1. Sample 500+ passages from private_data/ws-alpha/documents/
2. Use LLM to generate realistic user queries for each passage
5. Run oracle evaluation (BM25/Dense/Hybrid) to label best strategy
6. Split into train/val/test with no overlap to Tracks A-D
7. Train classifier with confidence gating

---

## 6. Phase 10 Decision

### Final Verdict

> **PHASE 10 BLOCKED — INSUFFICIENT INDEPENDENT ROUTING DATA**

### Evidence Summary

| Criterion | Status |
|---|---|
| Independent training data available | ❌ No (0 queries) |
| Independent validation data | ❌ No (0 queries) |
| Evaluation data protected | ✅ Yes (121 queries) |
| M4 data independent | ❌ No (same queries as Track A/B) |
| Any other query sources | ❌ None found |

### Phase 10 Status

> **PHASE 10 BLOCKED — INSUFFICIENT INDEPENDENT ROUTING DATA**

---

## 6. Recommendations for Phase 11+

If the project wishes to pursue learned routing in the future:

1. **Priority 1**: Generate 500+ synthetic queries from private_data documents using LLM
2. **Priority 2**: Run oracle labeling (BM25/Dense/Hybrid) on synthetic queries
3. **Priority 3**: Train lightweight classifier (Logistic Regression + TF-IDF)
4. **Priority 4**: Validate on held-out synthetic + Phase 9 oracle comparison
5. **Priority 5**: Deploy with confidence gating and fallback to Dense+BM25

---

## 7. Phase 10 Validation Gates

| Condition | Status |
|---|---|
| Independent training data available | ❌ BLOCKED (0 queries) |
| Independent validation data | ❌ BLOCKED (0 queries) |
| Evaluation data protected | ✅ Yes (121 queries) |
| M4 data independence verified | ✅ Confirmed NOT independent |
| No evaluation leakage | ✅ Confirmed |
| Phase 10 implementation | ❌ BLOCKED |
| Phase 10 validation | ✅ Documented |
| Phase 10 documentation | ✅ Complete |

---

## 8. Final Gate

| Gate | Status |
|---|---|
| Data sufficiency audit | ✅ Complete |
| Training data availability | ❌ BLOCKED |
| Phase 10 implementation | ❌ BLOCKED |
| Phase 10 validation | ✅ Documented |
| Phase 10 documentation | ✅ Complete |
| Git diff reviewed | ✅ |

---

**PHASE 10 BLOCKED — INSUFFICIENT INDEPENDENT ROUTING DATA**

---

**Final Verdict**: No learned query classifier can be trained without contaminating the evaluation sets. The Phase 9 oracle proves routing has theoretical value (Oracle Hit@5 = 0.8462 vs Baseline 0.7308), but current query features and available data cannot reliably capture the decision boundary. Phase 11 should focus on generating independent query data (synthetic or logged) before attempting learned routing again.

---

**PHASE 10 BLOCKED — INSUFFICIENT INDEPENDENT ROUTING DATA**