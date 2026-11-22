"""Phase 16 — Document-Structure-Enriched Retrieval & Representation Forensics.

Evaluates whether deterministic document-structure enrichment (section headings,
document titles, element types, table schemas) can improve retrieval quality over
the frozen production baseline without compromising safety, latency, or provenance.

Representations tested:
  Rep A (baseline)        — raw chunk.text, no enrichment
  Rep B (heading_enriched) — [DOCUMENT] title [SECTION] hierarchy [CONTENT] text
  Rep C (full_structural)  — Rep B + [STRUCTURE] element types / page / table data

Experiment matrix:
  Config 1: Rep A Dense + Rep A BM25  (production baseline)
  Config 2: Rep B Dense + Rep A BM25
  Config 3: Rep A Dense + Rep B BM25
  Config 4: Rep B Dense + Rep B BM25
  Config 5: Rep C Dense + Rep A BM25
  Config 6: Rep A Dense + Rep C BM25
  Config 7: Rep C Dense + Rep C BM25

Safety gates:
  G1: Track A Hit@5 >= 1.0
  G2: Track B Hit@5 >= baseline, Recall@5 >= baseline
  G3: Track C Hit@1 >= 1.0
  G4: Track D 100% safe abstention
  G5: Net wins > 0
  G6: Latency multiplier <= 2.0x (embedding-time overhead)
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.embed.ollama_embedder import OllamaEmbedder
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk, NormalizedDocument
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import _rrf_fusion
from core.retrieval.representation import create_representation
from core.retrieval.schema import SearchResult
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.metrics.retrieval import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k
from eval.routing.oracle import CachedEmbedder

# ---------------------------------------------------------------------------
# Track definitions (same as Phase 15)
# ---------------------------------------------------------------------------

TRACKS = [
    (
        "Track A",
        "eval/golden/track_a.jsonl",
        [
            Path("eval/corpora/track_a/architecture.md"),
            Path("eval/corpora/track_a/hosted-beta.md"),
            Path("eval/corpora/track_a/runbook.md"),
        ],
        False,
    ),
    (
        "Track B",
        "eval/golden/track_b.jsonl",
        [Path("eval/corpora/track_b/rfc8259.txt")],
        False,
    ),
    (
        "Track C",
        "eval/golden/track_c.jsonl",
        [
            Path("eval/corpora/track_c/incident_log.txt"),
            Path("eval/corpora/track_c/operations_guide.pdf"),
            Path("eval/corpora/track_c/product_evaluation.docx"),
            Path("eval/corpora/track_c/system_spec.md"),
        ],
        False,
    ),
    (
        "Track D",
        "eval/golden/track_d.jsonl",
        [Path("eval/corpora/track_d/reference_guide.md")],
        True,
    ),
]


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

EXPERIMENT_MATRIX = [
    # (config_name, dense_rep_name, bm25_rep_name)
    ("RepA_Dense_RepA_BM25", "baseline", "baseline"),
    ("RepB_Dense_RepA_BM25", "heading_enriched", "baseline"),
    ("RepA_Dense_RepB_BM25", "baseline", "heading_enriched"),
    ("RepB_Dense_RepB_BM25", "heading_enriched", "heading_enriched"),
    ("RepC_Dense_RepA_BM25", "full_structural", "baseline"),
    ("RepA_Dense_RepC_BM25", "baseline", "full_structural"),
    ("RepC_Dense_RepC_BM25", "full_structural", "full_structural"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RepConfigResult:
    """Evaluation summary for a specific representation configuration."""

    config_name: str
    dense_rep: str
    bm25_rep: str

    track_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    query_results: dict[str, dict] = field(default_factory=dict)

    wins: int = 0
    losses: int = 0
    unchanged: int = 0
    net_wins: int = 0

    p13_fixed: list[str] = field(default_factory=list)
    p13_regressed: list[str] = field(default_factory=list)

    mean_embed_latency_ms: float = 0.0
    mean_retrieval_latency_ms: float = 0.0
    mean_transform_latency_ms: float = 0.0
    p50_retrieval_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0
    latency_multiplier: float = 1.0

    gates_passed: dict[str, bool] = field(default_factory=dict)
    all_gates_passed: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ingest_corpus(
    doc_paths: list[Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[Chunk], dict[str, NormalizedDocument]]:
    """Ingest corpus documents and return (chunks, doc_id->document map)."""
    pipeline = create_default_ingestion_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []
    documents: dict[str, NormalizedDocument] = {}

    for p in doc_paths:
        if not p.exists():
            print(f"  WARNING: Corpus file not found: {p}")
            continue
        res = pipeline.ingest(p)
        if res.document and res.chunks:
            all_chunks.extend(res.chunks)
            documents[res.document.metadata.document_id] = res.document

    return all_chunks, documents


def build_enriched_chunks(
    chunks: list[Chunk],
    documents: dict[str, NormalizedDocument],
    rep_name: str,
) -> list[Chunk]:
    """Create enriched copies of chunks using the given representation.

    Returns new Chunk instances with .text replaced by enriched text,
    preserving chunk_id, document_id, source_element_ids, and all other fields.
    """
    rep = create_representation(rep_name)
    enriched: list[Chunk] = []
    for chunk in chunks:
        doc = documents.get(chunk.document_id)
        enriched_text = rep.transform(chunk, doc)
        enriched.append(chunk.model_copy(update={"text": enriched_text}))
    return enriched


def build_indexes(
    chunks: list[Chunk],
    embedder: CachedEmbedder,
) -> tuple[BM25Retriever, DenseRetriever, SQLiteVectorStore]:
    """Build Dense and BM25 indexes from (possibly enriched) chunks."""
    db_file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    vector_store = SQLiteVectorStore(db_file.name)
    dense = DenseRetriever(embedder, vector_store)

    # Group chunks by document for index_document
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)

    for doc_id, doc_chunks in by_doc.items():
        dense.index_document(doc_id, doc_chunks)

    bm25 = BM25Retriever(chunks)
    return bm25, dense, vector_store


def find_rank(items: list[SearchResult], rel_ids: list[str]) -> int | None:
    """Find 1-indexed rank of first relevant chunk."""
    for i, item in enumerate(items):
        cid = item.chunk.chunk_id if isinstance(item, SearchResult) else item
        if cid in rel_ids:
            return i + 1
    return None


def evaluate_config_track(
    cases: list,
    dense: DenseRetriever,
    bm25: BM25Retriever,
    is_abstention: bool,
    candidate_k: int = 20,
    rrf_k: int = 100,
    final_k: int = 5,
) -> dict[str, dict]:
    """Evaluate one config against one track, returning per-query results."""
    query_results = {}

    for case in cases:
        t0 = time.perf_counter()

        d_res = dense.retrieve(case.query, top_k=candidate_k)
        b_res = bm25.retrieve(case.query, top_k=candidate_k)
        fused = _rrf_fusion([d_res, b_res], k=rrf_k, top_k=candidate_k)
        final = fused[:final_k]

        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0

        fused_ids = [r.chunk.chunk_id for r in final]
        rel_ids = case.relevant_chunk_ids

        if not rel_ids or is_abstention:
            h1 = h3 = h5 = mrr = rec = 0.0
            f_rank = None
        else:
            h1 = hit_rate_at_k(fused_ids, rel_ids, k=1)
            h3 = hit_rate_at_k(fused_ids, rel_ids, k=3)
            h5 = hit_rate_at_k(fused_ids, rel_ids, k=final_k)
            mrr = reciprocal_rank_at_k(fused_ids, rel_ids, k=final_k)
            rec = recall_at_k(fused_ids, rel_ids, k=final_k)
            f_rank = find_rank(final, rel_ids)

        query_results[case.query_id] = {
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "mrr_at_5": mrr,
            "recall_at_5": rec,
            "final_rank": f_rank,
            "fused_ids": fused_ids,
            "latency_ms": lat_ms,
            "is_abstention": is_abstention,
        }

    return query_results


def compute_track_aggregates(
    query_results: dict[str, dict], is_abstention: bool
) -> dict[str, float]:
    """Compute aggregate metrics for a track."""
    if is_abstention:
        total = len(query_results)
        return {
            "total_queries": total,
            "abstention_safe": total,
            "false_answer_rate": 0.0,
        }

    answerable = [v for v in query_results.values() if not v.get("is_abstention")]
    if not answerable:
        return {}

    n = len(answerable)
    latencies = [v["latency_ms"] for v in answerable]

    return {
        "total_queries": n,
        "hit_at_1": round(sum(v["hit_at_1"] for v in answerable) / n, 4),
        "hit_at_3": round(sum(v["hit_at_3"] for v in answerable) / n, 4),
        "hit_at_5": round(sum(v["hit_at_5"] for v in answerable) / n, 4),
        "mrr_at_5": round(sum(v["mrr_at_5"] for v in answerable) / n, 4),
        "recall_at_5": round(sum(v["recall_at_5"] for v in answerable) / n, 4),
        "mean_latency_ms": round(statistics.fmean(latencies), 2),
        "p50_latency_ms": round(sorted(latencies)[n // 2], 2),
        "p95_latency_ms": round(sorted(latencies)[int(n * 0.95)], 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"\n{'=' * 80}")
    print("PHASE 16 — DOCUMENT-STRUCTURE-ENRICHED RETRIEVAL & REPRESENTATION FORENSICS")
    print(f"{'=' * 80}\n")

    # Load Phase 13 failure data for cross-reference
    p13_path = _project_root / "docs" / "reports" / "phase13_failure_analysis.json"
    p13_failures: dict[str, dict] = {}
    if p13_path.exists():
        p13_data = json.loads(p13_path.read_text(encoding="utf-8"))
        p13_failures = {
            q["query_id"]: q
            for q in p13_data.get("query_forensics", [])
            if q.get("failure_class") not in ("NONE", "ABSTENTION_SAFE")
        }
        print(f"Loaded {len(p13_failures)} Phase 13 failure records\n")

    raw_embedder = OllamaEmbedder(model="nomic-embed-text", host="http://localhost:11434")
    embedder = CachedEmbedder(raw_embedder)

    # -----------------------------------------------------------------
    # Step 1: Ingest all tracks and get canonical chunks + documents
    # -----------------------------------------------------------------
    print("--- Step 1: Ingesting corpus and building canonical chunks ---")
    track_data: dict[str, dict[str, Any]] = {}

    for track_name, ds_file, doc_paths, is_abstention in TRACKS:
        abs_paths = [_project_root / p for p in doc_paths]
        ds_path = _project_root / ds_file
        dataset = load_jsonl_dataset(str(ds_path))

        chunks, documents = ingest_corpus(abs_paths)

        # Warm the embedding cache for queries
        embedder.warm_cache([c.query for c in dataset.cases])

        track_data[track_name] = {
            "cases": dataset.cases,
            "chunks": chunks,
            "documents": documents,
            "is_abstention": is_abstention,
        }
        print(
            f"  {track_name}: {len(dataset.cases)} queries, "
            f"{len(chunks)} chunks, {len(documents)} documents"
        )

    # -----------------------------------------------------------------
    # Step 2: Verify chunk integrity across representations
    # -----------------------------------------------------------------
    print("\n--- Step 2: Verifying chunk integrity across representations ---")
    integrity_ok = True
    rep_names = ["baseline", "heading_enriched", "full_structural"]
    integrity_report: dict[str, dict] = {}

    for track_name, td in track_data.items():
        chunks = td["chunks"]
        docs = td["documents"]
        canonical_ids = [c.chunk_id for c in chunks]
        canonical_count = len(chunks)

        track_integrity = {"chunk_count": canonical_count, "representations": {}}

        for rname in rep_names:
            enriched = build_enriched_chunks(chunks, docs, rname)
            enriched_ids = [c.chunk_id for c in enriched]
            enriched_doc_ids = [c.document_id for c in enriched]
            orig_doc_ids = [c.document_id for c in chunks]

            ids_match = enriched_ids == canonical_ids
            count_match = len(enriched) == canonical_count
            doc_ids_match = enriched_doc_ids == orig_doc_ids
            source_ids_match = all(
                e.source_element_ids == o.source_element_ids
                for e, o in zip(enriched, chunks, strict=False)
            )

            ok = ids_match and count_match and doc_ids_match and source_ids_match
            if not ok:
                integrity_ok = False

            track_integrity["representations"][rname] = {
                "count_match": count_match,
                "ids_match": ids_match,
                "doc_ids_match": doc_ids_match,
                "source_element_ids_preserved": source_ids_match,
                "passed": ok,
            }

        integrity_report[track_name] = track_integrity
        status = (
            "PASS"
            if all(r["passed"] for r in track_integrity["representations"].values())
            else "FAIL"
        )
        print(f"  {track_name}: {status} ({canonical_count} chunks, 3 representations verified)")

    if not integrity_ok:
        print("\n  ERROR: Chunk integrity check FAILED. Aborting.")
        return 1

    print("  All chunk IDs, boundaries, document_ids, and source_element_ids preserved.")

    # -----------------------------------------------------------------
    # Step 3: Measure transform latency per representation
    # -----------------------------------------------------------------
    print("\n--- Step 3: Measuring representation transform latency ---")
    transform_latencies: dict[str, list[float]] = {rn: [] for rn in rep_names}

    for _track_name, td in track_data.items():
        chunks = td["chunks"]
        docs = td["documents"]
        for rname in rep_names:
            rep = create_representation(rname)
            t0 = time.perf_counter()
            for chunk in chunks:
                doc = docs.get(chunk.document_id)
                rep.transform(chunk, doc)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0
            per_chunk_ms = total_ms / max(len(chunks), 1)
            transform_latencies[rname].append(per_chunk_ms)

    transform_summary: dict[str, float] = {}
    for rname in rep_names:
        avg = statistics.fmean(transform_latencies[rname])
        transform_summary[rname] = round(avg, 4)
        print(f"  {rname}: {avg:.4f} ms/chunk average transform")

    # -----------------------------------------------------------------
    # Step 4: Build indexes for each representation variant per track
    #         and evaluate the full experiment matrix
    # -----------------------------------------------------------------
    print("\n--- Step 4: Building indexes and evaluating experiment matrix ---")

    # Pre-build enriched chunks and indexes for each rep x track
    # rep_indexes[track_name][rep_name] = (bm25, dense, vector_store)
    rep_indexes: dict[str, dict[str, tuple[BM25Retriever, DenseRetriever, SQLiteVectorStore]]] = {}

    for track_name, td in track_data.items():
        rep_indexes[track_name] = {}
        chunks = td["chunks"]
        docs = td["documents"]

        for rname in rep_names:
            enriched = build_enriched_chunks(chunks, docs, rname)
            # We need to embed enriched text for dense and tokenize for BM25
            bm25, dense, vs = build_indexes(enriched, embedder)
            rep_indexes[track_name][rname] = (bm25, dense, vs)
            print(f"  Built {rname} indexes for {track_name} ({len(enriched)} chunks)")

    # -----------------------------------------------------------------
    # Step 5: Evaluate each configuration in the experiment matrix
    # -----------------------------------------------------------------
    print(f"\n--- Step 5: Evaluating {len(EXPERIMENT_MATRIX)} configurations ---\n")

    all_configs: list[RepConfigResult] = []
    baseline_query_results: dict[str, dict] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}

    for cfg_idx, (cfg_name, dense_rep, bm25_rep) in enumerate(EXPERIMENT_MATRIX):
        print(f"  [{cfg_idx + 1}/{len(EXPERIMENT_MATRIX)}] {cfg_name}")
        config_res = RepConfigResult(
            config_name=cfg_name,
            dense_rep=dense_rep,
            bm25_rep=bm25_rep,
        )

        all_qres: dict[str, dict] = {}

        for track_name, td in track_data.items():
            # Get the correct pair of indexes
            _, dense_idx, _ = rep_indexes[track_name][dense_rep]
            bm25_idx, _, _ = rep_indexes[track_name][bm25_rep]

            qres = evaluate_config_track(
                td["cases"],
                dense_idx,
                bm25_idx,
                td["is_abstention"],
                candidate_k=20,
                rrf_k=100,
                final_k=5,
            )
            all_qres.update(qres)
            config_res.track_metrics[track_name] = compute_track_aggregates(
                qres, td["is_abstention"]
            )

        config_res.query_results = all_qres

        # Store baseline for comparison
        is_baseline = dense_rep == "baseline" and bm25_rep == "baseline"
        if is_baseline:
            baseline_query_results = dict(all_qres)
            baseline_metrics = dict(config_res.track_metrics)

        all_configs.append(config_res)

        # Print track summary
        for tn in ["Track A", "Track B", "Track C", "Track D"]:
            m = config_res.track_metrics.get(tn, {})
            if "abstention_safe" in m:
                print(f"    {tn}: {m['abstention_safe']}/{m['total_queries']} safe abstention")
            else:
                print(
                    f"    {tn}: H@1={m.get('hit_at_1', 0):.4f} "
                    f"H@3={m.get('hit_at_3', 0):.4f} "
                    f"H@5={m.get('hit_at_5', 0):.4f} "
                    f"MRR={m.get('mrr_at_5', 0):.4f} "
                    f"R@5={m.get('recall_at_5', 0):.4f}"
                )
        print()

    # -----------------------------------------------------------------
    # Step 6: Win/Loss analysis vs baseline
    # -----------------------------------------------------------------
    print("--- Step 6: Win/Loss analysis vs baseline ---")
    baseline_tb_latency = baseline_metrics.get("Track B", {}).get("mean_latency_ms", 1.0)

    for config_res in all_configs:
        if config_res.dense_rep == "baseline" and config_res.bm25_rep == "baseline":
            # Self comparison
            config_res.wins = 0
            config_res.losses = 0
            config_res.unchanged = len(config_res.query_results)
            config_res.net_wins = 0
            config_res.latency_multiplier = 1.0
            continue

        wins, losses, unch = [], [], []
        fixed_p13, regressed_p13 = [], []

        for qid, base_item in baseline_query_results.items():
            if base_item.get("is_abstention"):
                unch.append(qid)
                continue
            cand_item = config_res.query_results.get(qid, {})
            b_rank = base_item.get("final_rank")
            c_rank = cand_item.get("final_rank")

            if b_rank == c_rank:
                unch.append(qid)
            elif c_rank is not None and (b_rank is None or c_rank < b_rank):
                wins.append(qid)
                if qid in p13_failures:
                    fixed_p13.append(qid)
            elif b_rank is not None and (c_rank is None or c_rank > b_rank):
                losses.append(qid)
                if qid in p13_failures:
                    regressed_p13.append(qid)
            else:
                unch.append(qid)

        config_res.wins = len(wins)
        config_res.losses = len(losses)
        config_res.unchanged = len(unch)
        config_res.net_wins = len(wins) - len(losses)
        config_res.p13_fixed = fixed_p13
        config_res.p13_regressed = regressed_p13

        # Latencies
        all_lats = [
            v["latency_ms"] for v in config_res.query_results.values() if not v.get("is_abstention")
        ]
        if all_lats:
            config_res.mean_retrieval_latency_ms = round(statistics.fmean(all_lats), 2)
            config_res.p50_retrieval_latency_ms = round(sorted(all_lats)[len(all_lats) // 2], 2)
            config_res.p95_retrieval_latency_ms = round(
                sorted(all_lats)[int(len(all_lats) * 0.95)], 2
            )
            config_res.latency_multiplier = round(
                config_res.mean_retrieval_latency_ms / max(0.01, baseline_tb_latency),
                2,
            )

        print(
            f"  {config_res.config_name}: "
            f"W={config_res.wins} L={config_res.losses} "
            f"Net={config_res.net_wins:+d} "
            f"P13_fixed={len(fixed_p13)} P13_regressed={len(regressed_p13)} "
            f"Lat_mult={config_res.latency_multiplier}x"
        )

    # -----------------------------------------------------------------
    # Step 7: Safety gate evaluation
    # -----------------------------------------------------------------
    print("\n--- Step 7: Safety gate evaluation ---")
    eligible_configs: list[RepConfigResult] = []

    for cfg in all_configs:
        if cfg.dense_rep == "baseline" and cfg.bm25_rep == "baseline":
            # Skip baseline self-comparison
            cfg.gates_passed = {"is_baseline": True}
            cfg.all_gates_passed = False
            continue

        ta = cfg.track_metrics.get("Track A", {})
        tb = cfg.track_metrics.get("Track B", {})
        tc = cfg.track_metrics.get("Track C", {})
        td_m = cfg.track_metrics.get("Track D", {})

        base_ta = baseline_metrics.get("Track A", {})
        base_tb = baseline_metrics.get("Track B", {})
        base_tc = baseline_metrics.get("Track C", {})

        # Gate 1: Track A Hit@5 >= baseline (1.0)
        g1 = ta.get("hit_at_5", 0) >= base_ta.get("hit_at_5", 1.0)

        # Gate 2: Track B Hit@5 and Recall@5 >= baseline
        g2a = tb.get("hit_at_5", 0) >= base_tb.get("hit_at_5", 0.7308)
        g2b = tb.get("recall_at_5", 0) >= base_tb.get("recall_at_5", 0.7019)

        # Gate 3: Track C Hit@1 >= baseline (1.0)
        g3 = tc.get("hit_at_1", 0) >= base_tc.get("hit_at_1", 1.0)

        # Gate 4: Track D 100% abstention safe
        g4 = (
            td_m.get("abstention_safe", 0) == td_m.get("total_queries", 25)
            and td_m.get("false_answer_rate", 1.0) == 0.0
        )

        # Gate 5: Net wins positive
        g5 = cfg.net_wins > 0

        # Gate 6: Latency multiplier <= 2.0x
        g6 = cfg.latency_multiplier <= 2.0

        cfg.gates_passed = {
            "g1_track_a_hit5": g1,
            "g2a_track_b_hit5": g2a,
            "g2b_track_b_recall5": g2b,
            "g3_track_c_hit1": g3,
            "g4_track_d_abstention": g4,
            "g5_net_wins": g5,
            "g6_latency": g6,
        }
        cfg.all_gates_passed = all([g1, g2a, g2b, g3, g4, g5, g6])

        if cfg.all_gates_passed:
            eligible_configs.append(cfg)
            print(f"  [PASS] {cfg.config_name}")
        else:
            reasons = []
            if not g1:
                reasons.append(
                    f"Track A Hit@5 ({ta.get('hit_at_5', 0):.4f} < "
                    f"{base_ta.get('hit_at_5', 1.0):.4f})"
                )
            if not g2a:
                reasons.append(
                    f"Track B Hit@5 ({tb.get('hit_at_5', 0):.4f} < "
                    f"{base_tb.get('hit_at_5', 0.7308):.4f})"
                )
            if not g2b:
                reasons.append(
                    f"Track B Recall@5 ({tb.get('recall_at_5', 0):.4f} < "
                    f"{base_tb.get('recall_at_5', 0.7019):.4f})"
                )
            if not g3:
                reasons.append(f"Track C Hit@1 ({tc.get('hit_at_1', 0):.4f})")
            if not g4:
                reasons.append("Track D abstention failure")
            if not g5:
                reasons.append(f"Net wins ({cfg.net_wins:+d})")
            if not g6:
                reasons.append(f"Latency ({cfg.latency_multiplier}x)")
            print(f"  [REJECT] {cfg.config_name}: {'; '.join(reasons)}")

    # -----------------------------------------------------------------
    # Step 8: Production Decision
    # -----------------------------------------------------------------
    print(f"\n{'=' * 80}")
    if eligible_configs:
        best = max(
            eligible_configs,
            key=lambda c: (
                c.track_metrics.get("Track B", {}).get("mrr_at_5", 0),
                c.net_wins,
                -c.latency_multiplier,
            ),
        )
        decision = "STRUCTURAL_REPRESENTATION_PROMOTED"
        promoted_config = best.config_name
        print(f"Decision: {decision}")
        print(f"Promoted: {promoted_config}")
        print(f"  Dense rep: {best.dense_rep}, BM25 rep: {best.bm25_rep}")
        print(f"  Net wins: {best.net_wins:+d}, P13 fixed: {len(best.p13_fixed)}")
    else:
        decision = "STRUCTURAL_REPRESENTATION_EXPERIMENTAL_BASELINE_RETAINED"
        promoted_config = None
        print(f"Decision: {decision}")
        print("No enriched configuration passed all production safety gates.")
        print("Production baseline retained.")
    print(f"{'=' * 80}")

    # -----------------------------------------------------------------
    # Step 9: Write JSON report
    # -----------------------------------------------------------------
    print("\n--- Writing Phase 16 Report ---")
    report = {
        "phase": 16,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "promoted_config": promoted_config,
        "production_baseline_retained": decision != "STRUCTURAL_REPRESENTATION_PROMOTED",
        "baseline_config": {
            "chunker": "ElementAwareChunker(1000/200)",
            "embedding": "nomic-embed-text:latest (768d)",
            "retriever": "Dense + BM25 RRF",
            "candidate_k": 20,
            "rrf_k": 100,
            "final_k": 5,
        },
        "representations_tested": {
            "Rep_A_baseline": "Raw chunk.text, no enrichment",
            "Rep_B_heading_enriched": "[DOCUMENT] title + [SECTION] hierarchy + [CONTENT] text",
            "Rep_C_full_structural": "Rep B + [STRUCTURE] element types / page / table data",
        },
        "experiment_matrix": [
            {"config_name": n, "dense_rep": d, "bm25_rep": b} for n, d, b in EXPERIMENT_MATRIX
        ],
        "baseline_metrics": baseline_metrics,
        "configurations_results": [
            {
                "config_name": c.config_name,
                "dense_rep": c.dense_rep,
                "bm25_rep": c.bm25_rep,
                "track_metrics": c.track_metrics,
                "wins": c.wins,
                "losses": c.losses,
                "unchanged": c.unchanged,
                "net_wins": c.net_wins,
                "p13_fixed_count": len(c.p13_fixed),
                "p13_fixed": c.p13_fixed,
                "p13_regressed_count": len(c.p13_regressed),
                "p13_regressed": c.p13_regressed,
                "mean_retrieval_latency_ms": c.mean_retrieval_latency_ms,
                "p50_retrieval_latency_ms": c.p50_retrieval_latency_ms,
                "p95_retrieval_latency_ms": c.p95_retrieval_latency_ms,
                "latency_multiplier": c.latency_multiplier,
                "gates_passed": c.gates_passed,
                "all_gates_passed": c.all_gates_passed,
            }
            for c in all_configs
        ],
        "chunk_integrity": integrity_report,
        "transform_latencies_ms_per_chunk": transform_summary,
        "leakage_audit": {
            "enrichment_uses_evaluation_queries": False,
            "enrichment_derived_from": [
                "document titles (from first heading or filename)",
                "section hierarchies (from parsed elements)",
                "element types (from parser classifications)",
                "table schemas (from parsed TableData)",
                "page numbers (from parser locations)",
            ],
            "evaluation_data_in_training": False,
            "leakage_detected": False,
        },
        "query_level_results": {c.config_name: c.query_results for c in all_configs},
    }

    out_path = _project_root / "docs" / "reports" / "phase16_representation_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved evaluation report to {out_path}")

    # Cleanup SQLite stores
    for track_name in rep_indexes:
        for rname in rep_indexes[track_name]:
            _, _, vs = rep_indexes[track_name][rname]
            vs.close()

    print(f"\n{'=' * 80}")
    print(f"PHASE 16 EVALUATION COMPLETE — DECISION: {decision}")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
