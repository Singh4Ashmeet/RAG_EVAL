"""Phase 13 — Retrieval Failure Forensics & Bottleneck Analysis.

Runs the frozen production baseline (Dense + BM25 RRF) against all tracks,
captures query-level evidence, classifies every failure into the detailed
taxonomy, and cross-references with Phase 11 oracle data.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.embed.ollama_embedder import OllamaEmbedder
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.metrics.retrieval import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k
from eval.routing.oracle import CachedEmbedder

# ---------------------------------------------------------------------------
# Failure Taxonomy (Phase 13 — granular)
# ---------------------------------------------------------------------------
FAILURE_CATEGORIES = [
    "CORPUS_COVERAGE",  # A — info not in any indexed chunk
    "CHUNKING_BOUNDARY",  # B — answer exists but chunk boundary splits it
    "LEXICAL_MISMATCH",  # C — BM25 fails: different terminology
    "SEMANTIC_MISMATCH",  # D — Dense fails despite relevant content
    "FUSION_FAILURE",  # E — both retrievers have evidence, RRF ranks too low
    "RANKING_FAILURE",  # F — in candidates but lost in final top-k
    "IDENTIFIER_TOKENIZATION",  # G — IDs, versions, error codes, symbols
    "MULTI_CHUNK",  # H — answer requires combining multiple chunks
    "EVALUATION_ANNOTATION",  # I — ground truth annotation issue
    "GENERATION_FAILURE",  # J — retrieval OK but generation fails
    "NONE",  # success
    "ABSTENTION_SAFE",  # Track D — intentionally unanswerable
    "UNKNOWN",  # genuinely ambiguous
]


@dataclass
class QueryForensics:
    """Full forensic record for a single query."""

    query_id: str
    query: str
    track: str
    reference_answer: str
    relevant_chunk_ids: list[str]
    relevant_element_ids: list[str]
    category: str = ""
    difficulty: str = ""
    requires_multi_chunk: bool = False

    # Retrieval results
    hybrid_retrieved_ids: list[str] = field(default_factory=list)
    hybrid_scores: list[float] = field(default_factory=list)
    bm25_rank: int | None = None  # rank in BM25 top-20 (1-indexed), None=miss
    dense_rank: int | None = None  # rank in Dense top-20 (1-indexed), None=miss
    hybrid_rank: int | None = None  # rank in Hybrid top-5 (1-indexed), None=miss

    # Extended candidate ranks (top-20)
    bm25_rank_20: int | None = None
    dense_rank_20: int | None = None

    # Metrics
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr_at_5: float = 0.0
    recall_at_5: float = 0.0
    latency_ms: float = 0.0

    # Failure classification
    failure_class: str = "NONE"
    failure_reason: str = ""

    # Whether chunk text actually contains the answer evidence
    chunk_contains_evidence: bool | None = None

    # Cross-reference with previous experiments
    previous_experiments: dict = field(default_factory=dict)


@dataclass
class TrackForensicsSummary:
    """Aggregate forensics for a track."""

    track: str
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float
    recall_at_5: float
    failures_at_5: int
    suboptimal_rank: int
    success_at_1: int
    failure_distribution: dict[str, int] = field(default_factory=dict)


def classify_failure(
    qf: QueryForensics,
    chunk_texts: dict[str, str],
) -> tuple[str, str]:
    """Classify a failure into the Phase 13 taxonomy.

    Returns (failure_class, failure_reason).
    """
    if qf.hit_at_5 == 1.0:
        if qf.hit_at_1 == 1.0:
            return "NONE", "Retrieved at rank 1."
        return "RANKING_FAILURE", (
            f"Relevant chunk found at hybrid rank {qf.hybrid_rank} (not rank 1). "
            f"BM25 rank={qf.bm25_rank}, Dense rank={qf.dense_rank}."
        )

    # --- Hit@5 = 0: complete retrieval miss ---

    # Check if relevant chunk exists in corpus
    relevant_text_found = False
    for cid in qf.relevant_chunk_ids:
        if cid in chunk_texts and chunk_texts[cid].strip():
            relevant_text_found = True
            break

    if not relevant_text_found:
        return "CORPUS_COVERAGE", "Relevant chunk ID not found in indexed corpus."

    # Check if requires multi-chunk
    if qf.requires_multi_chunk or len(qf.relevant_chunk_ids) > 1:
        return "MULTI_CHUNK", (
            f"Answer requires {len(qf.relevant_chunk_ids)} chunks. "
            f"BM25 rank={qf.bm25_rank_20}, Dense rank={qf.dense_rank_20}."
        )

    # Both retrievers miss entirely (not even in top-20)
    if qf.bm25_rank_20 is None and qf.dense_rank_20 is None:
        # Check if the chunk text has the answer content
        chunk_texts.get(qf.relevant_chunk_ids[0], "")
        # Heuristic: if the chunk is very short or mostly ABNF/grammar notation
        has_identifiers = any(
            kw in qf.query.lower()
            for kw in [
                "nan",
                "infinity",
                "ieee",
                "utf-8",
                "ecma",
                "abnf",
                "rfc",
                "bcp",
                "errata",
                "section",
                "appendix",
            ]
        )
        if has_identifiers:
            return "IDENTIFIER_TOKENIZATION", (
                "Both BM25 and Dense miss entirely. Query contains technical "
                "identifiers/standards references that don't match chunk vocabulary."
            )
        return "SEMANTIC_MISMATCH", (
            "Both BM25 and Dense miss entirely in top-20. "
            "Chunk exists but query representation doesn't match content."
        )

    # Only Dense finds it (BM25 misses)
    if qf.bm25_rank_20 is None and qf.dense_rank_20 is not None:
        if qf.dense_rank_20 <= 5:
            return "FUSION_FAILURE", (
                f"Dense finds at rank {qf.dense_rank_20} but BM25 misses entirely. "
                f"RRF cannot boost a single-source result high enough."
            )
        return "LEXICAL_MISMATCH", (
            f"BM25 misses entirely; Dense finds at rank {qf.dense_rank_20}. "
            f"Query terminology differs from chunk vocabulary."
        )

    # Only BM25 finds it (Dense misses)
    if qf.bm25_rank_20 is not None and qf.dense_rank_20 is None:
        if qf.bm25_rank_20 <= 5:
            return "FUSION_FAILURE", (
                f"BM25 finds at rank {qf.bm25_rank_20} but Dense misses entirely. "
                f"RRF cannot boost a single-source result high enough."
            )
        return "SEMANTIC_MISMATCH", (
            f"Dense misses entirely; BM25 finds at rank {qf.bm25_rank_20}. "
            f"Semantic embedding doesn't capture relevance."
        )

    # Both find it, but outside top-5 of hybrid
    bm25_r = qf.bm25_rank_20 or 999
    dense_r = qf.dense_rank_20 or 999
    best_individual = min(bm25_r, dense_r)

    if best_individual <= 5:
        return "FUSION_FAILURE", (
            f"Best individual rank={best_individual} (BM25={qf.bm25_rank_20}, "
            f"Dense={qf.dense_rank_20}) but RRF fusion places it outside top-5."
        )

    if best_individual <= 10:
        return "RANKING_FAILURE", (
            f"Found in extended candidates (BM25={qf.bm25_rank_20}, "
            f"Dense={qf.dense_rank_20}) but ranked too low after fusion."
        )

    # Both find it but very deep
    has_identifier_terms = any(
        kw in qf.query.lower()
        for kw in [
            "ieee",
            "754",
            "rfc",
            "section",
            "errata",
            "bcp",
            "abnf",
            "ecma",
            "appendix",
            "version",
            "id",
        ]
    )
    if has_identifier_terms:
        return "IDENTIFIER_TOKENIZATION", (
            f"Technical identifier query. BM25={qf.bm25_rank_20}, "
            f"Dense={qf.dense_rank_20}. Both rank too deep."
        )

    return "SEMANTIC_MISMATCH", (
        f"Both retrievers rank the relevant chunk deep "
        f"(BM25={qf.bm25_rank_20}, Dense={qf.dense_rank_20})."
    )


def build_retrievers(
    doc_paths: list[Path],
    embedder: CachedEmbedder,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[BM25Retriever, DenseRetriever, HybridRetriever, SQLiteVectorStore, dict[str, str]]:
    """Build all retrievers and return chunk text lookup."""
    import tempfile

    pipeline = create_default_ingestion_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []
    chunk_texts: dict[str, str] = {}

    db_file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    vector_store = SQLiteVectorStore(db_file.name)
    dense = DenseRetriever(embedder, vector_store)

    for p in doc_paths:
        if not p.exists():
            continue
        res = pipeline.ingest(p)
        if res.document and res.chunks:
            all_chunks.extend(res.chunks)
            dense.index_document(res.document.metadata.document_id, res.chunks)
            for c in res.chunks:
                chunk_texts[c.chunk_id] = c.text

    bm25 = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=100, candidate_k=20)
    return bm25, dense, hybrid, vector_store, chunk_texts


def analyze_track(
    track_name: str,
    dataset_file: str,
    doc_paths: list[Path],
    embedder: CachedEmbedder,
    is_abstention: bool = False,
) -> tuple[list[QueryForensics], TrackForensicsSummary]:
    """Run full forensic analysis on a single track."""

    ds_path = _project_root / dataset_file
    dataset = load_jsonl_dataset(str(ds_path))
    cases = dataset.cases

    bm25, dense, hybrid, vs, chunk_texts = build_retrievers(doc_paths, embedder)
    embedder.warm_cache([c.query for c in cases])

    results: list[QueryForensics] = []

    for case in cases:
        t0 = time.perf_counter()
        hybrid_res = hybrid.retrieve(case.query, top_k=5)
        t1 = time.perf_counter()

        hybrid_ids = [r.chunk.chunk_id for r in hybrid_res]
        hybrid_scores = [r.score for r in hybrid_res]

        # Extended candidate search
        bm25_res_20 = bm25.retrieve(case.query, top_k=20)
        dense_res_20 = dense.retrieve(case.query, top_k=20)

        bm25_ids_20 = [r.chunk.chunk_id for r in bm25_res_20]
        dense_ids_20 = [r.chunk.chunk_id for r in dense_res_20]

        # Find ranks
        def find_rank(ids: list[str], relevant: list[str]) -> int | None:
            for i, cid in enumerate(ids):
                if cid in relevant:
                    return i + 1
            return None

        bm25_rank_5 = find_rank(bm25_ids_20[:5], case.relevant_chunk_ids)
        dense_rank_5 = find_rank(dense_ids_20[:5], case.relevant_chunk_ids)
        hybrid_rank = find_rank(hybrid_ids, case.relevant_chunk_ids)
        bm25_rank_20 = find_rank(bm25_ids_20, case.relevant_chunk_ids)
        dense_rank_20 = find_rank(dense_ids_20, case.relevant_chunk_ids)

        # Metrics
        rel_ids = case.relevant_chunk_ids
        if not rel_ids or is_abstention:
            h1 = h3 = h5 = mrr = rec = 0.0
        else:
            h1 = hit_rate_at_k(hybrid_ids, rel_ids, k=1)
            h3 = hit_rate_at_k(hybrid_ids, rel_ids, k=3)
            h5 = hit_rate_at_k(hybrid_ids, rel_ids, k=5)
            mrr = reciprocal_rank_at_k(hybrid_ids, rel_ids, k=5)
            rec = recall_at_k(hybrid_ids, rel_ids, k=5)

        qf = QueryForensics(
            query_id=case.query_id,
            query=case.query,
            track=track_name,
            reference_answer=case.reference_answer or "",
            relevant_chunk_ids=case.relevant_chunk_ids,
            relevant_element_ids=case.relevant_element_ids,
            category=getattr(case, "category", ""),
            difficulty=getattr(case, "difficulty", ""),
            requires_multi_chunk=getattr(case, "requires_multi_chunk", False),
            hybrid_retrieved_ids=hybrid_ids,
            hybrid_scores=hybrid_scores,
            bm25_rank=bm25_rank_5,
            dense_rank=dense_rank_5,
            hybrid_rank=hybrid_rank,
            bm25_rank_20=bm25_rank_20,
            dense_rank_20=dense_rank_20,
            hit_at_1=h1,
            hit_at_3=h3,
            hit_at_5=h5,
            mrr_at_5=mrr,
            recall_at_5=rec,
            latency_ms=(t1 - t0) * 1000.0,
        )

        if is_abstention:
            qf.failure_class = "ABSTENTION_SAFE"
            qf.failure_reason = "Intentionally unanswerable — abstention query."
        else:
            fc, fr = classify_failure(qf, chunk_texts)
            qf.failure_class = fc
            qf.failure_reason = fr

        results.append(qf)

    vs.close()

    # Aggregate
    n = len(results)
    answerable = [r for r in results if r.failure_class != "ABSTENTION_SAFE"]
    if answerable:
        agg_h1 = statistics.fmean(r.hit_at_1 for r in answerable)
        agg_h3 = statistics.fmean(r.hit_at_3 for r in answerable)
        agg_h5 = statistics.fmean(r.hit_at_5 for r in answerable)
        agg_mrr = statistics.fmean(r.mrr_at_5 for r in answerable)
        agg_rec = statistics.fmean(r.recall_at_5 for r in answerable)
    else:
        agg_h1 = agg_h3 = agg_h5 = agg_mrr = agg_rec = 0.0

    failures_at_5 = sum(1 for r in answerable if r.hit_at_5 == 0.0)
    suboptimal = sum(1 for r in answerable if r.hit_at_5 == 1.0 and r.hit_at_1 == 0.0)
    success_at_1 = sum(1 for r in answerable if r.hit_at_1 == 1.0)

    fd = Counter(r.failure_class for r in results)

    summary = TrackForensicsSummary(
        track=track_name,
        total_queries=n,
        hit_at_1=round(agg_h1, 4),
        hit_at_3=round(agg_h3, 4),
        hit_at_5=round(agg_h5, 4),
        mrr_at_5=round(agg_mrr, 4),
        recall_at_5=round(agg_rec, 4),
        failures_at_5=failures_at_5,
        suboptimal_rank=suboptimal,
        success_at_1=success_at_1,
        failure_distribution=dict(fd),
    )

    return results, summary


def analyze_oracle_data() -> dict:
    """Analyze Phase 11 oracle labels for strategy distribution and no-clear-winner queries."""

    oracle_path = _project_root / "data" / "routing" / "phase11" / "oracle_labels.jsonl"
    if not oracle_path.exists():
        return {"error": "oracle_labels.jsonl not found"}

    records = []
    with open(oracle_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    strategy_counts = Counter(r["oracle_strategy"] for r in records)

    # Analyze no_clear_winner queries
    no_winner = [r for r in records if r["oracle_strategy"] == "no_clear_winner"]
    no_winner_analysis = []
    for r in no_winner:
        bm25_h5 = r["bm25_metrics"]["hit_at_5"]
        dense_h5 = r["dense_metrics"]["hit_at_5"]
        hybrid_h5 = r["hybrid_metrics"]["hit_at_5"]
        all_miss = bm25_h5 == 0.0 and dense_h5 == 0.0 and hybrid_h5 == 0.0
        all_hit = bm25_h5 == 1.0 and dense_h5 == 1.0 and hybrid_h5 == 1.0

        no_winner_analysis.append(
            {
                "example_id": r["example_id"],
                "query": r["query"],
                "source_document": r["source_document"],
                "split": r["split"],
                "bm25_mrr": r["bm25_metrics"]["mrr_at_5"],
                "dense_mrr": r["dense_metrics"]["mrr_at_5"],
                "hybrid_mrr": r["hybrid_metrics"]["mrr_at_5"],
                "bm25_hit5": bm25_h5,
                "dense_hit5": dense_h5,
                "hybrid_hit5": hybrid_h5,
                "all_miss": all_miss,
                "all_hit": all_hit,
            }
        )

    # Categorize no_clear_winner reasons
    ncw_all_miss = sum(1 for x in no_winner_analysis if x["all_miss"])
    ncw_all_hit = sum(1 for x in no_winner_analysis if x["all_hit"])
    ncw_partial = len(no_winner_analysis) - ncw_all_miss - ncw_all_hit

    # Analyze ambiguous queries
    ambiguous = [r for r in records if r["oracle_strategy"] == "ambiguous"]
    ambig_analysis = []
    ambig_all_perfect = 0
    ambig_all_hit5 = 0
    ambig_mixed = 0

    for r in ambiguous:
        bm25_mrr = r["bm25_metrics"]["mrr_at_5"]
        dense_mrr = r["dense_metrics"]["mrr_at_5"]
        hybrid_mrr = r["hybrid_metrics"]["mrr_at_5"]
        bm25_h5 = r["bm25_metrics"]["hit_at_5"]
        dense_h5 = r["dense_metrics"]["hit_at_5"]
        hybrid_h5 = r["hybrid_metrics"]["hit_at_5"]

        all_perfect = bm25_mrr == 1.0 and dense_mrr == 1.0 and hybrid_mrr == 1.0
        all_hit = bm25_h5 == 1.0 and dense_h5 == 1.0 and hybrid_h5 == 1.0

        if all_perfect:
            ambig_all_perfect += 1
        elif all_hit:
            ambig_all_hit5 += 1
        else:
            ambig_mixed += 1

        ambig_analysis.append(
            {
                "example_id": r["example_id"],
                "query": r["query"],
                "source_document": r["source_document"],
                "bm25_mrr": bm25_mrr,
                "dense_mrr": dense_mrr,
                "hybrid_mrr": hybrid_mrr,
                "all_perfect_mrr": all_perfect,
            }
        )

    return {
        "total_records": len(records),
        "strategy_distribution": dict(strategy_counts),
        "no_clear_winner": {
            "count": len(no_winner),
            "all_strategies_miss": ncw_all_miss,
            "all_strategies_hit": ncw_all_hit,
            "partial_hits": ncw_partial,
            "details": no_winner_analysis,
        },
        "ambiguous": {
            "count": len(ambiguous),
            "all_perfect_mrr": ambig_all_perfect,
            "all_hit5_but_different_mrr": ambig_all_hit5,
            "mixed_hit_patterns": ambig_mixed,
            "details": ambig_analysis,
        },
    }


def build_previous_experiment_comparison() -> dict:
    """Extract Track B per-query results from Phase reports for cross-reference."""

    phase_data: dict[str, dict] = {}

    # Phase 4 data: Dense, BM25, Hybrid, Reranker
    phase_data["phase4"] = {
        "Dense": {"hit_at_1": 0.4231, "hit_at_3": 0.5192, "hit_at_5": 0.6538, "mrr_at_5": 0.5029},
        "BM25": {"hit_at_1": 0.4038, "hit_at_3": 0.5769, "hit_at_5": 0.6731, "mrr_at_5": 0.5019},
        "Hybrid_RRF": {
            "hit_at_1": 0.4038,
            "hit_at_3": 0.5962,
            "hit_at_5": 0.7308,
            "mrr_at_5": 0.5212,
        },
        "Reranker_MiniLM": {
            "hit_at_1": 0.5385,
            "hit_at_3": 0.6731,
            "hit_at_5": 0.7115,
            "mrr_at_5": 0.5994,
        },
    }

    # Phase 6: SPLADE
    phase_data["phase6"] = {
        "SPLADE_only": {
            "hit_at_1": 0.0577,
            "hit_at_3": 0.2308,
            "hit_at_5": 0.4231,
            "mrr_at_5": 0.1798,
        },
        "Dense_SPLADE": {
            "hit_at_1": 0.2692,
            "hit_at_3": 0.6346,
            "hit_at_5": 0.7308,
            "mrr_at_5": 0.4462,
        },
        "complementarity_splade_only_recovered": 2,
        "complementarity_none_recovered": 9,
    }

    # Phase 7: Reranking
    phase_data["phase7"] = {
        "MiniLM_L6": {
            "hit_at_1": 0.5385,
            "hit_at_3": 0.6731,
            "hit_at_5": 0.7115,
            "mrr_at_5": 0.5994,
        },
        "MiniLM_L12": {
            "hit_at_1": 0.4615,
            "hit_at_3": 0.6538,
            "hit_at_5": 0.7115,
            "mrr_at_5": 0.5679,
        },
        "TinyBERT": {
            "hit_at_1": 0.4231,
            "hit_at_3": 0.6538,
            "hit_at_5": 0.7115,
            "mrr_at_5": 0.5372,
        },
    }

    # Phase 8: HyDE
    phase_data["phase8"] = {
        "HyDE_Dense": {
            "hit_at_1": 0.3654,
            "hit_at_3": 0.6731,
            "hit_at_5": 0.7885,
            "mrr_at_5": 0.5247,
        },
        "Dense_HyDE_RRF": {
            "hit_at_1": 0.4615,
            "hit_at_3": 0.6346,
            "hit_at_5": 0.7115,
            "mrr_at_5": 0.5609,
        },
    }

    # Phase 9: Routing
    phase_data["phase9"] = {
        "Oracle_routing": {
            "hit_at_1": 0.5769,
            "hit_at_3": 0.7692,
            "hit_at_5": 0.8462,
            "mrr_at_5": 0.7115,
        },
    }

    return phase_data


def compute_headroom(baseline: dict, experiments: dict) -> dict:
    """Compute retrieval headroom from previous experiments."""

    best_config: dict[str, tuple[float, str]] = {}
    for metric in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5"]:
        best_val = baseline[metric]
        best_name = "Baseline (Dense+BM25 RRF)"

        for phase, configs in experiments.items():
            for config_name, metrics in configs.items():
                if isinstance(metrics, dict) and metric in metrics:
                    if metrics[metric] > best_val:
                        best_val = metrics[metric]
                        best_name = f"{phase}/{config_name}"

        best_config[metric] = (round(best_val, 4), best_name)

    return best_config


def main() -> int:
    print(f"\n{'=' * 80}")
    print("PHASE 13 — RETRIEVAL FAILURE FORENSICS & BOTTLENECK ANALYSIS")
    print(f"{'=' * 80}\n")

    raw_embedder = OllamaEmbedder(model="nomic-embed-text", host="http://localhost:11434")
    embedder = CachedEmbedder(raw_embedder)

    # --- Track configurations ---
    tracks = [
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
            [
                Path("eval/corpora/track_b/rfc8259.txt"),
            ],
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
            [
                Path("eval/corpora/track_d/reference_guide.md"),
            ],
            True,
        ),
    ]

    all_forensics: list[QueryForensics] = []
    track_summaries: list[TrackForensicsSummary] = []

    for track_name, ds_file, doc_paths, is_abstention in tracks:
        print(f"\n--- Analyzing {track_name} ---")
        results, summary = analyze_track(track_name, ds_file, doc_paths, embedder, is_abstention)
        all_forensics.extend(results)
        track_summaries.append(summary)

        print(f"  Queries: {summary.total_queries}")
        print(
            f"  Hit@1={summary.hit_at_1:.4f} Hit@3={summary.hit_at_3:.4f} "
            f"Hit@5={summary.hit_at_5:.4f} MRR@5={summary.mrr_at_5:.4f}"
        )
        print(
            f"  Failures@5: {summary.failures_at_5}, Suboptimal: {summary.suboptimal_rank}, "
            f"Success@1: {summary.success_at_1}"
        )
        print(f"  Failure distribution: {dict(summary.failure_distribution)}")

    # --- Oracle data analysis ---
    print("\n--- Analyzing Phase 11 Oracle Data ---")
    oracle_analysis = analyze_oracle_data()
    if "error" not in oracle_analysis:
        print(f"  Total oracle records: {oracle_analysis['total_records']}")
        print(f"  Strategy distribution: {oracle_analysis['strategy_distribution']}")
        ncw = oracle_analysis["no_clear_winner"]
        print(
            f"  No-clear-winner: {ncw['count']} "
            f"(all_miss={ncw['all_strategies_miss']}, all_hit={ncw['all_strategies_hit']}, "
            f"partial={ncw['partial_hits']})"
        )
        ambig = oracle_analysis["ambiguous"]
        print(
            f"  Ambiguous: {ambig['count']} "
            f"(perfect_mrr={ambig['all_perfect_mrr']}, hit5_diff_mrr={ambig['all_hit5_but_different_mrr']}, "
            f"mixed={ambig['mixed_hit_patterns']})"
        )

    # --- Previous experiment comparison ---
    print("\n--- Previous Experiment Comparison ---")
    prev_experiments = build_previous_experiment_comparison()

    baseline_metrics = {
        "hit_at_1": 0.4038,
        "hit_at_3": 0.5962,
        "hit_at_5": 0.7308,
        "mrr_at_5": 0.5212,
    }
    headroom = compute_headroom(baseline_metrics, prev_experiments)
    print("  Headroom analysis (Track B):")
    for metric, (val, name) in headroom.items():
        print(f"    {metric}: best={val} from {name} (baseline={baseline_metrics[metric]})")

    # --- Error clustering ---
    print("\n--- Error Clustering ---")
    answerable_failures = [
        r for r in all_forensics if r.failure_class not in ("NONE", "ABSTENTION_SAFE")
    ]
    failure_clusters = Counter(r.failure_class for r in answerable_failures)
    print(f"  {'Failure Cluster':<35} Count")
    print(f"  {'-' * 50}")
    for cluster, count in failure_clusters.most_common():
        print(f"  {cluster:<35} {count}")

    # --- Recoverable vs persistent failures ---
    # For Track B failures, check if any experimental method recovered them
    track_b_failures = [
        r
        for r in all_forensics
        if r.track == "Track B" and r.failure_class not in ("NONE", "ABSTENTION_SAFE")
    ]

    # Cross-reference with Phase 6 SPLADE complementarity (from report)
    # SPLADE alone found 2 queries that neither Dense nor BM25 found
    # None: 9 queries where no retriever found relevant chunk

    # --- Build output report ---
    report = {
        "phase": 13,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_config": {
            "chunker": "ElementAwareChunker",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "nomic-embed-text:latest",
            "embedding_dimensions": 768,
            "retrieval": "Dense + BM25 RRF",
            "candidate_k": 20,
            "rrf_k": 100,
            "final_k": 5,
            "generation_model": "llama3.2:3b",
        },
        "track_summaries": [asdict(s) for s in track_summaries],
        "failure_clusters": dict(failure_clusters.most_common()),
        "query_forensics": [asdict(qf) for qf in all_forensics],
        "oracle_analysis": oracle_analysis,
        "previous_experiments": prev_experiments,
        "headroom_analysis": {
            k: {"best_value": v[0], "best_config": v[1]} for k, v in headroom.items()
        },
        "recoverability": {
            "track_b_total_failures": len(track_b_failures),
            "track_b_failure_distribution": dict(
                Counter(r.failure_class for r in track_b_failures).most_common()
            ),
        },
    }

    # Write JSON report
    out_path = _project_root / "docs" / "reports" / "phase13_failure_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved report to {out_path}")

    print(f"\n{'=' * 80}")
    print("PHASE 13 FORENSIC ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
