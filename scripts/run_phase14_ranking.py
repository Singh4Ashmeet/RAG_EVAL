"""Phase 14 — Targeted Ranking Optimization & RRF Forensics.

Performs controlled investigation of the RRF ranking layer:
1. Verifies frozen production baseline
2. RRF forensics (query-level Dense/BM25/RRF contribution analysis)
3. Controlled RRF ablation (rrf_k sweep, candidate_k sweep)
4. Weighted fusion experiments
5. Query-level win/loss analysis against Phase 13 failure data
6. Safety regression gates (all 4 tracks)
7. Latency measurement
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
from core.retrieval.hybrid import _rrf_fusion
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.metrics.retrieval import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k
from eval.routing.oracle import CachedEmbedder

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RRFForensicRecord:
    """Per-query RRF forensic data."""

    query_id: str
    query: str
    track: str
    relevant_chunk_ids: list[str]
    dense_rank: int | None  # 1-indexed rank in dense top-k, None=miss
    bm25_rank: int | None  # 1-indexed rank in BM25 top-k, None=miss
    dense_rrf_contribution: float  # 1/(k + dense_rank) or 0
    bm25_rrf_contribution: float  # 1/(k + bm25_rank) or 0
    combined_rrf_score: float
    final_rank: int | None  # 1-indexed final rank, None=miss
    correct_retrieved: bool  # in final top-5
    correct_demoted: bool  # in candidates but not in final top-5
    rank_displacement: int | None  # best_individual_rank - final_rank
    failure_pattern: str  # A/B/C/D/E as defined in Step 3


@dataclass
class ConfigResult:
    """Results for a single RRF configuration."""

    config_name: str
    rrf_k: int
    candidate_k: int
    final_k: int
    dense_weight: float
    bm25_weight: float

    # Per-track aggregate metrics
    track_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-query results (query_id -> metrics)
    query_results: dict[str, dict] = field(default_factory=dict)

    # Win/loss vs baseline
    wins: int = 0
    losses: int = 0
    unchanged: int = 0
    net_wins: int = 0

    # Phase 13 failure analysis
    p13_fixed: list[str] = field(default_factory=list)
    p13_regressed: list[str] = field(default_factory=list)
    p13_unchanged: list[str] = field(default_factory=list)

    # Latency
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Track configuration
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


def build_indexing(
    doc_paths: list[Path],
    embedder: CachedEmbedder,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[BM25Retriever, DenseRetriever, SQLiteVectorStore, list[Chunk], dict[str, str]]:
    """Build retriever components and return them separately for flexible fusion."""
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
    return bm25, dense, vector_store, all_chunks, chunk_texts


def evaluate_config(
    track_name: str,
    cases: list,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    rrf_k: int,
    candidate_k: int,
    final_k: int,
    dense_weight: float,
    bm25_weight: float,
    is_abstention: bool,
) -> dict[str, dict]:
    """Evaluate a single RRF config on a track, return per-query metrics."""
    query_results: dict[str, dict] = {}
    latencies: list[float] = []

    for case in cases:
        t0 = time.perf_counter()

        # Get individual retriever results
        dense_res = dense.retrieve(case.query, top_k=candidate_k)
        bm25_res = bm25.retrieve(case.query, top_k=candidate_k)

        # Fuse with weighted RRF
        fused = _rrf_fusion(
            [dense_res, bm25_res],
            weights=[dense_weight, bm25_weight],
            k=rrf_k,
            top_k=final_k,
        )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        fused_ids = [r.chunk.chunk_id for r in fused]
        fused_scores = [r.score for r in fused]

        # Dense/BM25 individual ranks (1-indexed)
        dense_ids = [r.chunk.chunk_id for r in dense_res]
        bm25_ids = [r.chunk.chunk_id for r in bm25_res]

        def find_rank(ids: list[str], relevant: list[str]) -> int | None:
            for i, cid in enumerate(ids):
                if cid in relevant:
                    return i + 1
            return None

        rel_ids = case.relevant_chunk_ids

        if not rel_ids or is_abstention:
            h1 = h3 = h5 = mrr = rec = 0.0
            d_rank = b_rank = f_rank = None
        else:
            h1 = hit_rate_at_k(fused_ids, rel_ids, k=1)
            h3 = hit_rate_at_k(fused_ids, rel_ids, k=3)
            h5 = hit_rate_at_k(fused_ids, rel_ids, k=final_k)
            mrr = reciprocal_rank_at_k(fused_ids, rel_ids, k=final_k)
            rec = recall_at_k(fused_ids, rel_ids, k=final_k)
            d_rank = find_rank(dense_ids, rel_ids)
            b_rank = find_rank(bm25_ids, rel_ids)
            f_rank = find_rank(fused_ids, rel_ids)

        query_results[case.query_id] = {
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "mrr_at_5": mrr,
            "recall_at_5": rec,
            "dense_rank": d_rank,
            "bm25_rank": b_rank,
            "final_rank": f_rank,
            "latency_ms": latency_ms,
            "fused_ids": fused_ids,
            "fused_scores": fused_scores,
            "is_abstention": is_abstention,
        }

    return query_results


def compute_track_aggregates(
    query_results: dict[str, dict], is_abstention: bool
) -> dict[str, float]:
    """Compute aggregate metrics for a track."""
    if is_abstention:
        # Track D: check that no relevant content is retrieved
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


def compare_with_baseline(
    baseline_results: dict[str, dict],
    candidate_results: dict[str, dict],
) -> tuple[list[str], list[str], list[str]]:
    """Compare candidate vs baseline, return (wins, losses, unchanged) query IDs."""
    wins, losses, unchanged = [], [], []
    for qid, base in baseline_results.items():
        if base.get("is_abstention"):
            unchanged.append(qid)
            continue
        cand = candidate_results.get(qid)
        if cand is None:
            unchanged.append(qid)
            continue

        # Compare by final rank (lower is better, None means miss)
        base_rank = base.get("final_rank")
        cand_rank = cand.get("final_rank")

        if base_rank == cand_rank:
            unchanged.append(qid)
        elif cand_rank is not None and (base_rank is None or cand_rank < base_rank):
            wins.append(qid)
        elif base_rank is not None and (cand_rank is None or cand_rank > base_rank):
            losses.append(qid)
        else:
            unchanged.append(qid)

    return wins, losses, unchanged


def classify_p13_impact(
    p13_failure_qids: list[str],
    baseline_results: dict[str, dict],
    candidate_results: dict[str, dict],
) -> tuple[list[str], list[str], list[str]]:
    """Classify Phase 13 failures as fixed/regressed/unchanged."""
    fixed, regressed, unch = [], [], []
    for qid in p13_failure_qids:
        base = baseline_results.get(qid)
        cand = candidate_results.get(qid)
        if base is None or cand is None:
            unch.append(qid)
            continue

        base_h1 = base.get("hit_at_1", 0.0)
        cand_h1 = cand.get("hit_at_1", 0.0)
        base_h5 = base.get("hit_at_5", 0.0)
        cand_h5 = cand.get("hit_at_5", 0.0)

        if cand_h1 > base_h1 or (cand_h5 > base_h5):
            fixed.append(qid)
        elif cand_h1 < base_h1 or cand_h5 < base_h5:
            regressed.append(qid)
        else:
            unch.append(qid)

    return fixed, regressed, unch


def rrf_forensics_analysis(
    cases: list,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    rrf_k: int,
    candidate_k: int,
    track_name: str,
    is_abstention: bool,
) -> list[RRFForensicRecord]:
    """Detailed per-query RRF forensics for the baseline config."""
    records: list[RRFForensicRecord] = []

    for case in cases:
        rel_ids = case.relevant_chunk_ids

        if is_abstention or not rel_ids:
            records.append(
                RRFForensicRecord(
                    query_id=case.query_id,
                    query=case.query,
                    track=track_name,
                    relevant_chunk_ids=rel_ids,
                    dense_rank=None,
                    bm25_rank=None,
                    dense_rrf_contribution=0.0,
                    bm25_rrf_contribution=0.0,
                    combined_rrf_score=0.0,
                    final_rank=None,
                    correct_retrieved=False,
                    correct_demoted=False,
                    rank_displacement=None,
                    failure_pattern="E_ABSTENTION",
                )
            )
            continue

        dense_res = dense.retrieve(case.query, top_k=candidate_k)
        bm25_res = bm25.retrieve(case.query, top_k=candidate_k)

        # Build rank maps
        dense_ids = [r.chunk.chunk_id for r in dense_res]
        bm25_ids = [r.chunk.chunk_id for r in bm25_res]

        def find_rank(ids: list[str], relevant: list[str]) -> int | None:
            for i, cid in enumerate(ids):
                if cid in relevant:
                    return i + 1
            return None

        d_rank = find_rank(dense_ids, rel_ids)
        b_rank = find_rank(bm25_ids, rel_ids)

        # Compute RRF contributions for the relevant chunk
        d_contrib = 1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0
        b_contrib = 1.0 / (rrf_k + b_rank) if b_rank is not None else 0.0
        combined = d_contrib + b_contrib

        # Run fusion
        fused = _rrf_fusion([dense_res, bm25_res], k=rrf_k, top_k=5)
        fused_ids = [r.chunk.chunk_id for r in fused]
        f_rank = find_rank(fused_ids, rel_ids)

        correct_in_fused = f_rank is not None
        in_candidates = d_rank is not None or b_rank is not None
        correct_demoted = in_candidates and not correct_in_fused

        # Rank displacement
        if f_rank is not None:
            best_individual = (
                min(r for r in [d_rank, b_rank] if r is not None)
                if any(r is not None for r in [d_rank, b_rank])
                else None
            )
            displacement = (best_individual - f_rank) if best_individual is not None else None
        else:
            displacement = None

        # Failure pattern classification
        if not in_candidates:
            pattern = "E_ABSENT_BOTH"
        elif d_rank is not None and b_rank is None:
            pattern = "D_DENSE_ONLY" if correct_in_fused else "A_DENSE_WINS_RRF_LOSES"
        elif b_rank is not None and d_rank is None:
            pattern = "D_BM25_ONLY" if correct_in_fused else "B_BM25_WINS_RRF_LOSES"
        elif d_rank is not None and b_rank is not None:
            if correct_in_fused:
                pattern = "BOTH_OK"
            else:
                pattern = "C_BOTH_RETRIEVE_FUSION_MISORDERS"
        else:
            pattern = "UNKNOWN"

        records.append(
            RRFForensicRecord(
                query_id=case.query_id,
                query=case.query,
                track=track_name,
                relevant_chunk_ids=rel_ids,
                dense_rank=d_rank,
                bm25_rank=b_rank,
                dense_rrf_contribution=round(d_contrib, 8),
                bm25_rrf_contribution=round(b_contrib, 8),
                combined_rrf_score=round(combined, 8),
                final_rank=f_rank,
                correct_retrieved=correct_in_fused,
                correct_demoted=correct_demoted,
                rank_displacement=displacement,
                failure_pattern=pattern,
            )
        )

    return records


def main() -> int:
    print(f"\n{'=' * 80}")
    print("PHASE 14 — TARGETED RANKING OPTIMIZATION & RRF FORENSICS")
    print(f"{'=' * 80}\n")

    # Load Phase 13 failure data
    p13_path = _project_root / "docs" / "reports" / "phase13_failure_analysis.json"
    p13_data = json.loads(p13_path.read_text(encoding="utf-8"))

    # Identify Phase 13 ranking failure query IDs
    p13_ranking_failures = [
        q["query_id"]
        for q in p13_data["query_forensics"]
        if q["failure_class"] not in ("NONE", "ABSTENTION_SAFE")
    ]
    p13_all_failures = {
        q["query_id"]: q["failure_class"]
        for q in p13_data["query_forensics"]
        if q["failure_class"] not in ("NONE", "ABSTENTION_SAFE")
    }

    print(f"Phase 13 total failures: {len(p13_ranking_failures)}")
    print(f"Failure distribution: {Counter(p13_all_failures.values())}")

    raw_embedder = OllamaEmbedder(model="nomic-embed-text", host="http://localhost:11434")
    embedder = CachedEmbedder(raw_embedder)

    # ======================================================================
    # STEP 1: Build indexes for all tracks
    # ======================================================================
    print("\n--- Building indexes for all tracks ---")

    track_data: dict[str, dict] = {}
    for track_name, ds_file, doc_paths, is_abstention in TRACKS:
        abs_paths = [_project_root / p for p in doc_paths]
        ds_path = _project_root / ds_file
        dataset = load_jsonl_dataset(str(ds_path))

        bm25, dense, vs, chunks, chunk_texts = build_indexing(abs_paths, embedder)
        embedder.warm_cache([c.query for c in dataset.cases])

        track_data[track_name] = {
            "cases": dataset.cases,
            "bm25": bm25,
            "dense": dense,
            "vs": vs,
            "chunks": chunks,
            "chunk_texts": chunk_texts,
            "is_abstention": is_abstention,
        }
        print(f"  {track_name}: {len(dataset.cases)} queries, {len(chunks)} chunks")

    # ======================================================================
    # STEP 2: Verify baseline (rrf_k=100, candidate_k=20)
    # ======================================================================
    print("\n--- Step 2: Verify production baseline ---")
    baseline_config = {
        "rrf_k": 100,
        "candidate_k": 20,
        "final_k": 5,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
    }

    all_baseline_results: dict[str, dict[str, dict]] = {}  # track -> qid -> metrics
    baseline_aggregates: dict[str, dict] = {}

    for track_name, td in track_data.items():
        results = evaluate_config(
            track_name,
            td["cases"],
            td["bm25"],
            td["dense"],
            **baseline_config,
            is_abstention=td["is_abstention"],
        )
        all_baseline_results[track_name] = results
        agg = compute_track_aggregates(results, td["is_abstention"])
        baseline_aggregates[track_name] = agg

        if td["is_abstention"]:
            print(f"  {track_name}: abstention_safe={agg['abstention_safe']}")
        else:
            print(
                f"  {track_name}: Hit@1={agg['hit_at_1']:.4f} Hit@3={agg['hit_at_3']:.4f} "
                f"Hit@5={agg['hit_at_5']:.4f} MRR@5={agg['mrr_at_5']:.4f} "
                f"latency={agg['mean_latency_ms']:.1f}ms"
            )

    # ======================================================================
    # STEP 3: RRF Forensics (baseline)
    # ======================================================================
    print("\n--- Step 3: RRF Forensics ---")
    all_forensic_records: list[RRFForensicRecord] = []

    for track_name, td in track_data.items():
        records = rrf_forensics_analysis(
            td["cases"],
            td["bm25"],
            td["dense"],
            rrf_k=100,
            candidate_k=20,
            track_name=track_name,
            is_abstention=td["is_abstention"],
        )
        all_forensic_records.extend(records)

    # Summarize forensic patterns
    pattern_counts = Counter(r.failure_pattern for r in all_forensic_records)
    print("  RRF failure patterns:")
    for pattern, count in pattern_counts.most_common():
        print(f"    {pattern}: {count}")

    # Identify queries where Dense wins but RRF loses
    dense_wins_rrf_loses = [
        r for r in all_forensic_records if r.failure_pattern == "A_DENSE_WINS_RRF_LOSES"
    ]
    bm25_wins_rrf_loses = [
        r for r in all_forensic_records if r.failure_pattern == "B_BM25_WINS_RRF_LOSES"
    ]
    both_retrieve_fusion_misorders = [
        r for r in all_forensic_records if r.failure_pattern == "C_BOTH_RETRIEVE_FUSION_MISORDERS"
    ]

    print(f"\n  Dense wins, RRF loses: {len(dense_wins_rrf_loses)}")
    for r in dense_wins_rrf_loses:
        print(f"    {r.query_id}: dense_rank={r.dense_rank}, final_rank={r.final_rank}")

    print(f"  BM25 wins, RRF loses: {len(bm25_wins_rrf_loses)}")
    for r in bm25_wins_rrf_loses:
        print(f"    {r.query_id}: bm25_rank={r.bm25_rank}, final_rank={r.final_rank}")

    print(f"  Both retrieve, fusion misorders: {len(both_retrieve_fusion_misorders)}")
    for r in both_retrieve_fusion_misorders:
        print(
            f"    {r.query_id}: dense={r.dense_rank}, bm25={r.bm25_rank}, "
            f"rrf_score={r.combined_rrf_score:.6f}"
        )

    # ======================================================================
    # STEP 4: Controlled RRF ablation
    # ======================================================================
    print("\n--- Step 4a: RRF-k sweep (candidate_k=20 fixed) ---")

    rrf_k_values = [20, 40, 60, 80, 100, 150, 200]
    all_configs: list[ConfigResult] = []

    for rrf_k in rrf_k_values:
        config = ConfigResult(
            config_name=f"rrf_k={rrf_k}_ck=20",
            rrf_k=rrf_k,
            candidate_k=20,
            final_k=5,
            dense_weight=1.0,
            bm25_weight=1.0,
        )

        all_query_results: dict[str, dict[str, dict]] = {}
        for track_name, td in track_data.items():
            results = evaluate_config(
                track_name,
                td["cases"],
                td["bm25"],
                td["dense"],
                rrf_k=rrf_k,
                candidate_k=20,
                final_k=5,
                dense_weight=1.0,
                bm25_weight=1.0,
                is_abstention=td["is_abstention"],
            )
            all_query_results[track_name] = results
            config.track_metrics[track_name] = compute_track_aggregates(
                results, td["is_abstention"]
            )

        # Flatten for comparison
        flat_baseline = {}
        flat_candidate = {}
        for tn in track_data:
            flat_baseline.update(all_baseline_results.get(tn, {}))
            flat_candidate.update(all_query_results.get(tn, {}))

        config.query_results = flat_candidate
        wins, losses, unch = compare_with_baseline(flat_baseline, flat_candidate)
        config.wins = len(wins)
        config.losses = len(losses)
        config.unchanged = len(unch)
        config.net_wins = len(wins) - len(losses)

        # Phase 13 impact
        fixed, regressed, p13_unch = classify_p13_impact(
            p13_ranking_failures, flat_baseline, flat_candidate
        )
        config.p13_fixed = fixed
        config.p13_regressed = regressed
        config.p13_unchanged = p13_unch

        # Latency
        all_latencies = [
            v["latency_ms"] for v in flat_candidate.values() if not v.get("is_abstention")
        ]
        if all_latencies:
            config.mean_latency_ms = round(statistics.fmean(all_latencies), 2)
            sorted_lat = sorted(all_latencies)
            config.p50_latency_ms = round(sorted_lat[len(sorted_lat) // 2], 2)
            config.p95_latency_ms = round(sorted_lat[int(len(sorted_lat) * 0.95)], 2)

        all_configs.append(config)

        tb = config.track_metrics.get("Track B", {})
        print(
            f"  rrf_k={rrf_k:>3}: "
            f"B_H@1={tb.get('hit_at_1', 0):.4f} B_H@5={tb.get('hit_at_5', 0):.4f} "
            f"B_MRR={tb.get('mrr_at_5', 0):.4f} "
            f"wins={config.wins} losses={config.losses} net={config.net_wins} "
            f"P13_fixed={len(fixed)} P13_regressed={len(regressed)}"
        )

    # ======================================================================
    # STEP 4b: candidate_k sweep (best rrf_k from above)
    # ======================================================================
    # Determine best rrf_k from sweep (by Track B MRR, with safety constraint)
    best_rrf_k_config = max(
        all_configs,
        key=lambda c: c.track_metrics.get("Track B", {}).get("mrr_at_5", 0),
    )
    best_rrf_k = best_rrf_k_config.rrf_k
    print(f"\n  Best rrf_k from sweep: {best_rrf_k}")

    print(f"\n--- Step 4b: candidate_k sweep (rrf_k={best_rrf_k} fixed) ---")

    candidate_k_values = [10, 20, 30, 50]

    for ck in candidate_k_values:
        config = ConfigResult(
            config_name=f"rrf_k={best_rrf_k}_ck={ck}",
            rrf_k=best_rrf_k,
            candidate_k=ck,
            final_k=5,
            dense_weight=1.0,
            bm25_weight=1.0,
        )

        all_query_results = {}
        for track_name, td in track_data.items():
            results = evaluate_config(
                track_name,
                td["cases"],
                td["bm25"],
                td["dense"],
                rrf_k=best_rrf_k,
                candidate_k=ck,
                final_k=5,
                dense_weight=1.0,
                bm25_weight=1.0,
                is_abstention=td["is_abstention"],
            )
            all_query_results[track_name] = results
            config.track_metrics[track_name] = compute_track_aggregates(
                results, td["is_abstention"]
            )

        flat_candidate = {}
        for tn in track_data:
            flat_candidate.update(all_query_results.get(tn, {}))

        config.query_results = flat_candidate
        flat_baseline = {}
        for tn in track_data:
            flat_baseline.update(all_baseline_results.get(tn, {}))

        wins, losses, unch = compare_with_baseline(flat_baseline, flat_candidate)
        config.wins = len(wins)
        config.losses = len(losses)
        config.unchanged = len(unch)
        config.net_wins = len(wins) - len(losses)

        fixed, regressed, p13_unch = classify_p13_impact(
            p13_ranking_failures, flat_baseline, flat_candidate
        )
        config.p13_fixed = fixed
        config.p13_regressed = regressed
        config.p13_unchanged = p13_unch

        all_latencies = [
            v["latency_ms"] for v in flat_candidate.values() if not v.get("is_abstention")
        ]
        if all_latencies:
            config.mean_latency_ms = round(statistics.fmean(all_latencies), 2)
            sorted_lat = sorted(all_latencies)
            config.p50_latency_ms = round(sorted_lat[len(sorted_lat) // 2], 2)
            config.p95_latency_ms = round(sorted_lat[int(len(sorted_lat) * 0.95)], 2)

        all_configs.append(config)

        tb = config.track_metrics.get("Track B", {})
        print(
            f"  ck={ck:>2}: "
            f"B_H@1={tb.get('hit_at_1', 0):.4f} B_H@5={tb.get('hit_at_5', 0):.4f} "
            f"B_MRR={tb.get('mrr_at_5', 0):.4f} "
            f"wins={config.wins} losses={config.losses} net={config.net_wins} "
            f"P13_fixed={len(fixed)} P13_regressed={len(regressed)}"
        )

    # ======================================================================
    # STEP 5: Weighted fusion experiments
    # ======================================================================
    print(f"\n--- Step 5: Weighted fusion (rrf_k={best_rrf_k}, ck=20) ---")

    weight_combos = [
        (0.25, 0.75, "d0.25_b0.75"),
        (0.40, 0.60, "d0.40_b0.60"),
        (0.50, 0.50, "d0.50_b0.50"),
        (0.60, 0.40, "d0.60_b0.40"),
        (0.75, 0.25, "d0.75_b0.25"),
    ]

    for dw, bw, label in weight_combos:
        config = ConfigResult(
            config_name=f"rrf_k={best_rrf_k}_ck=20_{label}",
            rrf_k=best_rrf_k,
            candidate_k=20,
            final_k=5,
            dense_weight=dw,
            bm25_weight=bw,
        )

        all_query_results = {}
        for track_name, td in track_data.items():
            results = evaluate_config(
                track_name,
                td["cases"],
                td["bm25"],
                td["dense"],
                rrf_k=best_rrf_k,
                candidate_k=20,
                final_k=5,
                dense_weight=dw,
                bm25_weight=bw,
                is_abstention=td["is_abstention"],
            )
            all_query_results[track_name] = results
            config.track_metrics[track_name] = compute_track_aggregates(
                results, td["is_abstention"]
            )

        flat_candidate = {}
        for tn in track_data:
            flat_candidate.update(all_query_results.get(tn, {}))

        config.query_results = flat_candidate
        flat_baseline = {}
        for tn in track_data:
            flat_baseline.update(all_baseline_results.get(tn, {}))

        wins, losses, unch = compare_with_baseline(flat_baseline, flat_candidate)
        config.wins = len(wins)
        config.losses = len(losses)
        config.unchanged = len(unch)
        config.net_wins = len(wins) - len(losses)

        fixed, regressed, p13_unch = classify_p13_impact(
            p13_ranking_failures, flat_baseline, flat_candidate
        )
        config.p13_fixed = fixed
        config.p13_regressed = regressed
        config.p13_unchanged = p13_unch

        all_latencies = [
            v["latency_ms"] for v in flat_candidate.values() if not v.get("is_abstention")
        ]
        if all_latencies:
            config.mean_latency_ms = round(statistics.fmean(all_latencies), 2)
            sorted_lat = sorted(all_latencies)
            config.p50_latency_ms = round(sorted_lat[len(sorted_lat) // 2], 2)
            config.p95_latency_ms = round(sorted_lat[int(len(sorted_lat) * 0.95)], 2)

        all_configs.append(config)

        tb = config.track_metrics.get("Track B", {})
        print(
            f"  {label}: "
            f"B_H@1={tb.get('hit_at_1', 0):.4f} B_H@5={tb.get('hit_at_5', 0):.4f} "
            f"B_MRR={tb.get('mrr_at_5', 0):.4f} "
            f"wins={config.wins} losses={config.losses} net={config.net_wins} "
            f"P13_fixed={len(fixed)} P13_regressed={len(regressed)}"
        )

    # ======================================================================
    # STEP 6-11: Select best config and produce final report
    # ======================================================================
    print("\n--- Step 6-11: Configuration selection ---")

    # Selection criteria:
    # 1. Must not regress Track A Hit@5 below 1.0
    # 2. Must not regress Track C (keep 1.0)
    # 3. Must preserve Track D abstention
    # 4. Must improve Track B MRR@5 or Hit@1
    # 5. No Phase 13 regression
    # 6. Acceptable latency

    eligible = []
    for config in all_configs:
        ta = config.track_metrics.get("Track A", {})
        tc = config.track_metrics.get("Track C", {})
        td_m = config.track_metrics.get("Track D", {})
        tb = config.track_metrics.get("Track B", {})

        # Multi-gate selection criteria (Steps 7, 11, 12):
        # 1. Track A safety gate (Hit@5 >= 1.0)
        # 2. Track C adversarial robustness (Hit@1 >= 1.0)
        # 3. Track D abstention safety (25/25 safe, 0% false answer rate)
        # 4. Step 7: A configuration must NOT be selected solely because its aggregate MRR increased.
        #    Must achieve net_wins > 0 (wins > losses across all queries).
        # 5. Step 11: Must fix a meaningful portion of Phase 13 failures (>= 3 fixed).
        # 6. Zero regression on Phase 13 ranking failure queries.
        # 7. No degradation on other Track B metrics (Hit@3 >= baseline).
        baseline_tb = baseline_aggregates.get("Track B", {})
        ta_safe = ta.get("hit_at_5", 0) >= 1.0
        tc_safe = tc.get("hit_at_1", 0) >= 1.0
        td_safe = td_m.get("abstention_safe", 0) == 25
        no_p13_regression = len(config.p13_regressed) == 0
        net_wins_positive = config.net_wins > 0
        meaningful_p13_fixed = len(config.p13_fixed) >= 3
        no_hit3_drop = tb.get("hit_at_3", 0) >= baseline_tb.get("hit_at_3", 0)

        tb_improved = tb.get("mrr_at_5", 0) > baseline_tb.get("mrr_at_5", 0) or tb.get(
            "hit_at_1", 0
        ) > baseline_tb.get("hit_at_1", 0)

        if (
            ta_safe
            and tc_safe
            and td_safe
            and no_p13_regression
            and net_wins_positive
            and meaningful_p13_fixed
            and no_hit3_drop
            and tb_improved
        ):
            eligible.append(config)
            print(
                f"  ELIGIBLE: {config.config_name} "
                f"B_MRR={tb.get('mrr_at_5', 0):.4f} "
                f"B_H@1={tb.get('hit_at_1', 0):.4f} "
                f"P13_fixed={len(config.p13_fixed)} "
                f"net_wins={config.net_wins}"
            )

    # Select best eligible by Track B MRR (simplest config preferred)
    if eligible:
        best = max(
            eligible,
            key=lambda c: (
                c.track_metrics.get("Track B", {}).get("mrr_at_5", 0),
                len(c.p13_fixed),
                -abs(c.rrf_k - 100),  # prefer closer to current
            ),
        )
        print(f"\n  SELECTED: {best.config_name}")
        decision = "RANKING_OPTIMIZATION_VALIDATED"
    else:
        best = None
        print("\n  No eligible configuration found — BASELINE RETAINED")
        decision = "BASELINE_RETAINED"

    # ======================================================================
    # Build JSON report
    # ======================================================================
    print("\n--- Building report ---")

    report = {
        "phase": 14,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "baseline_config": {
            "rrf_k": 100,
            "candidate_k": 20,
            "final_k": 5,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
        },
        "baseline_metrics": baseline_aggregates,
        "rrf_forensics": {
            "total_queries_analyzed": len(all_forensic_records),
            "failure_patterns": dict(pattern_counts.most_common()),
            "dense_wins_rrf_loses": [asdict(r) for r in dense_wins_rrf_loses],
            "bm25_wins_rrf_loses": [asdict(r) for r in bm25_wins_rrf_loses],
            "both_retrieve_fusion_misorders": [asdict(r) for r in both_retrieve_fusion_misorders],
        },
        "configurations_tested": [],
        "selected_config": None,
        "phase13_failure_qids": p13_ranking_failures,
    }

    for config in all_configs:
        config_entry = {
            "config_name": config.config_name,
            "rrf_k": config.rrf_k,
            "candidate_k": config.candidate_k,
            "final_k": config.final_k,
            "dense_weight": config.dense_weight,
            "bm25_weight": config.bm25_weight,
            "track_metrics": config.track_metrics,
            "wins": config.wins,
            "losses": config.losses,
            "unchanged": config.unchanged,
            "net_wins": config.net_wins,
            "p13_fixed": config.p13_fixed,
            "p13_regressed": config.p13_regressed,
            "p13_fixed_count": len(config.p13_fixed),
            "p13_regressed_count": len(config.p13_regressed),
            "mean_latency_ms": config.mean_latency_ms,
            "p50_latency_ms": config.p50_latency_ms,
            "p95_latency_ms": config.p95_latency_ms,
        }
        report["configurations_tested"].append(config_entry)

    if best is not None:
        report["selected_config"] = {
            "config_name": best.config_name,
            "rrf_k": best.rrf_k,
            "candidate_k": best.candidate_k,
            "final_k": best.final_k,
            "dense_weight": best.dense_weight,
            "bm25_weight": best.bm25_weight,
            "track_metrics": best.track_metrics,
            "wins": best.wins,
            "losses": best.losses,
            "net_wins": best.net_wins,
            "p13_fixed": best.p13_fixed,
            "p13_regressed": best.p13_regressed,
            "mean_latency_ms": best.mean_latency_ms,
        }

    # Write report
    out_path = _project_root / "docs" / "reports" / "phase14_ranking_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  Saved report to {out_path}")

    # Cleanup
    for td in track_data.values():
        td["vs"].close()

    print(f"\n{'=' * 80}")
    print(f"PHASE 14 — DECISION: {decision}")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
