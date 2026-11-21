"""Comprehensive evaluation of learned retrieval router vs baseline across test and protected tracks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.embed.ollama_embedder import OllamaEmbedder
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from core.retrieval.base import SearchResult
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.router.learned import LearnedQueryRouter, LearnedRouteDecision, RoutingStrategy
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import EvaluationCase
from eval.metrics.retrieval import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k
from eval.routing.oracle import CachedEmbedder


@dataclass
class RouteEvalResult:
    query_id: str
    query: str
    decision: LearnedRouteDecision
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float
    recall_at_5: float
    latency_ms: float
    routing_latency_ms: float


@dataclass
class TrackSummary:
    name: str
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float
    recall_at_5: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    route_distribution: dict[str, int]
    fallback_rate: float
    wins_vs_baseline: int
    losses_vs_baseline: int
    unchanged_vs_baseline: int


class RoutedRetrieverWrapper:
    """Dispatches retrieval requests according to a LearnedQueryRouter."""

    def __init__(
        self,
        router: LearnedQueryRouter,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        hybrid: HybridRetriever,
        confidence_threshold: float | None = 0.70,
        enable_fallback: bool = True,
    ) -> None:
        self.router = router
        self.bm25 = bm25
        self.dense = dense
        self.hybrid = hybrid
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback

    def retrieve_with_diagnostics(
        self, query: str, top_k: int = 5
    ) -> tuple[list[SearchResult], LearnedRouteDecision, float, float]:
        t0 = time.perf_counter()
        if self.enable_fallback:
            dec = self.router.route(query, threshold=self.confidence_threshold)
        else:
            dec = self.router.route(query, threshold=0.0)
        t_route = time.perf_counter()

        routing_ms = (t_route - t0) * 1000.0

        if dec.strategy == RoutingStrategy.BM25:
            results = self.bm25.retrieve(query, top_k=top_k)
        elif dec.strategy == RoutingStrategy.DENSE:
            results = self.dense.retrieve(query, top_k=top_k)
        else:
            results = self.hybrid.retrieve(query, top_k=top_k)

        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000.0
        return results, dec, total_ms, routing_ms


def build_track_retrievers(
    doc_paths: Sequence[Path],
    embedder: CachedEmbedder,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[BM25Retriever, DenseRetriever, HybridRetriever, SQLiteVectorStore]:
    import tempfile

    pipeline = create_default_ingestion_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []

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

    bm25 = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=100, candidate_k=20)
    return bm25, dense, hybrid, vector_store


def evaluate_dataset(
    cases: Sequence[EvaluationCase],
    retriever_fn,
    top_k: int = 5,
) -> list[RouteEvalResult]:
    results: list[RouteEvalResult] = []
    for case in cases:
        ret_results, dec, total_ms, route_ms = retriever_fn(case.query, top_k=top_k)
        ret_ids = [r.chunk.chunk_id for r in ret_results]
        rel_ids = case.relevant_chunk_ids

        if not rel_ids:
            # Unanswerable / abstention test cases (e.g. Track D)
            h1 = 0.0
            h3 = 0.0
            h5 = 0.0
            mrr = 0.0
            rec = 0.0
        else:
            h1 = hit_rate_at_k(ret_ids, rel_ids, k=1)
            h3 = hit_rate_at_k(ret_ids, rel_ids, k=3)
            h5 = hit_rate_at_k(ret_ids, rel_ids, k=5)
            mrr = reciprocal_rank_at_k(ret_ids, rel_ids, k=5)
            rec = recall_at_k(ret_ids, rel_ids, k=5)

        results.append(
            RouteEvalResult(
                query_id=case.query_id,
                query=case.query,
                decision=dec,
                retrieved_chunk_ids=ret_ids,
                relevant_chunk_ids=rel_ids,
                hit_at_1=h1,
                hit_at_3=h3,
                hit_at_5=h5,
                mrr_at_5=mrr,
                recall_at_5=rec,
                latency_ms=total_ms,
                routing_latency_ms=route_ms,
            )
        )
    return results


def summarize_results(
    name: str,
    results: list[RouteEvalResult],
    baseline_results: list[RouteEvalResult] | None = None,
) -> TrackSummary:
    n = len(results)
    if n == 0:
        raise ValueError("Cannot summarize empty results.")

    h1 = statistics.fmean(r.hit_at_1 for r in results)
    h3 = statistics.fmean(r.hit_at_3 for r in results)
    h5 = statistics.fmean(r.hit_at_5 for r in results)
    mrr = statistics.fmean(r.mrr_at_5 for r in results)
    rec = statistics.fmean(r.recall_at_5 for r in results)

    lats = [r.latency_ms for r in results]
    mean_lat = statistics.fmean(lats)
    median_lat = statistics.median(lats)
    sorted_lats = sorted(lats)
    p95_lat = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else mean_lat

    route_counts = Counter(r.decision.strategy.value for r in results)
    fallbacks = sum(1 for r in results if r.decision.fallback)
    fb_rate = fallbacks / n

    wins, losses, unchanged = 0, 0, 0
    if baseline_results:
        for r, base in zip(results, baseline_results, strict=False):
            if r.mrr_at_5 > base.mrr_at_5 or (
                r.mrr_at_5 == base.mrr_at_5 and r.hit_at_1 > base.hit_at_1
            ):
                wins += 1
            elif r.mrr_at_5 < base.mrr_at_5 or (
                r.mrr_at_5 == base.mrr_at_5 and r.hit_at_1 < base.hit_at_1
            ):
                losses += 1
            else:
                unchanged += 1

    return TrackSummary(
        name=name,
        total_queries=n,
        hit_at_1=round(h1, 4),
        hit_at_3=round(h3, 4),
        hit_at_5=round(h5, 4),
        mrr_at_5=round(mrr, 4),
        recall_at_5=round(rec, 4),
        mean_latency_ms=round(mean_lat, 2),
        median_latency_ms=round(median_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        route_distribution=dict(route_counts),
        fallback_rate=round(fb_rate, 4),
        wins_vs_baseline=wins,
        losses_vs_baseline=losses,
        unchanged_vs_baseline=unchanged,
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned router vs production baseline.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/routing/phase12"),
        help="Directory containing trained model artifact (default: models/routing/phase12).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for gated routing (default: 0.70).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host endpoint URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to output evaluation report JSON.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    model_dir = parsed.model_dir

    print(f"\n{'=' * 80}")
    print("EVALUATING LEARNED QUERY ROUTER (PHASE 12)")
    print(f"  Model Directory       : {model_dir}")
    print(f"  Confidence Threshold : {parsed.confidence_threshold}")
    print(f"{'=' * 80}\n")

    router = LearnedQueryRouter.load(model_dir)
    raw_embedder = OllamaEmbedder(model=parsed.embed_model, host=parsed.ollama_host)
    embedder = CachedEmbedder(raw_embedder)

    eval_targets = [
        # Target 1: Held-out Phase 11 Test partition
        (
            "Phase 11 Test (Held-Out)",
            "data/routing/phase11/test.jsonl",
            [
                Path("README.md"),
                Path("eval/corpora/track_c/system_spec.md"),
                Path("docs/ci-cd.md"),
                Path("docs/privacy.md"),
            ],
        ),
        # Target 2: Protected Track A
        (
            "Track A (Historical Control)",
            "eval/golden/track_a.jsonl",
            [
                Path("eval/corpora/track_a/architecture.md"),
                Path("eval/corpora/track_a/hosted-beta.md"),
                Path("eval/corpora/track_a/runbook.md"),
            ],
        ),
        # Target 3: Protected Track B
        (
            "Track B (Primary Benchmark)",
            "eval/golden/track_b.jsonl",
            [
                Path("eval/corpora/track_b/rfc8259.txt"),
            ],
        ),
        # Target 4: Protected Track C
        (
            "Track C (Adversarial/Hard)",
            "eval/golden/track_c.jsonl",
            [
                Path("eval/corpora/track_c/incident_log.txt"),
                Path("eval/corpora/track_c/operations_guide.pdf"),
                Path("eval/corpora/track_c/product_evaluation.docx"),
                Path("eval/corpora/track_c/system_spec.md"),
            ],
        ),
        # Target 5: Protected Track D
        (
            "Track D (Abstention Safety)",
            "eval/golden/track_d.jsonl",
            [
                Path("eval/corpora/track_d/reference_guide.md"),
            ],
        ),
    ]

    report_data: dict[str, Any] = {}

    for target_name, dataset_file, doc_paths in eval_targets:
        ds_path = _project_root / dataset_file
        if not ds_path.exists():
            print(f"Skipping {target_name}: file {ds_path} not found.")
            continue

        print(f"\nEvaluating: {target_name} ({ds_path})")
        # Load cases
        if "track_" in dataset_file:
            dataset = load_jsonl_dataset(str(ds_path))
            cases = dataset.cases
        else:
            # Phase 11 test partition
            cases = []
            with open(ds_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    cases.append(
                        EvaluationCase(
                            query_id=rec["example_id"],
                            query=rec["query"],
                            relevant_chunk_ids=rec["source_chunk_ids"],
                            relevant_element_ids=rec["source_element_ids"],
                            reference_answer=rec.get("expected_answer"),
                        )
                    )

        print(f"  Loaded {len(cases)} queries. Indexing documents...")
        bm25, dense, hybrid, vector_store = build_track_retrievers(doc_paths, embedder)

        # Pre-warm query cache
        embedder.warm_cache([c.query for c in cases])

        # Define retriever functions
        def base_fn(q: str, top_k: int = 5, active_hybrid: HybridRetriever = hybrid):
            t0 = time.perf_counter()
            res = active_hybrid.retrieve(q, top_k=top_k)
            t1 = time.perf_counter()
            dec = LearnedRouteDecision(
                strategy=RoutingStrategy.HYBRID,
                confidence=1.0,
                raw_prediction=RoutingStrategy.HYBRID,
                fallback=False,
                probabilities={"hybrid": 1.0},
                model_version="baseline",
                reason="baseline_hybrid",
            )
            return res, dec, (t1 - t0) * 1000.0, 0.0

        raw_wrapper = RoutedRetrieverWrapper(
            router=router,
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            confidence_threshold=0.0,
            enable_fallback=False,
        )

        gated_wrapper = RoutedRetrieverWrapper(
            router=router,
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            confidence_threshold=parsed.confidence_threshold,
            enable_fallback=True,
        )

        # 1. Run Baseline
        base_res = evaluate_dataset(cases, base_fn, top_k=5)
        base_sum = summarize_results(f"{target_name} - Baseline", base_res)

        # 2. Run Raw Learned Router
        raw_res = evaluate_dataset(cases, raw_wrapper.retrieve_with_diagnostics, top_k=5)
        raw_sum = summarize_results(
            f"{target_name} - Learned (Raw)", raw_res, baseline_results=base_res
        )

        # 3. Run Gated Learned Router
        gated_res = evaluate_dataset(cases, gated_wrapper.retrieve_with_diagnostics, top_k=5)
        gated_sum = summarize_results(
            f"{target_name} - Gated (Thresh={parsed.confidence_threshold})",
            gated_res,
            baseline_results=base_res,
        )

        # Print comparison table
        print(
            f"\n  {'Strategy':<35} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8} {'MRR@5':<8} {'Latency':<10} {'Wins/Loss/Same':<15}"
        )
        print("  " + "-" * 95)
        for s in [base_sum, raw_sum, gated_sum]:
            wl = f"{s.wins_vs_baseline}/{s.losses_vs_baseline}/{s.unchanged_vs_baseline}"
            print(
                f"  {s.name:<35} {s.hit_at_1:<8.4f} {s.hit_at_3:<8.4f} {s.hit_at_5:<8.4f} {s.mrr_at_5:<8.4f} {s.mean_latency_ms:<10.2f} {wl:<15}"
            )

        print(
            f"  Gated Route Distribution: {gated_sum.route_distribution} (Fallback Rate: {gated_sum.fallback_rate:.1%})"
        )

        report_data[target_name] = {
            "baseline": base_sum.__dict__,
            "learned_raw": raw_sum.__dict__,
            "learned_gated": gated_sum.__dict__,
        }

        # Cleanup vector store
        vector_store.close()

    out_path = parsed.output_report or (
        _project_root / "docs" / "reports" / "phase12_evaluation_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to {out_path}")

    print(f"\n{'=' * 80}")
    print("PHASE 12 COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
