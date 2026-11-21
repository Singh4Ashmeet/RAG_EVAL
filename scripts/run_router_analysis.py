"""Query-level analysis for router decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tempfile

from core.embed.ollama_embedder import OllamaEmbedder
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.router import QueryRouter, RoutingStrategy
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run router query analysis.")
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["track_a", "track_b", "track_c"],
        help="Golden evaluation track to run.",
    )
    parser.add_argument(
        "--router",
        type=str,
        default="heuristic",
        choices=["heuristic", "oracle"],
        help="Router type to evaluate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200).",
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
        help="Ollama host URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Retrieval evaluation cutoffs (default: 1 3 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Ollama requests (default: 120.0).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size for hybrid fusion (default: 20).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=100,
        help="RRF k parameter (default: 100).",
    )
    return parser.parse_args(args)


def build_retrievers(
    track: CorpusTrack,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    ollama_host: str,
    timeout: float,
    candidate_k: int,
    rrf_k: int,
):
    """Build the retrievers."""

    corpus_dir = f"eval/corpora/{track.value}"

    pipeline = create_default_ingestion_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []
    supported_exts = pipeline.parser_registry.supported_extensions()
    source_files = sorted(
        p
        for p in Path(corpus_dir).iterdir()
        if p.is_file()
        and p.suffix.lower() in supported_exts
        and p.name not in {"manifest.json", "PROVENANCE.json"}
    )

    for source_file in source_files:
        result = pipeline.ingest(source_file)
        if result.document is not None and result.chunks:
            all_chunks.extend(result.chunks)

    print(f"Indexed {len(all_chunks)} chunks from {len(source_files)} documents")

    # Build dense retriever
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name
    embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
    vector_store = SQLiteVectorStore(db_path)
    dense_retriever = DenseRetriever(embedder, vector_store)
    for source_file in source_files:
        result = pipeline.ingest(source_file)
        if result.document is not None and result.chunks:
            dense_retriever.index_document(result.document.metadata.document_id, result.chunks)

    # Build BM25 retriever
    bm25_retriever = BM25Retriever(all_chunks)

    # Build hybrid retriever (baseline)
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        fusion_method="rrf",
        rrf_k=100,
        candidate_k=20,
    )

    return dense_retriever, bm25_retriever, hybrid_retriever


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)
    cutoffs = tuple(parsed.cutoffs)

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    print(f"\n=== Router Analysis: {track.value.upper()} | Router={parsed.router} ===")

    # Build retrievers
    print("Building retrievers...")
    dense_retriever, bm25_retriever, hybrid_retriever = build_retrievers(
        track=track,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        embed_model=parsed.embed_model,
        ollama_host=parsed.ollama_host,
        timeout=parsed.timeout,
        candidate_k=parsed.candidate_k,
        rrf_k=parsed.rrf_k,
    )

    # Build router
    router = QueryRouter(confidence_threshold=0.7)

    max(cutoffs)

    print(f"\n=== Router Analysis: {track.value.upper()} | Router={parsed.router} ===")
    print(
        f"{'Query ID':<20} {'Route':<10} {'Conf':<6} {'Baseline@5':<12} {'Router@5':<12} {'Delta':<8} {'Status':<10}"
    )
    print("-" * 100)

    router_wins = 0
    baseline_wins = 0
    same = 0
    both_fail = 0
    route_counts = {"bm25": 0, "dense": 0, "hybrid": 0, "fallback": 0}

    for case in dataset.cases:
        # Baseline (Dense + BM25 RRF)
        baseline_results = hybrid_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))
        baseline_ids = [r.chunk.chunk_id for r in baseline_results]
        baseline_hit5 = any(cid in baseline_ids for cid in case.relevant_chunk_ids)

        # Router decision
        if parsed.router == "oracle":
            # Oracle: pick the best strategy per query
            # Evaluate all three strategies and pick the best
            dense_results = dense_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))
            bm25_results = bm25_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))
            hybrid_results = hybrid_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))

            dense_ids = [r.chunk.chunk_id for r in dense_results]
            bm25_ids = [r.chunk.chunk_id for r in bm25_results]
            hybrid_ids = [r.chunk.chunk_id for r in hybrid_results]

            dense_hit5 = any(cid in dense_ids for cid in case.relevant_chunk_ids)
            bm25_hit5 = any(cid in bm25_ids for cid in case.relevant_chunk_ids)
            hybrid_hit5 = any(cid in hybrid_ids for cid in case.relevant_chunk_ids)

            # Pick the best
            if dense_hit5:
                decision = type(
                    "obj", (object,), {"strategy": RoutingStrategy.DENSE, "confidence": 1.0}
                )()
            elif bm25_hit5:
                decision = type(
                    "obj", (object,), {"strategy": RoutingStrategy.BM25, "confidence": 1.0}
                )()
            else:
                decision = type(
                    "obj", (object,), {"strategy": RoutingStrategy.HYBRID, "confidence": 1.0}
                )()
            router_hit5 = dense_hit5 or bm25_hit5 or hybrid_hit5
        else:
            # Heuristic router
            router = QueryRouter(confidence_threshold=0.7)
            decision = router.route(case.query)
            route_counts[decision.strategy.value] += 1

            # Get results from selected strategy
            if decision.strategy == RoutingStrategy.BM25:
                router_results = bm25_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))
            elif decision.strategy == RoutingStrategy.DENSE:
                router_results = dense_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))
            else:  # HYBRID
                router_results = hybrid_retriever.retrieve(case.query, top_k=max(parsed.cutoffs))

            router_ids = [r.chunk.chunk_id for r in router_results]
            router_hit5 = any(cid in router_ids for cid in case.relevant_chunk_ids)

        baseline_hit5 = 1 if baseline_hit5 else 0
        router_hit5 = 1 if router_hit5 else 0

        if router_hit5 and not baseline_hit5:
            router_wins += 1
            status = "WIN"
        elif baseline_hit5 and not router_hit5:
            baseline_wins += 1
            status = "LOSS"
        elif baseline_hit5 and router_hit5:
            same += 1
            status = "SAME"
        else:
            both_fail += 1
            status = "BOTH_FAIL"

        print(
            f"{case.query_id:<20} {decision.strategy.value:<10} {decision.confidence:<6.2f} "
            f"{baseline_hit5:<12} {router_hit5:<12} {router_hit5 - baseline_hit5:<+8} {status:<10}"
        )

    print("-" * 100)
    print(
        f"Summary: Router wins: {router_wins}, Baseline wins: {baseline_wins}, Same: {same}, Both fail: {both_fail}"
    )
    print(f"Route distribution: {route_counts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
