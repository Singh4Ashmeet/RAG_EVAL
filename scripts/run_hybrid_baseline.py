"""CLI script to run hybrid retrieval evaluation."""

from __future__ import annotations

import argparse
import sys
import time
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
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack
from eval.metrics.retrieval import (
    average_precision_at_k,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    source_element_hit_rate_at_k,
    source_element_recall_at_k,
    source_element_reciprocal_rank_at_k,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval evaluation for Track A, B, or C."
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["track_a", "track_b", "track_c"],
        help="Golden evaluation track to run.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        required=True,
        choices=["weighted", "rrf"],
        help="Fusion method: weighted or rrf.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Dense weight for weighted fusion (default: 0.5).",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
        help="Sparse weight for weighted fusion (default: 0.5).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k parameter (default: 60).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size from each retriever (default: 20).",
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
    return parser.parse_args(args)


def run_hybrid_ablation(
    track: CorpusTrack,
    fusion: str,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
    candidate_k: int,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    ollama_host: str,
    cutoffs: tuple[int, ...],
    timeout: float,
) -> dict:
    """Run hybrid retrieval-only ablation for a single track/config."""
    dataset_path = f"eval/golden/{track.value}.jsonl"
    corpus_dir = f"eval/corpora/{track.value}"

    print(
        f"\n=== Hybrid Ablation: {track.value.upper()} | fusion={fusion} candidate_k={candidate_k} ==="
    )
    if fusion == "weighted":
        print(f"  Weights: dense={dense_weight}, sparse={sparse_weight}")
    else:
        print(f"  RRF k={rrf_k}")
    print(f"Dataset: {dataset_path}")
    print(f"Corpus:  {corpus_dir}")

    # Load dataset
    dataset = load_jsonl_dataset(dataset_path)

    # Ingest corpus and build both indexes
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

    # Build dense index
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name

    embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
    vector_store = SQLiteVectorStore(db_path)
    dense_retriever = DenseRetriever(embedder, vector_store)

    for source_file in source_files:
        result = pipeline.ingest(source_file)
        if result.document is not None and result.chunks:
            dense_retriever.index_document(result.document.metadata.document_id, result.chunks)

    # Build BM25 index
    bm25_retriever = BM25Retriever(all_chunks)

    # Build hybrid retriever
    if fusion == "weighted":
        hybrid = HybridRetriever(
            dense_retriever,
            bm25_retriever,
            fusion_method="weighted",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            candidate_k=candidate_k,
        )
    else:
        hybrid = HybridRetriever(
            dense_retriever,
            bm25_retriever,
            fusion_method="rrf",
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )

    stats = hybrid.get_stats()
    print(f"Hybrid stats: {stats}")

    # Evaluate retrieval
    max_cutoff = max(cutoffs)
    retrieval_metrics: dict[str, list[float]] = {}
    latencies: list[float] = []

    for case in dataset.cases:
        start = time.perf_counter()
        retrieved_sources = hybrid.retrieve(case.query, top_k=max_cutoff)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r.chunk.chunk_id for r in retrieved_sources]
        retrieved_chunks = [r.chunk for r in retrieved_sources]

        for cutoff in cutoffs:
            if case.relevant_chunk_ids:
                retrieval_metrics.setdefault(f"recall@{cutoff}", []).append(
                    recall_at_k(retrieved_ids, case.relevant_chunk_ids, cutoff)
                )
                retrieval_metrics.setdefault(f"precision@{cutoff}", []).append(
                    precision_at_k(retrieved_ids, case.relevant_chunk_ids, cutoff)
                )
                retrieval_metrics.setdefault(f"mrr@{cutoff}", []).append(
                    reciprocal_rank_at_k(retrieved_ids, case.relevant_chunk_ids, cutoff)
                )
                retrieval_metrics.setdefault(f"map@{cutoff}", []).append(
                    average_precision_at_k(retrieved_ids, case.relevant_chunk_ids, cutoff)
                )
                retrieval_metrics.setdefault(f"hit_rate@{cutoff}", []).append(
                    hit_rate_at_k(retrieved_ids, case.relevant_chunk_ids, cutoff)
                )

            if case.relevant_element_ids:
                retrieval_metrics.setdefault(f"source_element_recall@{cutoff}", []).append(
                    source_element_recall_at_k(retrieved_chunks, case.relevant_element_ids, cutoff)
                )
                retrieval_metrics.setdefault(f"source_element_mrr@{cutoff}", []).append(
                    source_element_reciprocal_rank_at_k(
                        retrieved_chunks, case.relevant_element_ids, cutoff
                    )
                )
                retrieval_metrics.setdefault(f"source_element_hit_rate@{cutoff}", []).append(
                    source_element_hit_rate_at_k(
                        retrieved_chunks, case.relevant_element_ids, cutoff
                    )
                )

    # Aggregate
    from statistics import fmean

    agg = {}
    for name, values in retrieval_metrics.items():
        agg[name] = fmean(values) if values else 0.0
    agg["latency.retrieval_ms"] = fmean(latencies) if latencies else 0.0

    # Chunk stats
    chunk_sizes = [len(c.text) for c in all_chunks]
    agg["total_chunks"] = len(all_chunks)
    agg["mean_chunk_size"] = fmean(chunk_sizes) if chunk_sizes else 0
    agg["median_chunk_size"] = sorted(chunk_sizes)[len(chunk_sizes) // 2] if chunk_sizes else 0
    agg["min_chunk_size"] = min(chunk_sizes) if chunk_sizes else 0
    agg["max_chunk_size"] = max(chunk_sizes) if chunk_sizes else 0

    print("Results:")
    for cutoff in cutoffs:
        print(f"  Hit@{cutoff}:     {agg.get(f'hit_rate@{cutoff}', 0):.4f}")
        print(f"  Recall@{cutoff}: {agg.get(f'recall@{cutoff}', 0):.4f}")
        print(f"  MRR@{cutoff}:    {agg.get(f'mrr@{cutoff}', 0):.4f}")
        print(f"  MAP@{cutoff}:    {agg.get(f'map@{cutoff}', 0):.4f}")
    if any(k.startswith("source_element") for k in agg):
        print(f"  SourceElem Recall@5: {agg.get('source_element_recall@5', 0):.4f}")
        print(f"  SourceElem Hit@5:    {agg.get('source_element_hit_rate@5', 0):.4f}")
    print(f"  Latency:       {agg['latency.retrieval_ms']:.1f} ms")
    print(f"  Total chunks:  {agg['total_chunks']}")

    return agg


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)
    cutoffs = tuple(parsed.cutoffs)

    run_hybrid_ablation(
        track=track,
        fusion=parsed.fusion,
        dense_weight=parsed.dense_weight,
        sparse_weight=parsed.sparse_weight,
        rrf_k=parsed.rrf_k,
        candidate_k=parsed.candidate_k,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        embed_model=parsed.embed_model,
        ollama_host=parsed.ollama_host,
        cutoffs=cutoffs,
        timeout=parsed.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
