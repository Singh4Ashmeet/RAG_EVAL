"""Query-level analysis comparing dense vs hybrid retrieval."""

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
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query-level analysis for Track B.")
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
        default="rrf",
        choices=["dense", "bm25", "rrf", "weighted"],
        help="Retrieval method to evaluate.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )
    return parser.parse_args(args)


def build_retriever(
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
    timeout: float,
):
    """Build the specified retriever."""
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

    if fusion == "dense":
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
            db_path = f.name
        embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
        vector_store = SQLiteVectorStore(db_path)
        retriever = DenseRetriever(embedder, vector_store)
        for source_file in source_files:
            result = pipeline.ingest(source_file)
            if result.document is not None and result.chunks:
                retriever.index_document(result.document.metadata.document_id, result.chunks)
        return retriever

    elif fusion == "bm25":
        return BM25Retriever(all_chunks)

    elif fusion == "rrf":
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
            db_path = f.name
        embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
        vector_store = SQLiteVectorStore(db_path)
        dense_retriever = DenseRetriever(embedder, vector_store)
        for source_file in source_files:
            result = pipeline.ingest(source_file)
            if result.document is not None and result.chunks:
                dense_retriever.index_document(result.document.metadata.document_id, result.chunks)
        bm25_retriever = BM25Retriever(all_chunks)
        return HybridRetriever(
            dense_retriever,
            bm25_retriever,
            fusion_method="rrf",
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )

    elif fusion == "weighted":
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
            db_path = f.name
        embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
        vector_store = SQLiteVectorStore(db_path)
        dense_retriever = DenseRetriever(embedder, vector_store)
        for source_file in source_files:
            result = pipeline.ingest(source_file)
            if result.document is not None and result.chunks:
                dense_retriever.index_document(result.document.metadata.document_id, result.chunks)
        bm25_retriever = BM25Retriever(all_chunks)
        return HybridRetriever(
            dense_retriever,
            bm25_retriever,
            fusion_method="weighted",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            candidate_k=candidate_k,
        )


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)
    cutoffs = tuple(parsed.cutoffs)

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    # Build both retrievers
    dense_retriever = build_retriever(
        track,
        "dense",
        0.5,
        0.5,
        60,
        20,
        parsed.chunk_size,
        parsed.chunk_overlap,
        parsed.embed_model,
        parsed.ollama_host,
        parsed.timeout,
    )

    hybrid_retriever = build_retriever(
        track,
        parsed.fusion,
        parsed.dense_weight,
        parsed.sparse_weight,
        parsed.rrf_k,
        parsed.candidate_k,
        parsed.chunk_size,
        parsed.chunk_overlap,
        parsed.embed_model,
        parsed.ollama_host,
        parsed.timeout,
    )

    max_cutoff = max(cutoffs)

    print(
        f"\n=== Query-Level Analysis: {track.value.upper()} | Dense vs {parsed.fusion.upper()} ==="
    )
    print(
        f"{'Query ID':<20} {'Dense Rank':<12} {'Hybrid Rank':<12} {'Change':<8} {'Dense Hit@5':<12} {'Hybrid Hit@5':<12} {'Relevant Chunks'}"
    )
    print("-" * 120)

    dense_better = 0
    hybrid_better = 0
    same = 0

    for case in dataset.cases:
        dense_results = dense_retriever.retrieve(case.query, top_k=max_cutoff)
        hybrid_results = hybrid_retriever.retrieve(case.query, top_k=max_cutoff)

        dense_ids = [r.chunk.chunk_id for r in dense_results]
        hybrid_ids = [r.chunk.chunk_id for r in hybrid_results]

        # Find rank of first relevant chunk
        dense_rank = None
        hybrid_rank = None

        for cid in case.relevant_chunk_ids:
            if cid in dense_ids:
                dense_rank = dense_ids.index(cid) + 1
                break
        for cid in case.relevant_chunk_ids:
            if cid in hybrid_ids:
                hybrid_rank = hybrid_ids.index(cid) + 1
                break

        dense_hit5 = 1 if dense_rank and dense_rank <= 5 else 0
        hybrid_hit5 = 1 if hybrid_rank and hybrid_rank <= 5 else 0

        if dense_rank and hybrid_rank:
            change = hybrid_rank - dense_rank
            change_str = f"{change:+d}"
        elif hybrid_rank and not dense_rank:
            change_str = "NEW"
            hybrid_better += 1
        elif dense_rank and not hybrid_rank:
            change_str = "LOST"
            dense_better += 1
        else:
            change_str = "N/A"
            same += 1

        if dense_hit5 and not hybrid_hit5:
            dense_better += 1
        elif hybrid_hit5 and not dense_hit5:
            hybrid_better += 1
        else:
            same += 1

        dense_rank_str = str(dense_rank) if dense_rank else "miss"
        hybrid_rank_str = str(hybrid_rank) if hybrid_rank else "miss"

        print(
            f"{case.query_id:<20} {dense_rank_str:<12} {hybrid_rank_str:<12} {change_str:<8} {dense_hit5:<12} {hybrid_hit5:<12} {case.relevant_chunk_ids}"
        )

    print("-" * 120)
    print(f"Summary: Dense better: {dense_better}, Hybrid better: {hybrid_better}, Same: {same}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
