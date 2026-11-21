"""Complementarity analysis for HyDE vs Dense vs BM25."""

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
from core.hyde.retriever import HyDERetriever
from core.hyde.transformer import HyDEConfig, HyDETransformer
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complementarity analysis for HyDE.")
    parser.add_argument(
        "--track",
        type=str,
        default="track_b",
        choices=["track_a", "track_b", "track_c"],
        help="Golden evaluation track to run.",
    )
    parser.add_argument(
        "--hyde-model",
        type=str,
        default="llama3.2:3b",
    )
    parser.add_argument(
        "--hyde-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--hyde-max-tokens",
        type=int,
        default=512,
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
        "--timeout",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    return parser.parse_args(args)


def build_retrievers(
    track: CorpusTrack,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    ollama_host: str,
    timeout: float,
    hyde_model: str,
):
    """Build all retrievers: Dense, BM25, HyDE."""

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
    BM25Retriever(all_chunks)

    # Build HyDE retriever (uses dense_retriever internally)
    HyDEConfig()
    HyDETransformer(HyDEConfig())
    HyDERetriever(dense_retriever, HyDETransformer(HyDEConfig()), use_original_query=True)

    return (
        dense_retriever,
        BM25Retriever(all_chunks),
        HyDERetriever(dense_retriever, HyDETransformer(HyDEConfig()), use_original_query=True),
        all_chunks,
    )


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    print(f"\n=== Complementarity Analysis: {track.value.upper()} (Dense vs BM25 vs HyDE) ===")

    # Build all retrievers
    dense_retriever, bm25_retriever, hyde_retriever, all_chunks = build_retrievers(
        track=track,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        embed_model=parsed.embed_model,
        ollama_host=parsed.ollama_host,
        timeout=parsed.timeout,
        hyde_model=parsed.hyde_model,
    )

    print("\n=== Complementarity Analysis (Top-5) ===")
    print(f"{'Query ID':<20} {'Dense':<8} {'BM25':<8} {'HyDE':<8} {'Category':<10}")
    print("-" * 100)

    dense_only = 0
    bm25_only = 0
    hyde_only = 0
    dense_bm25 = 0
    dense_hyde = 0
    bm25_hyde = 0
    all_three = 0
    none = 0

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    for case in dataset.cases:
        dense_results = dense_retriever.retrieve(case.query, top_k=5)
        bm25_results = bm25_retriever.retrieve(case.query, top_k=5)
        hyde_results = hyde_retriever.retrieve(case.query, top_k=5)

        dense_ids = set(r.chunk.chunk_id for r in dense_results)
        bm25_ids = set(r.chunk.chunk_id for r in bm25_results)
        hyde_ids = set(r.chunk.chunk_id for r in hyde_results)

        dense_hit = any(cid in dense_ids for cid in case.relevant_chunk_ids)
        bm25_hit = any(cid in bm25_ids for cid in case.relevant_chunk_ids)
        hyde_hit = any(cid in hyde_ids for cid in case.relevant_chunk_ids)

        if dense_hit and bm25_hit and hyde_hit:
            all_three += 1
            category = "ALL"
        elif dense_hit and bm25_hit:
            dense_bm25 += 1
            category = "D+B"
        elif dense_hit and hyde_hit:
            dense_hyde += 1
            category = "D+H"
        elif bm25_hit and hyde_hit:
            bm25_hyde += 1
            category = "B+H"
        elif dense_hit:
            dense_only += 1
            category = "D"
        elif bm25_hit:
            bm25_only += 1
            category = "B"
        elif hyde_hit:
            hyde_only += 1
            category = "H"
        else:
            none += 1
            category = "NONE"

        dense_rank = next(
            (i + 1 for i, cid in enumerate(dense_ids) if cid in case.relevant_chunk_ids), "miss"
        )
        bm25_rank = next(
            (i + 1 for i, cid in enumerate(bm25_ids) if cid in case.relevant_chunk_ids), "miss"
        )
        hyde_rank = next(
            (i + 1 for i, cid in enumerate(hyde_ids) if cid in case.relevant_chunk_ids), "miss"
        )

        print(
            f"{case.query_id:<20} {str(dense_rank):<8} {str(bm25_rank):<8} {str(hyde_rank):<8} {category} | {case.relevant_chunk_ids}"
        )

    total = len(dataset.cases)
    print("-" * 100)
    print(f"Total queries: {total}")
    print(f"All three: {all_three}")
    print(f"Dense+BM25 only: {dense_bm25}")
    print(f"Dense+HyDE only: {dense_hyde}")
    print(f"BM25+HyDE only: {bm25_hyde}")
    print(f"Dense only: {dense_only}")
    print(f"BM25 only: {bm25_only}")
    print(f"HyDE only: {hyde_only}")
    print(f"None: {none}")

    print(f"\nHyDE recovered {hyde_only} queries that neither Dense nor BM25 found")
    print(f"HyDE + BM25 recovered {bm25_hyde} queries that Dense missed")
    print(f"HyDE + Dense recovered {dense_hyde} queries that BM25 missed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
