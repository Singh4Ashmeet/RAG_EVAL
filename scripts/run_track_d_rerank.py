"""Run Track D with reranked retriever for safety validation."""

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
from core.generate.pipeline import GroundedAnswerGenerator
from core.ingest import create_default_ingestion_pipeline
from core.llm.base import GenerationSettings
from core.llm.ollama_provider import OllamaLLMProvider
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.rerank import CrossEncoderReranker, RerankingRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack
from eval.metrics.abstention import is_abstention


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track D with reranked retriever.")
    parser.add_argument(
        "--track",
        type=str,
        default="track_d",
        choices=["track_d"],
        help="Track to run (default: track_d).",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="mini",
        choices=["mini", "tiny", "base"],
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="rrf",
        choices=["dense", "bm25", "rrf", "weighted"],
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
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
        "--chat-model",
        type=str,
        default="llama3.2:3b",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
    )
    return parser.parse_args(args)


def build_retriever(
    track: CorpusTrack,
    fusion: str,
    reranker_model: str,
    candidate_k: int,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    ollama_host: str,
    timeout: float,
    use_reranker: bool,
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
        base = retriever

    elif fusion == "bm25":
        base = BM25Retriever(all_chunks)

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
        base = HybridRetriever(
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
        base = HybridRetriever(
            dense_retriever,
            bm25_retriever,
            fusion_method="weighted",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            candidate_k=candidate_k,
        )

    if use_reranker:
        reranker = CrossEncoderReranker(model_name=reranker_model)
        return RerankingRetriever(base, reranker, candidate_k=candidate_k, final_k=5)
    else:
        return base


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    # Build reranked retriever
    print("Building retriever with reranker...")
    retriever = build_retriever(
        track=track,
        fusion=parsed.fusion,
        reranker_model=parsed.reranker_model,
        candidate_k=parsed.candidate_k,
        rrf_k=parsed.rrf_k,
        dense_weight=parsed.dense_weight,
        sparse_weight=parsed.sparse_weight,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        embed_model=parsed.embed_model,
        ollama_host=parsed.ollama_host,
        timeout=parsed.timeout,
        use_reranker=True,
    )

    # Build answer generator
    provider = OllamaLLMProvider(parsed.chat_model, host=parsed.ollama_host, timeout=parsed.timeout)
    settings = GenerationSettings(temperature=0.0, max_tokens=512)
    answerer = GroundedAnswerGenerator(retriever, provider, top_k=5, settings=settings)

    print(
        f"\n=== Track D Safety Validation: {track.value.upper()} | {parsed.fusion.upper()}+Reranker ==="
    )

    abstained_count = 0
    answered_count = 0
    correct_abstentions = 0
    false_answers = 0

    for case in dataset.cases:
        # Retrieve
        retrieved = retriever.retrieve(case.query, top_k=5)

        # Generate
        generated = answerer.answer_with_sources(case.query, retrieved)
        answer = generated.answer

        # Check abstention
        abstained = is_abstention(answer)

        if abstained:
            abstained_count += 1
            correct_abstentions += 1
            print(f"  [OK] {case.query_id}: ABSTAINED")
        else:
            answered_count += 1
            false_answers += 1
            print(f"  [FAIL] {case.query_id}: ANSWERED (false answer)")

    total = len(dataset.cases)
    abstention_acc = correct_abstentions / total
    far = false_answers / total

    print("\n=== Track D Results ===")
    print(f"Total queries: {total}")
    print(f"Correctly abstained: {correct_abstentions}")
    print(f"False answers: {false_answers}")
    print(f"Abstention Accuracy: {abstention_acc:.2f}")
    print(f"False Answer Rate: {far:.2f}")

    # Compare with Phase 2 baseline
    print("\n=== Comparison with Phase 2 Baseline ===")
    print("Phase 2: 0.96 abstention accuracy, 0.04 FAR")
    print(f"Current: {abstention_acc:.2f} abstention accuracy, {far:.2f} FAR")

    return 0


if __name__ == "__main__":
    sys.exit(main())
