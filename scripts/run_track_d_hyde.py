"""Run Track D with HyDE for safety validation."""

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
from core.hyde.retriever import HyDERetriever
from core.hyde.transformer import HyDEConfig, HyDETransformer
from core.ingest import create_default_ingestion_pipeline
from core.llm.base import GenerationSettings
from core.llm.ollama_provider import OllamaLLMProvider
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack
from eval.metrics.abstention import is_abstention


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track D with HyDE for safety validation.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="dense_bm25_hyde",
        choices=["dense_bm25", "hyde_dense", "dense_hyde", "dense_bm25_hyde"],
        help="Retrieval strategy to use.",
    )
    parser.add_argument(
        "--hyde-model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model for HyDE generation (default: llama3.2:3b).",
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
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=100,
    )
    return parser.parse_args(args)


def build_retriever(
    track: CorpusTrack,
    strategy: str,
    hyde_model: str,
    hyde_temperature: float,
    hyde_max_tokens: int,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    ollama_host: str,
    timeout: float,
    candidate_k: int,
    rrf_k: int,
):
    """Build the retriever with query transformation."""

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

    # Build hybrid retriever (RRF)
    HybridRetriever(
        dense_retriever,
        bm25_retriever,
        fusion_method="rrf",
        rrf_k=100,
        candidate_k=candidate_k,
    )

    # Build HyDE transformer and retriever
    hyde_config_obj = HyDEConfig(
        model_name=hyde_model,
        ollama_host=ollama_host,
        timeout=timeout,
        temperature=hyde_temperature,
        max_tokens=hyde_max_tokens,
        prompt_template=None,
        max_passage_length=2000,
        min_passage_length=50,
    )
    hyde_transformer = HyDETransformer(hyde_config_obj)
    hyde_retriever = HyDERetriever(
        dense_retriever,
        hyde_transformer,
        use_original_query=True,
    )

    # Build final retriever based on strategy
    if strategy == "dense_bm25":
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            fusion_method="rrf",
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )
    elif strategy == "hyde_dense":
        hyde_retriever = HyDERetriever(
            dense_retriever,
            hyde_transformer,
            use_original_query=True,
        )
        return hyde_retriever
    elif strategy == "dense_hyde":
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            hyde_retriever=HyDERetriever(
                dense_retriever,
                HyDETransformer(
                    HyDEConfig(
                        model_name=hyde_model,
                        ollama_host=ollama_host,
                        timeout=timeout,
                        temperature=hyde_temperature,
                        max_tokens=hyde_max_tokens,
                        prompt_template=None,
                        max_passage_length=2000,
                        min_passage_length=50,
                    )
                ),
                use_original_query=True,
            ),
            fusion_method="rrf",
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )
    elif strategy == "dense_bm25_hyde":
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            hyde_retriever=HyDERetriever(
                dense_retriever,
                HyDETransformer(
                    HyDEConfig(
                        model_name=hyde_model,
                        ollama_host=ollama_host,
                        timeout=timeout,
                        temperature=hyde_temperature,
                        max_tokens=hyde_max_tokens,
                        prompt_template=None,
                        max_passage_length=2000,
                        min_passage_length=50,
                    )
                ),
                use_original_query=True,
            ),
            fusion_method="rrf",
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack("track_d")

    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    # Build retriever
    print("Building retriever with query transformation...")
    retriever = build_retriever(
        track=track,
        strategy=parsed.strategy,
        hyde_model=parsed.hyde_model,
        hyde_temperature=parsed.hyde_temperature,
        hyde_max_tokens=parsed.hyde_max_tokens,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        embed_model=parsed.embed_model,
        ollama_host=parsed.ollama_host,
        timeout=parsed.timeout,
        candidate_k=parsed.candidate_k,
        rrf_k=parsed.rrf_k,
    )

    # Build answer generator
    provider = OllamaLLMProvider(parsed.chat_model, host=parsed.ollama_host, timeout=parsed.timeout)
    settings = GenerationSettings(temperature=0.0, max_tokens=512)
    answerer = GroundedAnswerGenerator(retriever, provider, top_k=5, settings=settings)

    print(f"\n=== Track D Safety Validation: {parsed.strategy.upper()} ===")

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
