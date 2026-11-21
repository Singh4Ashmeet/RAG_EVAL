"""Oracle labeling evaluator for query retrieval strategies."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from core.embed import Embedder
from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.metrics.retrieval import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k
from eval.routing.schema import (
    OracleLabelRecord,
    OracleStrategy,
    RetrievalMetrics,
    RoutingExample,
)


class CachedEmbedder:
    """Caching wrapper around an Embedder to avoid duplicate embedding calls."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", "cached_embedder")

    def embed_query(self, text: str) -> list[float]:
        cleaned = text.strip()
        if cleaned not in self._cache:
            self._cache[cleaned] = self._inner.embed_query(cleaned)
        return self._cache[cleaned]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Check which texts need embedding
        missing = [t for t in texts if t.strip() not in self._cache]
        if missing:
            vectors = self._inner.embed_documents(missing)
            for t, v in zip(missing, vectors, strict=False):
                self._cache[t.strip()] = v
        return [self._cache[t.strip()] for t in texts]

    def warm_cache(self, texts: list[str], batch_size: int = 25) -> None:
        """Pre-embed a collection of texts in efficient batches to warm up the cache."""
        missing = [t.strip() for t in texts if t.strip() not in self._cache]
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            vectors = self._inner.embed_documents(batch)
            for t, v in zip(batch, vectors, strict=False):
                self._cache[t] = v


def calculate_metrics_for_retrieval(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> RetrievalMetrics:
    """Calculate standard retrieval metrics for cutoffs 1, 3, 5."""
    return RetrievalMetrics(
        hit_at_1=hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=1),
        hit_at_3=hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=3),
        hit_at_5=hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=5),
        mrr_at_5=reciprocal_rank_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=5),
        recall_at_5=recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=5),
    )


def determine_oracle_strategy(
    bm25_m: RetrievalMetrics,
    dense_m: RetrievalMetrics,
    hybrid_m: RetrievalMetrics,
) -> tuple[OracleStrategy, bool, float]:
    """Determine winning retrieval strategy based on retrieval effectiveness.

    Scoring formula combines MRR@5 with Hit@1 preference:
        score = MRR@5 + 0.5 * Hit@1

    Returns:
        (winning_strategy, is_tie, margin)
    """
    bm25_score = bm25_m.mrr_at_5 + 0.5 * bm25_m.hit_at_1
    dense_score = dense_m.mrr_at_5 + 0.5 * dense_m.hit_at_1
    hybrid_score = hybrid_m.mrr_at_5 + 0.5 * hybrid_m.hit_at_1

    # Case 1: All fail
    if bm25_score == 0.0 and dense_score == 0.0 and hybrid_score == 0.0:
        return OracleStrategy.NO_CLEAR_WINNER, True, 0.0

    # Case 2: Clean BM25 win
    if bm25_score > dense_score and bm25_score > hybrid_score:
        margin = bm25_score - max(dense_score, hybrid_score)
        return OracleStrategy.BM25, False, round(margin, 4)

    # Case 3: Clean Dense win
    if dense_score > bm25_score and dense_score > hybrid_score:
        margin = dense_score - max(bm25_score, hybrid_score)
        return OracleStrategy.DENSE, False, round(margin, 4)

    # Case 4: Clean Hybrid win
    if hybrid_score > bm25_score and hybrid_score > dense_score:
        margin = hybrid_score - max(bm25_score, dense_score)
        return OracleStrategy.HYBRID, False, round(margin, 4)

    # Case 5: BM25 ties with Hybrid, beating Dense (BM25 preferred for efficiency)
    if bm25_score == hybrid_score and bm25_score > dense_score:
        margin = bm25_score - dense_score
        return OracleStrategy.BM25, True, round(margin, 4)

    # Case 6: Dense ties with Hybrid, beating BM25 (Dense preferred for efficiency)
    if dense_score == hybrid_score and dense_score > bm25_score:
        margin = dense_score - bm25_score
        return OracleStrategy.DENSE, True, round(margin, 4)

    # Case 7: All tie with non-zero score
    return OracleStrategy.AMBIGUOUS, True, 0.0


class OracleEvaluator:
    """Indexes documents and runs oracle evaluations over query examples."""

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        candidate_k: int = 20,
        rrf_k: int = 100,
    ) -> None:
        self.embedder = CachedEmbedder(embedder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.pipeline = create_default_ingestion_pipeline(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.all_chunks: list[Chunk] = []
        self._temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.vector_store = SQLiteVectorStore(self._temp_db.name)
        self.dense_retriever = DenseRetriever(self.embedder, self.vector_store)
        self.bm25_retriever: BM25Retriever | None = None
        self.hybrid_retriever: HybridRetriever | None = None

    def index_documents(self, document_paths: Sequence[Path]) -> None:
        """Ingest and index all source documents for retrieval."""
        all_chunks: list[Chunk] = []
        for doc_path in document_paths:
            if not doc_path.exists():
                continue
            res = self.pipeline.ingest(doc_path)
            if res.document and res.chunks:
                all_chunks.extend(res.chunks)
                self.dense_retriever.index_document(res.document.metadata.document_id, res.chunks)

        self.all_chunks = all_chunks
        self.bm25_retriever = BM25Retriever(all_chunks)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever,
            self.bm25_retriever,
            fusion_method="rrf",
            rrf_k=self.rrf_k,
            candidate_k=self.candidate_k,
        )

    def warm_query_cache(self, queries: Sequence[str], batch_size: int = 25) -> None:
        """Pre-embed all query strings in efficient batches."""
        self.embedder.warm_cache(list(queries), batch_size=batch_size)

    def evaluate_example(self, example: RoutingExample) -> OracleLabelRecord:
        """Evaluate one example across BM25, Dense, and Hybrid."""
        if not self.bm25_retriever or not self.hybrid_retriever:
            raise RuntimeError("Documents must be indexed before evaluation.")

        top_k = 5
        bm25_res = self.bm25_retriever.retrieve(example.query, top_k=top_k)
        dense_res = self.dense_retriever.retrieve(example.query, top_k=top_k)
        hybrid_res = self.hybrid_retriever.retrieve(example.query, top_k=top_k)

        bm25_ids = [r.chunk.chunk_id for r in bm25_res]
        dense_ids = [r.chunk.chunk_id for r in dense_res]
        hybrid_ids = [r.chunk.chunk_id for r in hybrid_res]

        bm25_metrics = calculate_metrics_for_retrieval(bm25_ids, example.source_chunk_ids)
        dense_metrics = calculate_metrics_for_retrieval(dense_ids, example.source_chunk_ids)
        hybrid_metrics = calculate_metrics_for_retrieval(hybrid_ids, example.source_chunk_ids)

        oracle_strategy, is_tie, margin = determine_oracle_strategy(
            bm25_metrics, dense_metrics, hybrid_metrics
        )

        return OracleLabelRecord(
            example_id=example.example_id,
            query=example.query,
            source_document=example.source_document,
            split=example.split,
            bm25_metrics=bm25_metrics,
            dense_metrics=dense_metrics,
            hybrid_metrics=hybrid_metrics,
            oracle_strategy=oracle_strategy,
            is_tie=is_tie,
            margin=margin,
        )

    def close(self) -> None:
        """Clean up vector store and temporary database."""
        self.vector_store.close()
        try:
            Path(self._temp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
