"""Unit tests for embedding and retrieval contracts, protocols, and exceptions."""

import json
import math

import pytest
from pydantic import ValidationError

from core.embed import Embedder
from core.errors import EmbedderError, RAGEvalError, RetrievalError
from core.normalize.schema import Chunk
from core.retrieval import Reranker, RetrievalStrategy, Retriever, SearchResult

# ---------------------------------------------------------------------------
# Dummy Test Implementations
# ---------------------------------------------------------------------------


class DummyEmbedder:
    """Conforming dummy implementation of Embedder protocol."""

    @property
    def dimension(self) -> int:
        return 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dimension


class IncompleteEmbedderMissingDimension:
    """Class missing dimension property."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class IncompleteEmbedderMissingMethods:
    """Class missing embed methods."""

    @property
    def dimension(self) -> int:
        return 128


class DummyRetriever:
    """Conforming dummy implementation of Retriever protocol."""

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        chunk = Chunk(
            chunk_id="chk-001",
            document_id="doc-001",
            text="Sample chunk for retriever test.",
            chunk_index=0,
        )
        return [
            SearchResult(
                chunk=chunk,
                score=0.88,
                rank=0,
                strategy=RetrievalStrategy.DENSE,
            )
        ][:top_k]


class IncompleteRetriever:
    """Class missing retrieve method."""

    def query(self, text: str) -> list[str]:
        return [text]


class DummyReranker:
    """Conforming dummy implementation of Reranker protocol."""

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        return [
            SearchResult(
                chunk=item.chunk,
                score=item.score,
                rank=idx,
                strategy=item.strategy,
            )
            for idx, item in enumerate(sorted_candidates[:top_k])
        ]


class IncompleteReranker:
    """Class missing rerank method."""

    def rank(self, candidates: list[str]) -> list[str]:
        return candidates


# ---------------------------------------------------------------------------
# Helper fixture
# ---------------------------------------------------------------------------


def make_sample_chunk(chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-test",
        text="Valid chunk text for retrieval testing.",
        chunk_index=0,
    )


# ---------------------------------------------------------------------------
# Exception Hierarchy Tests
# ---------------------------------------------------------------------------


class TestRetrievalExceptionHierarchy:
    def test_inheritance(self):
        assert issubclass(EmbedderError, RAGEvalError)
        assert issubclass(RetrievalError, RAGEvalError)
        assert issubclass(RAGEvalError, Exception)

    def test_catchable_as_rag_eval_error(self):
        try:
            raise EmbedderError("embedding failed")
        except RAGEvalError as exc:
            assert str(exc) == "embedding failed"

        try:
            raise RetrievalError("retrieval query timed out")
        except RAGEvalError as exc:
            assert str(exc) == "retrieval query timed out"


# ---------------------------------------------------------------------------
# Embedder Protocol Tests
# ---------------------------------------------------------------------------


class TestEmbedderProtocol:
    def test_conforming_embedder_satisfies_protocol(self):
        embedder = DummyEmbedder()
        assert isinstance(embedder, Embedder)

    def test_incomplete_embedder_rejected_by_protocol(self):
        assert not isinstance(IncompleteEmbedderMissingDimension(), Embedder)
        assert not isinstance(IncompleteEmbedderMissingMethods(), Embedder)

    def test_embedder_signatures_and_output(self):
        embedder = DummyEmbedder()
        assert embedder.dimension == 384

        docs_vecs = embedder.embed_documents(["First doc", "Second doc"])
        assert isinstance(docs_vecs, list)
        assert len(docs_vecs) == 2
        assert len(docs_vecs[0]) == 384
        assert len(docs_vecs[1]) == 384

        query_vec = embedder.embed_query("search query")
        assert isinstance(query_vec, list)
        assert len(query_vec) == 384


# ---------------------------------------------------------------------------
# SearchResult Schema Tests
# ---------------------------------------------------------------------------


class TestSearchResultSchema:
    def test_valid_search_result(self):
        chunk = make_sample_chunk()
        result = SearchResult(
            chunk=chunk,
            score=0.9125,
            rank=0,
            strategy=RetrievalStrategy.DENSE,
        )
        assert result.chunk == chunk
        assert result.score == 0.9125
        assert result.rank == 0
        assert result.strategy == RetrievalStrategy.DENSE

    def test_strategies_accepted(self):
        chunk = make_sample_chunk()
        dense_res = SearchResult(chunk=chunk, score=10.5, rank=0, strategy=RetrievalStrategy.DENSE)
        sparse_res = SearchResult(
            chunk=chunk, score=15.2, rank=1, strategy=RetrievalStrategy.SPARSE
        )
        hybrid_res = SearchResult(
            chunk=chunk, score=0.88, rank=2, strategy=RetrievalStrategy.HYBRID
        )
        assert dense_res.strategy == "dense"
        assert sparse_res.strategy == "sparse"
        assert hybrid_res.strategy == "hybrid"

    def test_invalid_strategy_rejected(self):
        chunk = make_sample_chunk()
        with pytest.raises(ValidationError):
            SearchResult(
                chunk=chunk,
                score=1.0,
                rank=0,
                strategy="semantic_neural",  # type: ignore[arg-type]
            )

    def test_extra_fields_forbidden(self):
        chunk = make_sample_chunk()
        with pytest.raises(ValidationError):
            SearchResult(
                chunk=chunk,
                score=0.5,
                rank=0,
                strategy=RetrievalStrategy.DENSE,
                extra_field="disallowed",
            )

    def test_negative_rank_rejected(self):
        chunk = make_sample_chunk()
        with pytest.raises(ValidationError):
            SearchResult(
                chunk=chunk,
                score=0.5,
                rank=-1,
                strategy=RetrievalStrategy.DENSE,
            )

    def test_non_finite_scores_rejected(self):
        chunk = make_sample_chunk()

        with pytest.raises(ValidationError, match="finite float"):
            SearchResult(
                chunk=chunk,
                score=math.nan,
                rank=0,
                strategy=RetrievalStrategy.DENSE,
            )

        with pytest.raises(ValidationError, match="finite float"):
            SearchResult(
                chunk=chunk,
                score=math.inf,
                rank=0,
                strategy=RetrievalStrategy.DENSE,
            )

        with pytest.raises(ValidationError, match="finite float"):
            SearchResult(
                chunk=chunk,
                score=-math.inf,
                rank=0,
                strategy=RetrievalStrategy.DENSE,
            )

    def test_serialization_roundtrip(self):
        chunk = make_sample_chunk()
        original = SearchResult(
            chunk=chunk,
            score=-12.34,  # e.g. log-likelihood / negative distance
            rank=4,
            strategy=RetrievalStrategy.SPARSE,
        )

        serialized = original.model_dump_json()
        data = json.loads(serialized)
        reconstructed = SearchResult.model_validate(data)

        assert reconstructed == original
        assert reconstructed.chunk.chunk_id == chunk.chunk_id
        assert reconstructed.score == -12.34
        assert reconstructed.strategy == RetrievalStrategy.SPARSE

        reconstructed_from_json = SearchResult.model_validate_json(serialized)
        assert reconstructed_from_json == original


# ---------------------------------------------------------------------------
# Retriever and Reranker Protocol Tests
# ---------------------------------------------------------------------------


class TestRetrieverProtocol:
    def test_conforming_retriever_satisfies_protocol(self):
        retriever = DummyRetriever()
        assert isinstance(retriever, Retriever)

    def test_incomplete_retriever_rejected_by_protocol(self):
        assert not isinstance(IncompleteRetriever(), Retriever)

    def test_retriever_execution(self):
        retriever = DummyRetriever()
        results = retriever.retrieve("query text", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].strategy == RetrievalStrategy.DENSE


class TestRerankerProtocol:
    def test_conforming_reranker_satisfies_protocol(self):
        reranker = DummyReranker()
        assert isinstance(reranker, Reranker)

    def test_incomplete_reranker_rejected_by_protocol(self):
        assert not isinstance(IncompleteReranker(), Reranker)

    def test_reranker_execution(self):
        reranker = DummyReranker()
        chunk_a = make_sample_chunk("chunk-a")
        chunk_b = make_sample_chunk("chunk-b")

        cand1 = SearchResult(chunk=chunk_a, score=0.60, rank=0, strategy=RetrievalStrategy.DENSE)
        cand2 = SearchResult(chunk=chunk_b, score=0.92, rank=1, strategy=RetrievalStrategy.DENSE)

        reranked = reranker.rerank("target query", [cand1, cand2], top_k=2)
        assert len(reranked) == 2
        # Higher score chunk-b should now be rank 0
        assert reranked[0].chunk.chunk_id == "chunk-b"
        assert reranked[0].rank == 0
        assert reranked[1].chunk.chunk_id == "chunk-a"
        assert reranked[1].rank == 1


# ---------------------------------------------------------------------------
# Public Export Tests
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_public_imports(self):
        from core.embed import Embedder as ImportedEmbedder
        from core.errors import EmbedderError as ImportedEmbedderError
        from core.errors import RetrievalError as ImportedRetrievalError
        from core.retrieval import Reranker as ImportedReranker
        from core.retrieval import RetrievalStrategy as ImportedStrategy
        from core.retrieval import Retriever as ImportedRetriever
        from core.retrieval import SearchResult as ImportedSearchResult

        assert ImportedEmbedder is Embedder
        assert ImportedEmbedderError is EmbedderError
        assert ImportedRetrievalError is RetrievalError
        assert ImportedRetriever is Retriever
        assert ImportedReranker is Reranker
        assert ImportedSearchResult is SearchResult
        assert ImportedStrategy is RetrievalStrategy
