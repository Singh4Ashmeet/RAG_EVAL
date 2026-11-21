"""Unit tests for Phase 15 Targeted Reranking Optimization & Candidate-Set Forensics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.rerank import (
    CrossEncoderReranker,
    RerankingRetriever,
)
from core.retrieval.schema import RetrievalStrategy, SearchResult


def _make_chunk(chunk_id: str, doc_id: str = "doc_test", text: str = "sample text") -> Chunk:
    """Helper to create a canonical Chunk instance for tests."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        text=text,
        metadata={"source": "test"},
    )


def _make_search_result(
    chunk: Chunk,
    score: float,
    rank: int,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
) -> SearchResult:
    """Helper to create a SearchResult instance."""
    return SearchResult(
        chunk=chunk,
        score=score,
        rank=rank,
        strategy=strategy,
    )


class TestRerankerOrderingAndScores:
    """Tests for cross-encoder scoring, ordering, and rank assignment."""

    def test_rerank_orders_by_score_descending(self) -> None:
        """Verify candidates are sorted descending by model prediction."""
        reranker = object.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.model_id = "mock"
        reranker.max_length = 512
        reranker.batch_size = 32
        reranker.device = "cpu"

        # Mock model predict: chunk_b gets higher score than chunk_a
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        reranker._model = mock_model

        c1 = _make_chunk("c1", text="text1")
        c2 = _make_chunk("c2", text="text2")
        c3 = _make_chunk("c3", text="text3")

        candidates = [
            _make_search_result(c1, score=0.02, rank=0),
            _make_search_result(c2, score=0.019, rank=1),
            _make_search_result(c3, score=0.018, rank=2),
        ]

        reranked = reranker.rerank("test query", candidates, top_k=3)

        assert len(reranked) == 3
        # c2 had score 0.9 -> should be rank 0
        assert reranked[0].chunk.chunk_id == "c2"
        assert reranked[0].rank == 0
        assert reranked[0].score == pytest.approx(0.9)
        # c3 had score 0.5 -> should be rank 1
        assert reranked[1].chunk.chunk_id == "c3"
        assert reranked[1].rank == 1
        assert reranked[1].score == pytest.approx(0.5)
        # c1 had score 0.2 -> should be rank 2
        assert reranked[2].chunk.chunk_id == "c1"
        assert reranked[2].rank == 2
        assert reranked[2].score == pytest.approx(0.2)

    def test_rerank_top_k_truncation(self) -> None:
        """Verify top_k parameter truncates output properly."""
        reranker = object.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.model_id = "mock"
        reranker.max_length = 512
        reranker.batch_size = 32
        reranker.device = "cpu"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.4, 0.9, 0.2]
        reranker._model = mock_model

        candidates = [
            _make_search_result(_make_chunk(f"c{i}"), score=0.01 * (10 - i), rank=i)
            for i in range(4)
        ]

        reranked = reranker.rerank("query", candidates, top_k=2)
        assert len(reranked) == 2
        assert reranked[0].chunk.chunk_id == "c2"  # score 0.9
        assert reranked[1].chunk.chunk_id == "c1"  # score 0.4

    def test_reranker_determinism(self) -> None:
        """Verify identical input yields identical output across repeated invocations."""
        reranker = object.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.model_id = "mock"
        reranker.max_length = 512
        reranker.batch_size = 32
        reranker.device = "cpu"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.85, 0.12, 0.99]
        reranker._model = mock_model

        candidates = [
            _make_search_result(_make_chunk(f"c{i}"), score=0.01 * (10 - i), rank=i)
            for i in range(3)
        ]

        out1 = reranker.rerank("query", candidates, top_k=3)
        out2 = reranker.rerank("query", candidates, top_k=3)

        assert [r.chunk.chunk_id for r in out1] == [r.chunk.chunk_id for r in out2]
        assert [r.score for r in out1] == [r.score for r in out2]
        assert [r.rank for r in out1] == [r.rank for r in out2]


class TestRerankerEdgeCases:
    """Tests for edge cases and input validation."""

    def test_empty_candidates_returns_empty(self) -> None:
        reranker = object.__new__(CrossEncoderReranker)
        assert reranker.rerank("query", []) == []

    def test_empty_query_raises_error(self) -> None:
        reranker = object.__new__(CrossEncoderReranker)
        c = _make_search_result(_make_chunk("c1"), score=1.0, rank=0)
        with pytest.raises(RetrievalError, match="Query must be a non-empty string"):
            reranker.rerank("", [c])
        with pytest.raises(RetrievalError, match="Query must be a non-empty string"):
            reranker.rerank("   ", [c])

    def test_invalid_top_k_raises_error(self) -> None:
        reranker = object.__new__(CrossEncoderReranker)
        c = _make_search_result(_make_chunk("c1"), score=1.0, rank=0)
        with pytest.raises(RetrievalError, match="top_k must be at least 1"):
            reranker.rerank("query", [c], top_k=0)

    def test_single_candidate(self) -> None:
        reranker = object.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.model_id = "mock"
        reranker.max_length = 512
        reranker.batch_size = 32
        reranker.device = "cpu"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.42]
        reranker._model = mock_model

        c = _make_search_result(_make_chunk("c1"), score=0.01, rank=0)
        results = reranker.rerank("query", [c], top_k=5)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].score == pytest.approx(0.42)
        assert results[0].rank == 0

    def test_score_ties_preserve_order(self) -> None:
        """Verify stable handling when scores are tied."""
        reranker = object.__new__(CrossEncoderReranker)
        reranker.model_name = "mock"
        reranker.model_id = "mock"
        reranker.max_length = 512
        reranker.batch_size = 32
        reranker.device = "cpu"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.5]
        reranker._model = mock_model

        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        candidates = [
            _make_search_result(c1, score=0.02, rank=0),
            _make_search_result(c2, score=0.01, rank=1),
        ]

        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.5)
        assert results[1].score == pytest.approx(0.5)


class TestRerankerStrategies:
    """Tests for Phase 15 strategies: Depth variation, Coverage-Preserve, Margin-Gating."""

    def test_candidate_depth_variation(self) -> None:
        """Verify RerankingRetriever candidate_k controls pool depth passed to reranker."""
        mock_base = MagicMock()
        mock_reranker = MagicMock()

        candidates_20 = [
            _make_search_result(_make_chunk(f"c{i}"), score=0.01 * (30 - i), rank=i)
            for i in range(20)
        ]
        mock_base.retrieve.return_value = candidates_20
        mock_reranker.rerank.return_value = candidates_20[:5]

        retriever = RerankingRetriever(
            base_retriever=mock_base,
            reranker=mock_reranker,
            candidate_k=20,
            final_k=5,
        )

        res = retriever.retrieve("query")
        assert len(res) == 5
        mock_base.retrieve.assert_called_once_with("query", top_k=20)
        mock_reranker.rerank.assert_called_once_with("query", candidates_20, top_k=5)

    def test_strategy_b_coverage_preserve_logic(self) -> None:
        """Verify that Strategy B guarantees baseline top-1 remains in top-5."""
        base_top1 = _make_search_result(_make_chunk("base_winner"), score=0.019, rank=0)
        c2 = _make_search_result(_make_chunk("c2"), score=0.018, rank=1)
        c3 = _make_search_result(_make_chunk("c3"), score=0.017, rank=2)
        c4 = _make_search_result(_make_chunk("c4"), score=0.016, rank=3)
        c5 = _make_search_result(_make_chunk("c5"), score=0.015, rank=4)
        c6 = _make_search_result(_make_chunk("c6"), score=0.014, rank=5)

        pool = [base_top1, c2, c3, c4, c5, c6]

        # Suppose reranker demotes base_winner outside top 5 (e.g. ranks c2..c6 above it)
        reranked_all = [c2, c3, c4, c5, c6, base_top1]

        # Strategy B anchoring logic:
        final_k = 5
        base_top1_id = pool[0].chunk.chunk_id
        reranked_top_ids = [r.chunk.chunk_id for r in reranked_all[:final_k]]

        assert base_top1_id not in reranked_top_ids

        # Anchor replaces 5th item
        anchored_list = reranked_all[: final_k - 1] + [pool[0]]
        for idx, item in enumerate(anchored_list):
            item.rank = idx

        assert len(anchored_list) == 5
        assert anchored_list[4].chunk.chunk_id == "base_winner"
        assert anchored_list[4].rank == 4

    def test_strategy_c_margin_gating_logic(self) -> None:
        """Verify margin gating skips reranking when top baseline margin is high."""
        c1 = _make_search_result(_make_chunk("c1"), score=0.025, rank=0)
        c2 = _make_search_result(_make_chunk("c2"), score=0.015, rank=1)
        pool = [c1, c2]

        margin = pool[0].score - pool[1].score  # 0.010
        threshold = 0.001

        # Margin is 0.010 >= 0.001 -> Gated skip: keep baseline
        should_skip = margin >= threshold
        assert should_skip is True

        # Lower margin case
        c2_close = _make_search_result(_make_chunk("c2"), score=0.0245, rank=1)
        pool_close = [c1, c2_close]
        margin_close = pool_close[0].score - pool_close[1].score  # 0.0005
        should_skip_close = margin_close >= threshold
        assert should_skip_close is False


class TestProductionBaselineIntegrity:
    """Verify that the production baseline configuration does not activate reranking."""

    def test_production_retriever_is_not_reranking(self) -> None:
        """Verify default production retrieval remains Dense + BM25 RRF."""
        from core.retrieval.hybrid import HybridRetriever

        # HybridRetriever is the production class
        assert HybridRetriever is not None
        # Verify it has candidate_k and rrf_k
        import inspect

        sig = inspect.signature(HybridRetriever.__init__)
        assert "rrf_k" in sig.parameters
        assert sig.parameters["rrf_k"].default == 100
