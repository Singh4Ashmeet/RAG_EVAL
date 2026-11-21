"""Unit tests for Phase 14 Targeted Ranking Optimization & RRF Forensics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.hybrid import HybridRetriever, _rrf_fusion
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
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE,
) -> SearchResult:
    """Helper to create a SearchResult instance."""
    return SearchResult(
        chunk=chunk,
        score=score,
        rank=rank,
        strategy=strategy,
    )


class TestRRFScoring:
    """Tests for Reciprocal Rank Fusion scoring mathematics."""

    def test_single_retriever_scoring(self) -> None:
        """Verify RRF(d) = 1 / (k + rank) for single retriever."""
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")

        res = [
            _make_search_result(c1, score=0.9, rank=0),
            _make_search_result(c2, score=0.8, rank=1),
        ]

        fused = _rrf_fusion([res], k=100, top_k=2)

        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "c1"
        assert fused[0].score == pytest.approx(1.0 / (100 + 1))
        assert fused[1].chunk.chunk_id == "c2"
        assert fused[1].score == pytest.approx(1.0 / (100 + 2))

    def test_dual_retriever_scoring(self) -> None:
        """Verify RRF sum over two retrievers."""
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")

        dense = [
            _make_search_result(c1, score=0.9, rank=0, strategy=RetrievalStrategy.DENSE),
            _make_search_result(c2, score=0.8, rank=1, strategy=RetrievalStrategy.DENSE),
        ]
        bm25 = [
            _make_search_result(c1, score=5.0, rank=0, strategy=RetrievalStrategy.SPARSE),
            _make_search_result(c2, score=3.0, rank=1, strategy=RetrievalStrategy.SPARSE),
        ]

        fused = _rrf_fusion([dense, bm25], k=60, top_k=2)

        expected_c1 = (1.0 / 61) + (1.0 / 61)
        expected_c2 = (1.0 / 62) + (1.0 / 62)

        assert fused[0].chunk.chunk_id == "c1"
        assert fused[0].score == pytest.approx(expected_c1)
        assert fused[1].chunk.chunk_id == "c2"
        assert fused[1].score == pytest.approx(expected_c2)

    def test_disjoint_candidates_scoring(self) -> None:
        """Chunks appearing in only one retriever receive single-source score."""
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")

        dense = [_make_search_result(c1, score=0.9, rank=0, strategy=RetrievalStrategy.DENSE)]
        bm25 = [_make_search_result(c2, score=5.0, rank=0, strategy=RetrievalStrategy.SPARSE)]

        fused = _rrf_fusion([dense, bm25], k=100, top_k=2)

        assert len(fused) == 2
        # Both have rank 0 in their respective retriever, so score is 1/(100+1)
        assert fused[0].score == pytest.approx(1.0 / 101)
        assert fused[1].score == pytest.approx(1.0 / 101)


class TestDeterministicOrderingAndTieHandling:
    """Tests for deterministic ordering and tie-breaking behavior."""

    def test_tie_handling_is_deterministic(self) -> None:
        """When two chunks have the exact same RRF score, tie-break by chunk_id."""
        c_a = _make_chunk("chunk_alpha")
        c_b = _make_chunk("chunk_beta")

        # Symmetric ranks: c_a is (0 in dense, 1 in bm25); c_b is (1 in dense, 0 in bm25)
        # Both have score = 1/(100+1) + 1/(100+2)
        dense = [
            _make_search_result(c_a, score=0.9, rank=0),
            _make_search_result(c_b, score=0.8, rank=1),
        ]
        bm25 = [
            _make_search_result(c_b, score=5.0, rank=0),
            _make_search_result(c_a, score=3.0, rank=1),
        ]

        # Run multiple times to verify determinism across invocations
        for _ in range(5):
            fused = _rrf_fusion([dense, bm25], k=100, top_k=2)
            assert fused[0].score == pytest.approx(fused[1].score)
            # chunk_alpha < chunk_beta lexicographically
            assert fused[0].chunk.chunk_id == "chunk_alpha"
            assert fused[1].chunk.chunk_id == "chunk_beta"

    def test_reverse_input_preserves_deterministic_order(self) -> None:
        """Changing input candidate order does not affect final deterministic tie order."""
        c_a = _make_chunk("chunk_alpha")
        c_b = _make_chunk("chunk_beta")

        dense = [
            _make_search_result(c_b, score=0.9, rank=0),
            _make_search_result(c_a, score=0.8, rank=1),
        ]
        bm25 = [
            _make_search_result(c_a, score=5.0, rank=0),
            _make_search_result(c_b, score=3.0, rank=1),
        ]

        fused = _rrf_fusion([dense, bm25], k=100, top_k=2)
        assert fused[0].chunk.chunk_id == "chunk_alpha"
        assert fused[1].chunk.chunk_id == "chunk_beta"


class TestWeightedFusion:
    """Tests for weighted RRF fusion."""

    def test_equal_weights_matches_unweighted(self) -> None:
        """Weighted RRF with equal weights produces same ordering as unweighted."""
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")

        dense = [
            _make_search_result(c1, score=0.9, rank=0),
            _make_search_result(c2, score=0.8, rank=1),
        ]
        bm25 = [
            _make_search_result(c2, score=5.0, rank=0),
            _make_search_result(c1, score=3.0, rank=1),
        ]

        unweighted = _rrf_fusion([dense, bm25], k=60, top_k=2, weights=None)
        weighted = _rrf_fusion([dense, bm25], k=60, top_k=2, weights=[1.0, 1.0])

        assert [r.chunk.chunk_id for r in unweighted] == [r.chunk.chunk_id for r in weighted]
        for u, w in zip(unweighted, weighted, strict=True):
            assert u.score == pytest.approx(w.score)

    def test_asymmetric_weights_shift_ranking(self) -> None:
        """Higher weight for dense causes dense winner to overcome bm25 winner."""
        c_dense_winner = _make_chunk("c_dense_winner")
        c_bm25_winner = _make_chunk("c_bm25_winner")

        dense = [
            _make_search_result(c_dense_winner, score=0.9, rank=0),
            _make_search_result(c_bm25_winner, score=0.8, rank=1),
        ]
        bm25 = [
            _make_search_result(c_bm25_winner, score=5.0, rank=0),
            _make_search_result(c_dense_winner, score=3.0, rank=1),
        ]

        # With dense weight 0.8 and bm25 weight 0.2:
        # c_dense: 0.8/61 + 0.2/62 = 0.01311 + 0.00323 = 0.01634
        # c_bm25:  0.8/62 + 0.2/61 = 0.01290 + 0.00328 = 0.01618
        fused = _rrf_fusion([dense, bm25], k=60, top_k=2, weights=[0.8, 0.2])
        assert fused[0].chunk.chunk_id == "c_dense_winner"
        assert fused[1].chunk.chunk_id == "c_bm25_winner"

    def test_weights_length_validation(self) -> None:
        """Mismatched weights length raises RetrievalError."""
        c1 = _make_chunk("c1")
        dense = [_make_search_result(c1, score=0.9, rank=0)]
        bm25 = [_make_search_result(c1, score=5.0, rank=0)]

        with pytest.raises(RetrievalError, match="weights must have the same length"):
            _rrf_fusion([dense, bm25], k=60, weights=[1.0])


class TestEdgeCasesAndValidation:
    """Tests for edge cases and parameter validation."""

    def test_empty_results_list(self) -> None:
        """Empty input results list returns empty."""
        fused = _rrf_fusion([], k=60, top_k=5)
        assert fused == []

    def test_empty_individual_retriever(self) -> None:
        """One retriever returns empty, other returns results."""
        c1 = _make_chunk("c1")
        dense: list[SearchResult] = []
        bm25 = [_make_search_result(c1, score=5.0, rank=0)]

        fused = _rrf_fusion([dense, bm25], k=100, top_k=5)
        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "c1"
        assert fused[0].score == pytest.approx(1.0 / 101)

    def test_top_k_truncation(self) -> None:
        """top_k truncates results to requested count."""
        chunks = [_make_chunk(f"c{i}") for i in range(10)]
        dense = [_make_search_result(c, score=1.0 - i * 0.1, rank=i) for i, c in enumerate(chunks)]

        fused = _rrf_fusion([dense], k=100, top_k=3)
        assert len(fused) == 3

    def test_invalid_rrf_k(self) -> None:
        """Invalid rrf_k < 1 raises RetrievalError in HybridRetriever."""
        from unittest.mock import MagicMock

        mock_dense = MagicMock()
        mock_bm25 = MagicMock()

        with pytest.raises(RetrievalError, match="must be >= 1"):
            HybridRetriever(mock_dense, mock_bm25, rrf_k=0)


class TestProductionConfigurationWiring:
    """Tests verifying production baseline configuration is preserved."""

    def test_default_hybrid_retriever_config(self) -> None:
        """Verify HybridRetriever default configuration matches frozen baseline."""
        from unittest.mock import MagicMock

        mock_dense = MagicMock()
        mock_bm25 = MagicMock()

        retriever = HybridRetriever(mock_dense, mock_bm25)
        stats = retriever.get_stats()

        assert stats["fusion_method"] == "rrf"
        assert stats["rrf_k"] == 100
        assert stats["candidate_k"] == 20
        assert stats["weights"] is None

    def test_phase14_report_exists_and_retains_baseline(self) -> None:
        """Verify Phase 14 evaluation report exists and records BASELINE_RETAINED."""
        report_path = Path("docs/reports/phase14_ranking_evaluation.json")
        assert report_path.exists(), "Phase 14 report must exist"

        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["phase"] == 14
        assert data["decision"] == "BASELINE_RETAINED"
        assert data["baseline_config"]["rrf_k"] == 100
        assert data["baseline_config"]["candidate_k"] == 20
        assert data["baseline_config"]["final_k"] == 5

    def test_all_121_queries_analyzed_in_report(self) -> None:
        """Verify that all 121 protected queries are analyzed in the forensic report."""
        report_path = Path("docs/reports/phase14_ranking_evaluation.json")
        data = json.loads(report_path.read_text(encoding="utf-8"))

        total_analyzed = data["rrf_forensics"]["total_queries_analyzed"]
        assert total_analyzed == 121

        # Check track totals
        baseline_metrics = data["baseline_metrics"]
        assert baseline_metrics["Track A"]["total_queries"] == 14
        assert baseline_metrics["Track B"]["total_queries"] == 52
        assert baseline_metrics["Track C"]["total_queries"] == 30
        assert baseline_metrics["Track D"]["total_queries"] == 25

    def test_track_safety_in_baseline(self) -> None:
        """Verify all safety constraints are satisfied by the retained baseline."""
        report_path = Path("docs/reports/phase14_ranking_evaluation.json")
        data = json.loads(report_path.read_text(encoding="utf-8"))

        bm = data["baseline_metrics"]
        assert bm["Track A"]["hit_at_5"] == 1.0, "Track A Hit@5 must be 1.0"
        assert bm["Track C"]["hit_at_1"] == 1.0, "Track C Hit@1 must be 1.0"
        assert bm["Track D"]["abstention_safe"] == 25, "Track D must have 25 abstention safe"
        assert bm["Track D"]["false_answer_rate"] == 0.0, "Track D false answer rate must be 0.0"
