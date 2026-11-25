"""Regression tests for the dependency-light shared RRF implementation."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from core.normalize.schema import Chunk
from core.retrieval.rrf import _rrf_fusion
from core.retrieval.schema import RetrievalStrategy, SearchResult


def _result(chunk_id: str, rank: int, strategy: RetrievalStrategy) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document",
            text=f"text for {chunk_id}",
            chunk_index=rank,
        ),
        score=1.0,
        rank=rank,
        strategy=strategy,
    )


def test_shared_rrf_preserves_previous_scores_order_and_weights() -> None:
    dense = [
        _result("dense-first", 0, RetrievalStrategy.DENSE),
        _result("shared", 1, RetrievalStrategy.DENSE),
    ]
    bm25 = [
        _result("shared", 0, RetrievalStrategy.SPARSE),
        _result("bm25-second", 1, RetrievalStrategy.SPARSE),
    ]

    fused = _rrf_fusion([dense, bm25], k=100, top_k=5, weights=[1.0, 1.0])

    assert [result.chunk.chunk_id for result in fused] == [
        "shared",
        "dense-first",
        "bm25-second",
    ]
    assert [result.rank for result in fused] == [0, 1, 2]
    assert [result.strategy for result in fused] == [RetrievalStrategy.HYBRID] * 3
    assert fused[0].score == pytest.approx((1 / 102) + (1 / 101))
    assert fused[1].score == pytest.approx(1 / 101)
    assert fused[2].score == pytest.approx(1 / 102)


def test_hybrid_reexports_the_shared_rrf_function() -> None:
    from core.retrieval.hybrid import _rrf_fusion as hybrid_rrf_fusion

    assert hybrid_rrf_fusion is _rrf_fusion


def test_lightweight_rrf_import_does_not_load_optional_ml_runtimes() -> None:
    program = """
import json
import sys
import core.retrieval.rrf
blocked = ('torch', 'sentence_transformers', 'core.sparse')
print(json.dumps(sorted(name for name in sys.modules if name.startswith(blocked))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
