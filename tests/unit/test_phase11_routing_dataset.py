"""Unit tests for Phase 11 independent routing dataset schemas, generator, oracle, and leakage."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.routing.generator import DEFAULT_DOCUMENT_SPECS, DocumentSpec, RoutingDatasetGenerator
from eval.routing.leakage import (
    check_query_leakage,
)
from eval.routing.oracle import (
    CachedEmbedder,
    determine_oracle_strategy,
)
from eval.routing.schema import (
    OracleStrategy,
    QueryCategory,
    RetrievalMetrics,
    RoutingExample,
    RoutingSplit,
)


class DummyTestEmbedder:
    """Mock embedder for unit test isolation."""

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension
        self.model_id = "test:mock-embedder"
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        return [0.1] * self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [[0.1] * self._dim for _ in texts]


def test_routing_split_enum() -> None:
    assert [s.value for s in RoutingSplit] == ["train", "validation", "test"]


def test_query_category_enum() -> None:
    categories = {c.value for c in QueryCategory}
    assert "direct_factual" in categories
    assert "technical_identifier" in categories
    assert "numerical" in categories
    assert "procedural" in categories
    assert "specific_detail" in categories
    assert "configuration" in categories
    assert "multi_step" in categories


def test_retrieval_metrics_validation() -> None:
    valid = RetrievalMetrics(
        hit_at_1=1.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=1.0, recall_at_5=1.0
    )
    assert valid.hit_at_1 == 1.0

    with pytest.raises(ValidationError):
        RetrievalMetrics(hit_at_1=1.5, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=1.0, recall_at_5=1.0)


def test_routing_example_schema_validation() -> None:
    ex = RoutingExample(
        example_id="route_p11_train_0001",
        source_document_id="doc_12345678",
        source_document="architecture.md",
        source_element_ids=["elem_0001"],
        source_chunk_ids=["chunk_0001"],
        generation_method="deterministic_chunk_synthesis",
        query="What is the architecture of RAG_EVAL?",
        query_category=QueryCategory.DIRECT_FACTUAL,
        expected_answer="RAG_EVAL is organized as Python modules with explicit contracts.",
        candidate_retrieval_strategies=["bm25", "dense", "hybrid"],
        oracle_strategy=OracleStrategy.HYBRID,
        generation_seed=42,
        dataset_version="1.0.0",
        split=RoutingSplit.TRAIN,
    )
    assert ex.example_id == "route_p11_train_0001"
    assert ex.split == RoutingSplit.TRAIN

    # Reject whitespace-only query
    with pytest.raises(ValidationError):
        RoutingExample(
            example_id="route_p11_train_0001",
            source_document_id="doc_12345678",
            source_document="architecture.md",
            source_element_ids=["elem_0001"],
            source_chunk_ids=["chunk_0001"],
            generation_method="deterministic_chunk_synthesis",
            query="   ",
            query_category=QueryCategory.DIRECT_FACTUAL,
            expected_answer="Valid answer",
            split=RoutingSplit.TRAIN,
            generation_seed=42,
        )


def test_determine_oracle_strategy_clear_wins() -> None:
    # 1. Clear BM25 win
    bm25_m = RetrievalMetrics(
        hit_at_1=1.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=1.0, recall_at_5=1.0
    )
    dense_m = RetrievalMetrics(
        hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.0, recall_at_5=0.0
    )
    hybrid_m = RetrievalMetrics(
        hit_at_1=0.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=0.5, recall_at_5=1.0
    )

    strategy, is_tie, margin = determine_oracle_strategy(bm25_m, dense_m, hybrid_m)
    assert strategy == OracleStrategy.BM25
    assert not is_tie
    assert margin > 0.0

    # 2. Clear Dense win
    strategy, is_tie, margin = determine_oracle_strategy(dense_m, bm25_m, hybrid_m)
    assert strategy == OracleStrategy.DENSE
    assert not is_tie
    assert margin > 0.0

    # 3. Clear Hybrid win
    hybrid_win_m = RetrievalMetrics(
        hit_at_1=1.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=1.0, recall_at_5=1.0
    )
    bm25_low_m = RetrievalMetrics(
        hit_at_1=0.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=0.5, recall_at_5=1.0
    )
    dense_low_m = RetrievalMetrics(
        hit_at_1=0.0, hit_at_3=0.0, hit_at_5=1.0, mrr_at_5=0.2, recall_at_5=1.0
    )

    strategy, is_tie, margin = determine_oracle_strategy(bm25_low_m, dense_low_m, hybrid_win_m)
    assert strategy == OracleStrategy.HYBRID
    assert not is_tie
    assert margin > 0.0


def test_determine_oracle_strategy_ties_and_ambiguous() -> None:
    perfect = RetrievalMetrics(
        hit_at_1=1.0, hit_at_3=1.0, hit_at_5=1.0, mrr_at_5=1.0, recall_at_5=1.0
    )
    zero = RetrievalMetrics(hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_5=0.0, recall_at_5=0.0)

    # All three succeed equally
    strategy, is_tie, margin = determine_oracle_strategy(perfect, perfect, perfect)
    assert strategy == OracleStrategy.AMBIGUOUS
    assert is_tie
    assert margin == 0.0

    # All three fail
    strategy, is_tie, margin = determine_oracle_strategy(zero, zero, zero)
    assert strategy == OracleStrategy.NO_CLEAR_WINNER
    assert is_tie
    assert margin == 0.0

    # BM25 ties with Hybrid, beating Dense
    strategy, is_tie, margin = determine_oracle_strategy(perfect, zero, perfect)
    assert strategy == OracleStrategy.BM25
    assert is_tie
    assert margin > 0.0


def test_leakage_detection_exact_and_normalized() -> None:
    protected = [
        "How does the evaluation runner coordinate retrieval and generation?",
        "What four primitive types can JSON represent?",
    ]

    # Exact match
    is_leak, match, sim = check_query_leakage(
        "How does the evaluation runner coordinate retrieval and generation?",
        protected,
    )
    assert is_leak
    assert sim == 1.0

    # Case-insensitive & punctuation variation
    is_leak, match, sim = check_query_leakage(
        "WHAT FOUR PRIMITIVE TYPES CAN JSON REPRESENT",
        protected,
    )
    assert is_leak
    assert sim == 1.0

    # Independent query should not leak
    is_leak, match, sim = check_query_leakage(
        "What cloud deployment steps are required for Render web services?",
        protected,
    )
    assert not is_leak
    assert sim < 0.70


def test_leakage_detection_fuzzy_threshold() -> None:
    protected = ["What four primitive types can JSON represent?"]

    # Close paraphrase
    is_leak, match, sim = check_query_leakage(
        "What four primitive types can JSON format represent?",
        protected,
        fuzzy_threshold=0.70,
    )
    assert is_leak
    assert sim >= 0.70


def test_document_split_separation() -> None:
    train_docs = {s.filename for s in DEFAULT_DOCUMENT_SPECS if s.split == RoutingSplit.TRAIN}
    val_docs = {s.filename for s in DEFAULT_DOCUMENT_SPECS if s.split == RoutingSplit.VALIDATION}
    test_docs = {s.filename for s in DEFAULT_DOCUMENT_SPECS if s.split == RoutingSplit.TEST}

    # Zero overlap across partitions
    assert len(train_docs & val_docs) == 0, f"Overlap train/val: {train_docs & val_docs}"
    assert len(train_docs & test_docs) == 0, f"Overlap train/test: {train_docs & test_docs}"
    assert len(val_docs & test_docs) == 0, f"Overlap val/test: {val_docs & test_docs}"

    assert len(train_docs) == 8
    assert len(val_docs) == 5
    assert len(test_docs) == 4
    assert len(train_docs | val_docs | test_docs) == 17


def test_cached_embedder_behavior() -> None:
    mock = DummyTestEmbedder()
    cached = CachedEmbedder(mock)

    # First call embeds
    v1 = cached.embed_query("query 1")
    assert mock.call_count == 1

    # Second call uses cache
    v2 = cached.embed_query("query 1")
    assert mock.call_count == 1
    assert v1 == v2

    # Warm cache in batches
    cached.warm_cache(["query 2", "query 3"], batch_size=2)
    assert mock.call_count == 3  # +2 from batch

    # Lookups are cached
    _ = cached.embed_query("query 2")
    _ = cached.embed_query("query 3")
    assert mock.call_count == 3


def test_generator_reproducibility() -> None:
    # Test reproducibility on a single test doc spec
    small_specs = [
        DocumentSpec(
            "docs/ci-cd.md",
            "ci-cd.md",
            "CI/CD Pipeline",
            RoutingSplit.TEST,
        )
    ]
    gen1 = RoutingDatasetGenerator(document_specs=small_specs, seed=42)
    gen2 = RoutingDatasetGenerator(document_specs=small_specs, seed=42)

    exs1 = gen1.generate()
    exs2 = gen2.generate()

    assert len(exs1) == len(exs2)
    assert len(exs1) > 0
    for e1, e2 in zip(exs1, exs2, strict=False):
        assert e1.example_id == e2.example_id
        assert e1.query == e2.query
        assert e1.expected_answer == e2.expected_answer
        assert e1.query_category == e2.query_category


def test_generated_phase11_dataset_integrity() -> None:
    from scripts.validate_routing_dataset import validate_dataset

    dataset_dir = Path("data/routing/phase11")
    if not dataset_dir.exists():
        pytest.skip("Dataset directory not yet generated.")

    is_valid, diagnostics = validate_dataset(dataset_dir)
    assert is_valid, f"Dataset validation failed: {diagnostics}"
