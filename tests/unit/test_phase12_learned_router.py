"""Unit tests for Phase 12 Learned Query Router and Confidence Gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import RetrievalError
from core.retrieval.router.learned import LearnedQueryRouter, LearnedRouteDecision
from core.retrieval.router.router import RouteDecision, RoutingStrategy


@pytest.fixture
def sample_training_data() -> tuple[list[str], list[str]]:
    queries = [
        "exact keyword match search syntax error 404",
        "find error code 500 in logs",
        "what is the definition of architecture concepts",
        "explain semantic concepts and meanings of system design",
        "hybrid overview with both code and high level meaning",
        "detailed summary of system operations and procedures",
    ]
    labels = [
        "bm25",
        "bm25",
        "dense",
        "dense",
        "hybrid",
        "hybrid",
    ]
    return queries, labels


@pytest.fixture
def trained_router(sample_training_data: tuple[list[str], list[str]]) -> LearnedQueryRouter:
    queries, labels = sample_training_data
    return LearnedQueryRouter.train(
        queries=queries,
        labels=labels,
        confidence_threshold=0.60,
    )


class TestLearnedQueryRouterTraining:
    def test_train_success(self, sample_training_data: tuple[list[str], list[str]]) -> None:
        queries, labels = sample_training_data
        router = LearnedQueryRouter.train(queries, labels)
        assert router.pipeline is not None
        assert router.confidence_threshold == LearnedQueryRouter.DEFAULT_CONFIDENCE_THRESHOLD
        assert router.model_version == LearnedQueryRouter.MODEL_VERSION

    def test_train_empty_dataset_raises(self) -> None:
        with pytest.raises(RetrievalError, match="No valid training examples"):
            LearnedQueryRouter.train([], [])

    def test_train_invalid_labels_ignored(self) -> None:
        queries = ["query1", "query2"]
        labels = ["invalid_strategy", "another_invalid"]
        with pytest.raises(RetrievalError, match="No valid training examples"):
            LearnedQueryRouter.train(queries, labels)

    def test_uninitialized_router_raises(self) -> None:
        router = LearnedQueryRouter()
        with pytest.raises(RetrievalError, match="not been trained or loaded"):
            router.route("test query")


class TestLearnedQueryRouterInference:
    def test_route_returns_decision(self, trained_router: LearnedQueryRouter) -> None:
        decision = trained_router.route("find error code in logs")
        assert isinstance(decision, LearnedRouteDecision)
        expected_strategies = {RoutingStrategy.BM25, RoutingStrategy.DENSE, RoutingStrategy.HYBRID}
        assert decision.strategy in expected_strategies
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.model_version == trained_router.model_version
        assert set(decision.probabilities.keys()) == {"bm25", "dense", "hybrid"}

    def test_empty_query_raises(self, trained_router: LearnedQueryRouter) -> None:
        with pytest.raises(RetrievalError, match="non-empty string"):
            trained_router.route("   ")

    def test_confidence_fallback_mechanism(self, trained_router: LearnedQueryRouter) -> None:
        # Forcing a 1.0 threshold guarantees fallback to HYBRID
        decision = trained_router.route("search query", threshold=1.0)
        assert decision.strategy == RoutingStrategy.HYBRID
        assert decision.fallback is True
        assert "fallback to baseline Hybrid" in decision.reason

    def test_high_confidence_no_fallback(self, trained_router: LearnedQueryRouter) -> None:
        # A 0.0 threshold guarantees NO fallback
        decision = trained_router.route("find error code 500 in logs", threshold=0.0)
        assert decision.fallback is False
        assert decision.strategy == decision.raw_prediction
        assert "High confidence" in decision.reason

    def test_predict_strategy_raw(self, trained_router: LearnedQueryRouter) -> None:
        strat = trained_router.predict_strategy("find error code 500 in logs")
        assert isinstance(strat, RoutingStrategy)

    def test_to_route_decision_conversion(self, trained_router: LearnedQueryRouter) -> None:
        decision = trained_router.route("explain semantic concepts")
        base_decision = decision.to_route_decision()
        assert isinstance(base_decision, RouteDecision)
        assert base_decision.strategy == decision.strategy
        assert base_decision.confidence == decision.confidence
        assert "probabilities" in base_decision.features_used
        assert "fallback" in base_decision.features_used


class TestLearnedQueryRouterSerialization:
    def test_save_and_load(self, trained_router: LearnedQueryRouter, tmp_path: Path) -> None:
        save_dir = tmp_path / "router_model"
        custom_meta = {"training_records": 6, "eval_accuracy": 1.0}
        trained_router.save(save_dir, metadata=custom_meta)

        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "metadata.json").exists()

        # Check metadata json content
        meta_content = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta_content["confidence_threshold"] == trained_router.confidence_threshold
        assert meta_content["extra_metadata"] == custom_meta

        # Load and verify behavior parity
        loaded_router = LearnedQueryRouter.load(save_dir)
        assert loaded_router.confidence_threshold == trained_router.confidence_threshold
        assert loaded_router.model_version == trained_router.model_version

        query = "find error code in logs"
        d1 = trained_router.route(query)
        d2 = loaded_router.route(query)

        assert d1.strategy == d2.strategy
        assert d1.confidence == d2.confidence
        assert d1.raw_prediction == d2.raw_prediction
        assert d1.fallback == d2.fallback

    def test_load_missing_model_raises(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="not found"):
            LearnedQueryRouter.load(missing_dir)

    def test_save_uninitialized_raises(self, tmp_path: Path) -> None:
        uninit = LearnedQueryRouter()
        with pytest.raises(RetrievalError, match="Cannot save an uninitialized router"):
            uninit.save(tmp_path)


class TestProtectedDatasetIsolation:
    def test_protected_tracks_not_in_training_corpus(self) -> None:
        """Verify that training dataset partition does not contain protected query IDs."""
        train_path = Path("data/routing/phase11/train.jsonl")
        if not train_path.exists():
            pytest.skip("Phase 11 train.jsonl not present")

        protected_ids: set[str] = set()
        for track in ["track_a", "track_b", "track_c", "track_d"]:
            p = Path(f"eval/golden/{track}.jsonl")
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            protected_ids.add(json.loads(line)["query_id"])

        assert len(protected_ids) == 121

        train_ids: set[str] = set()
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_ids.add(json.loads(line)["example_id"])

        leakage = train_ids.intersection(protected_ids)
        assert len(leakage) == 0, f"Found leaked protected query IDs in train set: {leakage}"
