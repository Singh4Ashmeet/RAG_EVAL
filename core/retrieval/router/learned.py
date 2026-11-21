"""Learned query router with confidence gating for adaptive retrieval strategy selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from core.errors import RetrievalError
from core.retrieval.router.router import RouteDecision, RoutingStrategy


@dataclass(frozen=True)
class LearnedRouteDecision:
    """Routing decision produced by the learned classifier."""

    strategy: RoutingStrategy
    confidence: float
    raw_prediction: RoutingStrategy
    fallback: bool
    probabilities: dict[str, float]
    model_version: str
    reason: str

    def to_route_decision(self) -> RouteDecision:
        """Convert to base RouteDecision for backwards compatibility."""
        return RouteDecision(
            strategy=self.strategy,
            confidence=self.confidence,
            features_used={"probabilities": self.probabilities, "fallback": self.fallback},
            reasoning=self.reason,
        )


class LearnedQueryRouter:
    """Lightweight classifier predicting BM25, Dense, or Hybrid retrieval.

    Uses a TF-IDF + Logistic Regression pipeline with confidence gating. When
    prediction confidence falls below the specified threshold, the router
    falls back conservatively to the production baseline (Hybrid).
    """

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
    MODEL_VERSION: str = "phase12-v1.0.0"

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.pipeline = pipeline
        self.confidence_threshold = confidence_threshold
        self.model_version = model_version

    @classmethod
    def train(
        cls,
        queries: list[str],
        labels: list[str],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int = 1000,
        c_param: float = 5.0,
        random_state: int = 42,
    ) -> LearnedQueryRouter:
        """Train a lightweight Logistic Regression classifier on query texts."""
        valid_labels = {"bm25", "dense", "hybrid"}
        cleaned_queries: list[str] = []
        cleaned_labels: list[str] = []

        for q, label in zip(queries, labels, strict=False):
            if label.lower() in valid_labels and q.strip():
                cleaned_queries.append(q.strip())
                cleaned_labels.append(label.lower())

        if not cleaned_queries:
            raise RetrievalError("No valid training examples provided for learned router.")

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c_param,
                        random_state=random_state,
                        max_iter=300,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(cleaned_queries, cleaned_labels)
        return cls(
            pipeline=pipeline,
            confidence_threshold=confidence_threshold,
        )

    def route(
        self,
        query: str,
        threshold: float | None = None,
    ) -> LearnedRouteDecision:
        """Predict retrieval strategy with confidence gating."""
        if self.pipeline is None:
            raise RetrievalError("LearnedQueryRouter has not been trained or loaded.")

        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")

        active_threshold = threshold if threshold is not None else self.confidence_threshold

        clf = self.pipeline.named_steps["clf"]
        classes = [str(c) for c in clf.classes_]
        probs = self.pipeline.predict_proba([query.strip()])[0]

        prob_dict: dict[str, float] = {
            cls_name: float(round(p, 4)) for cls_name, p in zip(classes, probs, strict=False)
        }

        best_idx = int(probs.argmax())
        raw_pred_str = classes[best_idx]
        confidence = float(probs[best_idx])
        raw_strategy = RoutingStrategy(raw_pred_str)

        # Confidence gating: fallback to Hybrid if below threshold
        if confidence < active_threshold:
            strategy = RoutingStrategy.HYBRID
            fallback = True
            reason = (
                f"Confidence {confidence:.3f} below threshold {active_threshold:.3f}; "
                f"fallback to baseline Hybrid (raw prediction: {raw_pred_str})"
            )
        else:
            strategy = raw_strategy
            fallback = False
            reason = (
                f"High confidence ({confidence:.3f} >= {active_threshold:.3f}): "
                f"selected {strategy.value}"
            )

        return LearnedRouteDecision(
            strategy=strategy,
            confidence=round(confidence, 4),
            raw_prediction=raw_strategy,
            fallback=fallback,
            probabilities=prob_dict,
            model_version=self.model_version,
            reason=reason,
        )

    def predict_strategy(self, query: str) -> RoutingStrategy:
        """Raw prediction without gating."""
        decision = self.route(query, threshold=0.0)
        return decision.raw_prediction

    def save(self, model_dir: Path | str, metadata: dict[str, Any] | None = None) -> None:
        """Serialize pipeline and metadata to disk."""
        if self.pipeline is None:
            raise RetrievalError("Cannot save an uninitialized router pipeline.")

        target_dir = Path(model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "model.joblib"
        joblib.dump(self.pipeline, model_path)

        meta = {
            "model_version": self.model_version,
            "confidence_threshold": self.confidence_threshold,
            "classes": [str(c) for c in self.pipeline.named_steps["clf"].classes_],
            "feature_config": {
                "vectorizer": "TfidfVectorizer",
                "ngram_range": list(self.pipeline.named_steps["tfidf"].ngram_range),
                "max_features": self.pipeline.named_steps["tfidf"].max_features,
            },
            "classifier_config": {
                "algorithm": "LogisticRegression",
                "C": float(self.pipeline.named_steps["clf"].C),
            },
            "extra_metadata": metadata or {},
        }
        meta_path = target_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path | str) -> LearnedQueryRouter:
        """Load a serialized router pipeline and metadata from disk."""
        target_dir = Path(model_dir)
        model_path = target_dir / "model.joblib"
        meta_path = target_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        pipeline = joblib.load(model_path)
        confidence_threshold = cls.DEFAULT_CONFIDENCE_THRESHOLD
        model_version = cls.MODEL_VERSION

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                confidence_threshold = float(
                    meta.get("confidence_threshold", cls.DEFAULT_CONFIDENCE_THRESHOLD)
                )
                model_version = str(meta.get("model_version", cls.MODEL_VERSION))
            except Exception:
                pass

        return cls(
            pipeline=pipeline,
            confidence_threshold=confidence_threshold,
            model_version=model_version,
        )
