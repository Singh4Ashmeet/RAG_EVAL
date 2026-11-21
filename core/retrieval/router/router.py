"""Query router for adaptive retrieval strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.retrieval.router.features import QueryFeatures, extract_features


class RoutingStrategy(StrEnum):
    """Supported retrieval strategies for routing."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"  # Dense + BM25


@dataclass(frozen=True)
class RouteDecision:
    """Decision made by the router for a query."""

    strategy: RoutingStrategy
    confidence: float
    features_used: dict[str, Any]
    reasoning: str


class QueryRouter:
    """
    Deterministic query router that selects retrieval strategy based on query features.

    The router uses heuristic rules based on query features to select the most
    appropriate retrieval strategy. It has a conservative fallback to the
    production baseline (Dense + BM25 RRF) when uncertain.
    """

    # Confidence threshold for routing decisions
    CONFIDENCE_THRESHOLD: float = 0.7

    # Strategy mapping for router decisions
    STRATEGY_RETRIEVER_MAP: dict[str, str] = {
        "bm25": "bm25",
        "dense": "dense",
        "hybrid": "hybrid",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.confidence_threshold = confidence_threshold

    def route(self, query: str) -> RouteDecision:
        """
        Determine the optimal retrieval strategy for a query.

        Args:
            query: The user query to route.

        Returns:
            RouteDecision with the selected strategy and confidence.
        """
        features = QueryFeatures.from_query(query)
        extract_features(query)

        # Route based on heuristic rules
        strategy, confidence, reasoning = self._determine_strategy(features)

        # Apply confidence threshold - fall back to hybrid if uncertain
        if confidence < self.confidence_threshold:
            strategy = RoutingStrategy.HYBRID
            confidence = max(confidence, 0.5)
            reasoning += " (fallback to hybrid due to low confidence)"

        return RouteDecision(
            strategy=strategy,
            confidence=confidence,
            features_used=self._get_relevant_features(features),
            reasoning=reasoning,
        )

    def _determine_strategy(self, features: QueryFeatures) -> tuple[RoutingStrategy, float, str]:
        """Determine the best strategy based on query features."""

        # Rule 1: Lexical exact queries (exact identifiers, versions, hashes, UUIDs) -> BM25
        if (
            features.contains_uuid
            or features.contains_hash
            or features.contains_version
            or features.contains_hex
        ):
            return (
                RoutingStrategy.BM25,
                0.9,
                "Lexical exact match indicators (UUID/hash/version/hex) favor BM25",
            )

        # Rule 2: Error codes and exact technical terms -> BM25
        if features.contains_error_code:
            return RoutingStrategy.BM25, 0.85, "Error code queries favor BM25 for exact matching"

        # Rule 3: Exact lexical queries with quotes, acronyms -> BM25
        if features.contains_quotes or features.contains_acronym:
            return (
                RoutingStrategy.BM25,
                0.8,
                "Quoted terms or acronyms favor BM25 for exact matching",
            )

        # Rule 3: Numerical/range queries with exact values -> BM25
        if features.contains_number and not self._is_semantic_question(features):
            return RoutingStrategy.BM25, 0.75, "Numerical exact queries favor BM25"

        # Rule 4: Semantic/natural language questions -> Dense
        if features.ends_with_question and features.starts_with_wh:
            return (
                RoutingStrategy.DENSE,
                0.8,
                "Natural language question favors dense semantic retrieval",
            )

        # Rule 5: How-to / why questions -> Dense
        if features.starts_with_wh and features.ends_with_question:
            return RoutingStrategy.DENSE, 0.75, "How/why questions favor dense semantic retrieval"

        # Rule 6: Technical identifier queries -> Hybrid (both semantic and exact)
        if features.contains_technical_term:
            return (
                RoutingStrategy.HYBRID,
                0.7,
                "Technical identifier queries benefit from both dense and BM25",
            )

        # Rule 5: Table/structured data queries -> Hybrid
        if features.contains_table_terms:
            return (
                RoutingStrategy.HYBRID,
                0.7,
                "Table/structured queries benefit from both retrieval modes",
            )

        # Rule 6: Provenance/source tracking -> Hybrid
        if features.contains_source_terms:
            return (
                RoutingStrategy.HYBRID,
                0.7,
                "Provenance queries benefit from both retrieval modes",
            )

        # Rule 7: Comparison queries -> Hybrid
        if features.contains_comparison:
            return (
                RoutingStrategy.HYBRID,
                0.7,
                "Comparison queries benefit from both retrieval modes",
            )

        # Default: Conservative fallback to Hybrid (production baseline)
        return RoutingStrategy.HYBRID, 0.5, "Default fallback to hybrid (production baseline)"

    def _is_semantic_question(self, features: QueryFeatures) -> bool:
        """Detect if the query is a semantic/natural language question."""
        return features.ends_with_question and features.starts_with_wh

    def _get_relevant_features(self, features: QueryFeatures) -> dict[str, Any]:
        """Extract relevant features for the decision."""
        return {
            "contains_technical_term": features.contains_technical_term,
            "contains_number": features.contains_number,
            "contains_version": features.contains_version,
            "contains_uuid": features.contains_uuid,
            "contains_hash": features.contains_hash,
            "contains_error_code": features.contains_error_code,
            "contains_quotes": features.contains_quotes,
            "contains_acronym": features.contains_acronym,
            "ends_with_question": features.ends_with_question,
            "starts_with_wh": features.starts_with_wh,
            "contains_table_terms": features.contains_table_terms,
            "contains_source_terms": features.contains_source_terms,
            "contains_comparison": features.contains_comparison,
            "technical_token_ratio": features.technical_token_ratio,
        }
