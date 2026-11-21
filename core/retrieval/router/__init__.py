"""Query routing for adaptive retrieval strategy selection."""

from __future__ import annotations

from .features import QueryFeatures, extract_features
from .router import QueryRouter, RouteDecision, RoutingStrategy

__all__ = [
    "QueryFeatures",
    "extract_features",
    "QueryRouter",
    "RoutingStrategy",
    "RouteDecision",
]
