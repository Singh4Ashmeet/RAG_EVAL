"""Learned sparse retrieval using SPLADE."""

from __future__ import annotations

from .encoder import SparseEncoder, SparseVector
from .retriever import SparseRetriever

__all__ = [
    "SparseEncoder",
    "SparseVector",
    "SparseRetriever",
]
