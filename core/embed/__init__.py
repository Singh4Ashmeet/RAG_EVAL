"""Embedding interfaces and implementations."""

from core.embed.base import Embedder
from core.embed.ollama_embedder import OllamaEmbedder

__all__ = ["Embedder", "OllamaEmbedder"]
