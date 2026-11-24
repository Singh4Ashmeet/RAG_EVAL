"""Embedding interfaces and implementations."""

from core.embed.base import Embedder
from core.embed.ollama_embedder import OllamaEmbedder
from core.embed.openai_compatible_embedder import OpenAICompatibleEmbedder

__all__ = ["Embedder", "OllamaEmbedder", "OpenAICompatibleEmbedder"]
