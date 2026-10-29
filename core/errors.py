"""Shared domain exceptions for RAG_EVAL."""


class RAGEvalError(Exception):
    """Base exception for all RAG_EVAL domain errors."""


class ParserError(RAGEvalError):
    """Raised when document parsing fails."""


class ChunkerError(RAGEvalError):
    """Raised when document chunking fails."""


class EmbedderError(RAGEvalError):
    """Raised when text embedding fails."""


class RetrievalError(RAGEvalError):
    """Raised when document retrieval or reranking fails."""
