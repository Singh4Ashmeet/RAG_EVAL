"""Shared domain exceptions for RAG_EVAL."""


class RAGEvalError(Exception):
    """Base exception for all RAG_EVAL domain errors."""


class ParserError(RAGEvalError):
    """Raised when document parsing fails."""


class UnsupportedFormatError(ParserError):
    """Raised when an input file format is not supported."""


class CorruptedFileError(ParserError):
    """Raised when a supported input file cannot be opened or decoded."""


class ChunkerError(RAGEvalError):
    """Raised when document chunking fails."""


class EmbedderError(RAGEvalError):
    """Raised when text embedding fails."""


class VectorStoreError(RAGEvalError):
    """Raised when vector storage or index compatibility checks fail."""


class RetrievalError(RAGEvalError):
    """Raised when document retrieval or reranking fails."""


class LLMError(RAGEvalError):
    """Raised when a language model request fails or returns an invalid response."""


class GenerationError(RAGEvalError):
    """Raised when a RAG answer cannot be generated from retrieved context."""


class EvaluationError(RAGEvalError):
    """Raised when evaluation data or report handling is invalid."""
