"""Dense retrieval using an embedder and persistent vector store."""

from __future__ import annotations

from collections.abc import Sequence

from core.embed.base import Embedder
from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
from core.storage.base import VectorStore


class DenseRetriever:
    """Index chunks and retrieve them by exact dense cosine similarity."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        model_id: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        configured_model_id = model_id or getattr(embedder, "model_id", None)
        if not isinstance(configured_model_id, str) or not configured_model_id.strip():
            raise RetrievalError("A stable model_id is required on the embedder or DenseRetriever.")
        self.model_id = configured_model_id.strip()

    def index_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        """Embed and atomically replace one document's indexed chunks."""
        if not document_id.strip():
            raise RetrievalError("document_id cannot be empty.")
        chunk_list = list(chunks)
        if any(chunk.document_id != document_id for chunk in chunk_list):
            raise RetrievalError("All chunks must belong to the document being indexed.")

        if not chunk_list:
            self.vector_store.delete_document(document_id)
            return

        try:
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunk_list])
            dimension = len(vectors[0]) if vectors else 0
            if dimension < 1 or any(len(vector) != dimension for vector in vectors):
                raise RetrievalError("Embedder returned empty or inconsistent vector dimensions.")
            if len(vectors) != len(chunk_list):
                raise RetrievalError(
                    f"Embedder returned {len(vectors)} vectors for {len(chunk_list)} chunks."
                )
            self.vector_store.replace_document(
                document_id,
                chunk_list,
                vectors,
                model_id=self.model_id,
                dimension=dimension,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Failed to index document '{document_id}': {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Delete all indexed chunks for a document."""
        return self.vector_store.delete_document(document_id)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve the top matching chunks with cosine similarity scores in [-1, 1]."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")

        try:
            embedding = self.embedder.embed_query(query)
            matches = self.vector_store.search(
                embedding,
                model_id=self.model_id,
                top_k=top_k,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Dense retrieval failed: {exc}") from exc

        return [
            SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.DENSE,
            )
            for rank, (chunk, score) in enumerate(matches)
        ]
