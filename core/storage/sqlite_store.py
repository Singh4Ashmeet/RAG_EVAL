"""SQLite-backed persistent vector store using exact cosine similarity."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from threading import RLock

from core.errors import VectorStoreError
from core.normalize.schema import Chunk


class SQLiteVectorStore:
    """Persist chunks and vectors in SQLite and rank by exact cosine similarity.

    This backend is intended for local development and small to medium collections.
    Search scans all indexed vectors in Python, so it does not provide ANN-scale
    performance. A database is bound to one embedding model and dimension until
    :meth:`clear` is called.
    """

    def __init__(self, database_path: str | Path = "data/rag_eval.sqlite3") -> None:
        self.database_path = str(database_path)
        self._lock = RLock()
        try:
            if self.database_path != ":memory:":
                Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._initialize_schema()
        except (OSError, sqlite3.Error) as exc:
            raise VectorStoreError(f"Failed to open SQLite vector store: {exc}") from exc

    def replace_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        model_id: str,
        dimension: int,
    ) -> None:
        """Atomically replace a document's indexed chunks and clear stale rows."""
        if not document_id.strip():
            raise VectorStoreError("document_id cannot be empty.")
        if not model_id.strip():
            raise VectorStoreError("model_id cannot be empty.")
        if dimension < 1:
            raise VectorStoreError("Embedding dimension must be positive.")
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts must match.")
        chunk_ids: set[str] = set()
        normalized_vectors: list[list[float]] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            if chunk.document_id != document_id:
                raise VectorStoreError(
                    f"Chunk '{chunk.chunk_id}' belongs to '{chunk.document_id}', "
                    f"not '{document_id}'."
                )
            if chunk.chunk_id in chunk_ids:
                raise VectorStoreError(f"Duplicate chunk id '{chunk.chunk_id}'.")
            chunk_ids.add(chunk.chunk_id)
            normalized_vectors.append(self._validate_vector(vector, dimension))

        with self._lock:
            try:
                with self._connection:
                    metadata = self._connection.execute(
                        "SELECT model_id, dimension FROM vector_store_metadata WHERE id = 1"
                    ).fetchone()
                    if metadata is not None and (
                        metadata["model_id"] != model_id or metadata["dimension"] != dimension
                    ):
                        raise VectorStoreError(
                            "Embedding model or dimension does not match this index; "
                            "clear and rebuild the index before changing embeddings."
                        )

                    if chunks and metadata is None:
                        self._connection.execute(
                            "INSERT INTO vector_store_metadata (id, model_id, dimension) "
                            "VALUES (1, ?, ?)",
                            (model_id, dimension),
                        )

                    self._connection.execute(
                        "DELETE FROM indexed_chunks WHERE document_id = ?", (document_id,)
                    )
                    self._connection.executemany(
                        """INSERT INTO indexed_chunks
                           (chunk_id, document_id, chunk_index, chunk_json, embedding_json)
                           VALUES (?, ?, ?, ?, ?)""",
                        [
                            (
                                chunk.chunk_id,
                                document_id,
                                chunk.chunk_index,
                                chunk.model_dump_json(),
                                json.dumps(vector, allow_nan=False, separators=(",", ":")),
                            )
                            for chunk, vector in zip(chunks, normalized_vectors, strict=True)
                        ],
                    )
            except VectorStoreError:
                raise
            except (sqlite3.Error, ValueError, TypeError) as exc:
                raise VectorStoreError(f"Failed to replace indexed document: {exc}") from exc

    def search(
        self,
        embedding: list[float],
        *,
        model_id: str,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Scan vectors and return exact cosine similarity matches."""
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")
        if not model_id.strip():
            raise VectorStoreError("model_id cannot be empty.")
        if not isinstance(embedding, (list, tuple)) or not embedding:
            raise VectorStoreError("Query embedding must be a non-empty numeric vector.")

        with self._lock:
            try:
                metadata = self._connection.execute(
                    "SELECT model_id, dimension FROM vector_store_metadata WHERE id = 1"
                ).fetchone()
                if metadata is None:
                    self._validate_vector(embedding, len(embedding))
                    return []
                if metadata["model_id"] != model_id:
                    raise VectorStoreError(
                        f"Query model '{model_id}' does not match indexed model "
                        f"'{metadata['model_id']}'."
                    )
                query_vector = self._validate_vector(embedding, metadata["dimension"])
                rows = self._connection.execute(
                    "SELECT chunk_json, embedding_json FROM indexed_chunks"
                ).fetchall()
            except VectorStoreError:
                raise
            except sqlite3.Error as exc:
                raise VectorStoreError(f"Vector search failed: {exc}") from exc

        scored: list[tuple[Chunk, float]] = []
        try:
            for row in rows:
                stored_vector = self._validate_vector(
                    json.loads(row["embedding_json"]), metadata["dimension"]
                )
                score = self._cosine_similarity(query_vector, stored_vector)
                chunk = Chunk.model_validate_json(row["chunk_json"])
                scored.append((chunk, score))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Stored vector data is invalid: {exc}") from exc

        scored.sort(key=lambda item: (-item[1], item[0].chunk_index, item[0].chunk_id))
        return scored[:top_k]

    def delete_document(self, document_id: str) -> int:
        """Remove one document's chunks without changing index model metadata."""
        if not document_id.strip():
            raise VectorStoreError("document_id cannot be empty.")
        with self._lock:
            try:
                with self._connection:
                    cursor = self._connection.execute(
                        "DELETE FROM indexed_chunks WHERE document_id = ?", (document_id,)
                    )
                    return cursor.rowcount
            except sqlite3.Error as exc:
                raise VectorStoreError(
                    f"Failed to delete document from vector store: {exc}"
                ) from exc

    def clear(self) -> None:
        """Delete all vectors and reset model compatibility metadata."""
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute("DELETE FROM indexed_chunks")
                    self._connection.execute("DELETE FROM vector_store_metadata")
            except sqlite3.Error as exc:
                raise VectorStoreError(f"Failed to clear vector store: {exc}") from exc

    def count(self) -> int:
        """Return the number of indexed chunks."""
        with self._lock:
            try:
                row = self._connection.execute("SELECT COUNT(*) FROM indexed_chunks").fetchone()
                return int(row[0])
            except sqlite3.Error as exc:
                raise VectorStoreError(f"Failed to count indexed chunks: {exc}") from exc

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteVectorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS vector_store_metadata (
                       id INTEGER PRIMARY KEY CHECK (id = 1),
                       model_id TEXT NOT NULL,
                       dimension INTEGER NOT NULL CHECK (dimension > 0)
                   )"""
            )
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS indexed_chunks (
                       chunk_id TEXT PRIMARY KEY,
                       document_id TEXT NOT NULL,
                       chunk_index INTEGER NOT NULL,
                       chunk_json TEXT NOT NULL,
                       embedding_json TEXT NOT NULL,
                       UNIQUE (document_id, chunk_index)
                   )"""
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_indexed_chunks_document_id "
                "ON indexed_chunks (document_id)"
            )

    @staticmethod
    def _validate_vector(vector: object, dimension: int) -> list[float]:
        if not isinstance(vector, (list, tuple)) or len(vector) != dimension:
            raise VectorStoreError(f"Embedding must contain exactly {dimension} numeric values.")
        result: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise VectorStoreError("Embedding values must be finite numbers.")
            try:
                numeric = float(value)
            except (OverflowError, ValueError) as exc:
                raise VectorStoreError("Embedding values must be finite numbers.") from exc
            if not math.isfinite(numeric):
                raise VectorStoreError("Embedding values must be finite numbers.")
            result.append(numeric)
        return result

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        left_scale = max(abs(value) for value in left)
        right_scale = max(abs(value) for value in right)
        if left_scale == 0.0 or right_scale == 0.0:
            return 0.0

        scaled_left = [value / left_scale for value in left]
        scaled_right = [value / right_scale for value in right]
        left_norm = math.sqrt(math.fsum(value * value for value in scaled_left))
        right_norm = math.sqrt(math.fsum(value * value for value in scaled_right))
        score = math.fsum(
            (left_value / left_norm) * (right_value / right_norm)
            for left_value, right_value in zip(scaled_left, scaled_right, strict=True)
        )
        # Floating point round-off can move normalized scores just outside the range.
        return min(1.0, max(-1.0, score))
