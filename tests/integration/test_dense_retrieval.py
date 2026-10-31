"""End-to-end dense retrieval tests with a deterministic local fake embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval import DenseRetriever, RetrievalStrategy
from core.storage import SQLiteVectorStore


class LexicalVectorEmbedder:
    """Map simple test vocabulary onto orthogonal deterministic vectors."""

    model_id = "test:lexical-v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [float("cat" in lowered), float("ocean" in lowered), 0.0]

    @property
    def dimension(self) -> int:
        return 3


class UnidentifiedEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]

    @property
    def dimension(self) -> int:
        return 1


def make_chunk(chunk_id: str, doc_id: str, index: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id=doc_id, chunk_index=index, text=text)


def test_index_reopen_query_and_delete_document(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.sqlite3"
    first_document = [
        make_chunk("a0", "doc-a", 0, "Cats are curious mammals."),
        make_chunk("a1", "doc-a", 1, "The Pacific Ocean is vast."),
    ]
    second_document = [make_chunk("b0", "doc-b", 0, "A cat naps in sunlight.")]

    with SQLiteVectorStore(database) as store:
        retriever = DenseRetriever(LexicalVectorEmbedder(), store)
        retriever.index_document("doc-a", first_document)
        retriever.index_document("doc-b", second_document)
        matches = retriever.retrieve("Tell me about cats", top_k=2)
        assert [result.chunk.chunk_id for result in matches] == ["a0", "b0"]
        assert matches[0].score == pytest.approx(1.0)
        assert matches[0].rank == 0
        assert matches[0].strategy == RetrievalStrategy.DENSE

    with SQLiteVectorStore(database) as reopened:
        retriever = DenseRetriever(LexicalVectorEmbedder(), reopened)
        assert [r.chunk.chunk_id for r in retriever.retrieve("ocean")] == ["a1", "a0", "b0"]
        retriever.index_document("doc-a", [make_chunk("a2", "doc-a", 0, "Cat facts updated.")])
        assert reopened.count() == 2
        assert {r.chunk.chunk_id for r in retriever.retrieve("cat")} == {"a2", "b0"}
        assert retriever.delete_document("doc-a") == 1
        assert [r.chunk.chunk_id for r in retriever.retrieve("cat")] == ["b0"]


def test_empty_document_indexing_removes_existing_rows(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "knowledge.sqlite3") as store:
        retriever = DenseRetriever(LexicalVectorEmbedder(), store)
        retriever.index_document("doc-a", [make_chunk("a0", "doc-a", 0, "cat")])
        retriever.index_document("doc-a", [])
        assert store.count() == 0


@pytest.mark.parametrize("query,top_k", [(" ", 1), ("cat", 0)])
def test_retrieval_validates_query_and_top_k(tmp_path: Path, query: str, top_k: int) -> None:
    with SQLiteVectorStore(tmp_path / "knowledge.sqlite3") as store:
        retriever = DenseRetriever(LexicalVectorEmbedder(), store)
        with pytest.raises(RetrievalError):
            retriever.retrieve(query, top_k=top_k)


def test_indexing_rejects_chunks_from_another_document(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "knowledge.sqlite3") as store:
        retriever = DenseRetriever(LexicalVectorEmbedder(), store)
        with pytest.raises(RetrievalError, match="belong to"):
            retriever.index_document("doc-a", [make_chunk("b0", "doc-b", 0, "cat")])


def test_custom_embedder_must_provide_or_receive_a_model_id(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "knowledge.sqlite3") as store:
        with pytest.raises(RetrievalError, match="stable model_id"):
            DenseRetriever(UnidentifiedEmbedder(), store)
        retriever = DenseRetriever(UnidentifiedEmbedder(), store, model_id="custom:v1")
        assert retriever.model_id == "custom:v1"
