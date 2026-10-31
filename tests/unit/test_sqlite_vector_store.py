"""Tests for persistent SQLite chunk and vector storage."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.errors import VectorStoreError
from core.normalize.schema import Chunk
from core.storage import SQLiteVectorStore, VectorStore


def make_chunk(chunk_id: str, document_id: str, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=index,
        source_element_ids=[f"{document_id}_e{index}"],
    )


def test_store_is_a_vector_store_implementation(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        assert isinstance(store, VectorStore)


def test_search_orders_cosine_scores_and_persists_chunks(tmp_path: Path) -> None:
    database = tmp_path / "vectors.sqlite3"
    chunks = [
        make_chunk("c0", "doc-a", 0, "x axis"),
        make_chunk("c1", "doc-a", 1, "y axis"),
        make_chunk("c2", "doc-a", 2, "opposite x"),
    ]
    with SQLiteVectorStore(database) as store:
        store.replace_document(
            "doc-a", chunks, [[1, 0], [0, 1], [-1, 0]], model_id="fake-v1", dimension=2
        )
        assert store.count() == 3

    with SQLiteVectorStore(database) as reopened:
        results = reopened.search([1, 0], model_id="fake-v1", top_k=10)
        assert [chunk.chunk_id for chunk, _ in results] == ["c0", "c1", "c2"]
        assert [score for _, score in results] == pytest.approx([1.0, 0.0, -1.0])
        assert results[0][0].source_element_ids == ["doc-a_e0"]


def test_replacing_document_is_idempotent_and_removes_stale_chunks(
    tmp_path: Path,
) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("old-0", "doc-a", 0, "old"), make_chunk("old-1", "doc-a", 1, "stale")],
            [[1, 0], [0, 1]],
            model_id="fake-v1",
            dimension=2,
        )
        replacement = [make_chunk("new-0", "doc-a", 0, "new")]
        store.replace_document("doc-a", replacement, [[1, 0]], model_id="fake-v1", dimension=2)
        store.replace_document("doc-a", replacement, [[1, 0]], model_id="fake-v1", dimension=2)

        results = store.search([1, 0], model_id="fake-v1", top_k=10)
        assert store.count() == 1
        assert [chunk.chunk_id for chunk, _ in results] == ["new-0"]


def test_failed_insert_rolls_back_and_preserves_previous_document(
    tmp_path: Path,
) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        original = make_chunk("original", "doc-a", 0, "original")
        store.replace_document("doc-a", [original], [[1, 0]], model_id="fake-v1", dimension=2)
        with pytest.raises(VectorStoreError, match="UNIQUE constraint failed"):
            store.replace_document(
                "doc-a",
                [
                    make_chunk("replacement-0", "doc-a", 0, "replacement one"),
                    make_chunk("replacement-1", "doc-a", 0, "replacement two"),
                ],
                [[0, 1], [1, 0]],
                model_id="fake-v1",
                dimension=2,
            )
        results = store.search([1, 0], model_id="fake-v1", top_k=10)
        assert [chunk.chunk_id for chunk, _ in results] == ["original"]


def test_model_and_dimension_must_match_until_index_is_cleared(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("c0", "doc-a", 0, "text")],
            [[1, 0]],
            model_id="fake-v1",
            dimension=2,
        )
        with pytest.raises(VectorStoreError, match="does not match"):
            store.search([1, 0], model_id="fake-v2", top_k=1)
        with pytest.raises(VectorStoreError, match="does not match"):
            store.replace_document(
                "doc-b",
                [make_chunk("b0", "doc-b", 0, "text")],
                [[1, 0, 0]],
                model_id="fake-v1",
                dimension=3,
            )

        store.clear()
        store.replace_document(
            "doc-b",
            [make_chunk("b0", "doc-b", 0, "text")],
            [[1, 0, 0]],
            model_id="fake-v2",
            dimension=3,
        )
        assert store.count() == 1


def test_empty_replacement_removes_stale_document_chunks(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("c0", "doc-a", 0, "text")],
            [[1, 0]],
            model_id="fake-v1",
            dimension=2,
        )
        store.replace_document("doc-a", [], [], model_id="fake-v1", dimension=2)
        assert store.count() == 0
        assert store.search([1, 0], model_id="fake-v1", top_k=1) == []


def test_deletion_returns_count_and_keeps_other_documents(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("a0", "doc-a", 0, "a")],
            [[1, 0]],
            model_id="fake-v1",
            dimension=2,
        )
        store.replace_document(
            "doc-b",
            [make_chunk("b0", "doc-b", 0, "b")],
            [[0, 1]],
            model_id="fake-v1",
            dimension=2,
        )
        assert store.delete_document("doc-a") == 1
        assert store.delete_document("doc-a") == 0
        results = store.search([0, 1], model_id="fake-v1", top_k=2)
        assert [chunk.chunk_id for chunk, _ in results] == ["b0"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(tmp_path: Path, top_k: int) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        with pytest.raises(VectorStoreError, match="top_k"):
            store.search([1], model_id="fake", top_k=top_k)


def test_empty_store_still_rejects_malformed_query_vector(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        with pytest.raises(VectorStoreError, match="non-empty"):
            store.search([], model_id="fake", top_k=1)


@pytest.mark.parametrize(
    "vector",
    [[], [1], [float("nan"), 0], [float("inf"), 0], [True, 0], ["1", 0]],
)
def test_invalid_vectors_are_rejected(tmp_path: Path, vector: list[object]) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        with pytest.raises(VectorStoreError):
            store.replace_document(
                "doc-a",
                [make_chunk("c0", "doc-a", 0, "text")],
                [vector],  # type: ignore[list-item]
                model_id="fake-v1",
                dimension=2,
            )


def test_zero_norm_similarity_is_defined_as_zero(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("c0", "doc-a", 0, "zero")],
            [[0, 0]],
            model_id="fake-v1",
            dimension=2,
        )
        assert store.search([0, 0], model_id="fake-v1", top_k=1)[0][1] == 0.0
        assert math.isfinite(store.search([1, 0], model_id="fake-v1", top_k=1)[0][1])


def test_cosine_similarity_remains_stable_for_large_finite_values(tmp_path: Path) -> None:
    with SQLiteVectorStore(tmp_path / "vectors.sqlite3") as store:
        store.replace_document(
            "doc-a",
            [make_chunk("c0", "doc-a", 0, "large vector")],
            [[1e308, 1e308]],
            model_id="fake-v1",
            dimension=2,
        )
        result = store.search([1e308, 1e308], model_id="fake-v1", top_k=1)
        assert result[0][1] == pytest.approx(1.0)
