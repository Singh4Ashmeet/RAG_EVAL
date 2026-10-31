"""Validation and digest tests for JSONL evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import EvaluationError
from eval.datasets import EvaluationCase, EvaluationDataset, load_jsonl_dataset


def record(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "query_id": "q-1",
        "query": "What color is the sky?",
        "relevant_chunk_ids": ["chunk-1"],
        "reference_answer": "Blue.",
        "metadata": {"topic": "weather"},
    }
    values.update(overrides)
    return values


def test_jsonl_loader_validates_records_and_hashes_exact_bytes(tmp_path: Path) -> None:
    contents = (
        json.dumps(record(), ensure_ascii=False)
        + "\n\n"
        + json.dumps(record(query_id="q-2", relevant_chunk_ids=[]), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    dataset_path = tmp_path / "weather.jsonl"
    dataset_path.write_bytes(contents)

    dataset = load_jsonl_dataset(dataset_path)

    assert dataset.dataset_id == "weather"
    assert dataset.sha256 == hashlib.sha256(contents).hexdigest()
    assert len(dataset.cases) == 2
    assert dataset.cases[0].metadata == {"topic": "weather"}
    assert dataset.cases[1].reference_answer == "Blue."


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Invalid evaluation dataset"):
        load_jsonl_dataset(path)


def test_invalid_json_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(record()) + "\n{broken}\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match=r"broken.jsonl:2"):
        load_jsonl_dataset(path)


def test_invalid_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(EvaluationError, match="must be UTF-8"):
        load_jsonl_dataset(path)


def test_missing_dataset_is_reported(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="Failed to read"):
        load_jsonl_dataset(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "values",
    [
        {"query_id": "q", "query": "question"},
        {"query_id": "q", "query": "question", "relevant_chunk_ids": ["x", "x"]},
        {"query_id": "q", "query": "question", "reference_answer": "  "},
    ],
)
def test_case_requires_valid_ground_truth_and_unique_ids(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(values)


def test_dataset_query_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        EvaluationDataset(
            dataset_id="dup",
            sha256="a" * 64,
            cases=[
                EvaluationCase.model_validate(record()),
                EvaluationCase.model_validate(record()),
            ],
        )
