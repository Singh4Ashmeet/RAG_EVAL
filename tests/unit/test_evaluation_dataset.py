"""Validation and digest tests for JSONL evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import EvaluationError
from eval.datasets import (
    AnnotationProvenance,
    AnnotationStatus,
    Answerability,
    CorpusTrack,
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    QuestionCategory,
    load_jsonl_dataset,
)


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


def test_legacy_record_defaults_to_unverified_and_undetermined() -> None:
    case = EvaluationCase.model_validate(record())
    assert case.status == AnnotationStatus.UNVERIFIED
    assert case.answerability == Answerability.UNDETERMINED
    assert case.category is None
    assert case.difficulty is None
    assert case.provenance is None
    assert case.track is None


def test_verified_case_requires_provenance() -> None:
    with pytest.raises(ValidationError, match="must include verification provenance"):
        EvaluationCase.model_validate(
            record(
                status="verified",
                provenance=None,
            )
        )


def test_verified_case_with_valid_provenance() -> None:
    prov = AnnotationProvenance(
        annotator="engineer-1",
        verified_at=datetime.now(UTC),
        source_documents=["doc_a.md"],
        verification_notes="Manually verified against doc_a.",
        evidence_references=["doc_a_e0001"],
    )
    case = EvaluationCase.model_validate(
        record(
            status=AnnotationStatus.VERIFIED,
            answerability=Answerability.ANSWERABLE,
            category=QuestionCategory.DIRECT_FACTUAL,
            difficulty=Difficulty.EASY,
            provenance=prov,
            track=CorpusTrack.TRACK_A,
        )
    )
    assert case.status == AnnotationStatus.VERIFIED
    assert case.answerability == Answerability.ANSWERABLE
    assert case.provenance is not None
    assert case.provenance.annotator == "engineer-1"
    assert case.track == CorpusTrack.TRACK_A


def test_unanswerable_case_permits_empty_ground_truth() -> None:
    case = EvaluationCase.model_validate(
        {
            "query_id": "q-unanswerable",
            "query": "What is the secret recipe?",
            "answerability": "unanswerable",
            "category": "unanswerable",
            "status": "unverified",
        }
    )
    assert case.answerability == Answerability.UNANSWERABLE
    assert case.relevant_chunk_ids == []
    assert case.relevant_element_ids == []
    assert case.reference_answer is None


def test_extra_fields_forbidden_on_case_and_provenance() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(record(extra_field="not_allowed"))

    with pytest.raises(ValidationError):
        AnnotationProvenance(
            annotator="e1",
            verified_at=datetime.now(UTC),
            source_documents=["d.md"],
            extra_provenance="forbidden",  # type: ignore[call-arg]
        )


def test_case_models_are_frozen() -> None:
    case = EvaluationCase.model_validate(record())
    with pytest.raises(ValidationError):
        case.query = "new question"  # type: ignore[misc]


def test_provenance_rejects_empty_fields_and_duplicate_docs() -> None:
    with pytest.raises(ValidationError):
        AnnotationProvenance(
            annotator="   ",
            verified_at=datetime.now(UTC),
            source_documents=["doc.md"],
        )

    with pytest.raises(ValidationError, match="duplicates"):
        AnnotationProvenance(
            annotator="valid",
            verified_at=datetime.now(UTC),
            source_documents=["doc.md", "doc.md"],
        )
