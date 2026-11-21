"""Automated quality and integrity tests for all golden evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets.schema import EvaluationCase
from eval.datasets.validation import DatasetValidator

GOLDEN_DIR = Path("eval/golden")
CORPORA_DIR = Path("eval/corpora")
TRACKS = ["track_a", "track_b", "track_c", "track_d"]


@pytest.mark.parametrize("track", TRACKS)
def test_golden_file_exists_and_non_empty(track: str) -> None:
    path = GOLDEN_DIR / f"{track}.jsonl"
    assert path.is_file(), f"Missing golden dataset: {path}"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) > 0, f"Golden dataset is empty: {path}"


@pytest.mark.parametrize("track", TRACKS)
def test_golden_cases_conform_to_schema(track: str) -> None:
    path = GOLDEN_DIR / f"{track}.jsonl"
    cases: list[EvaluationCase] = []
    with open(path, encoding="utf-8") as f:
        for _idx, line in enumerate(f):
            data = json.loads(line)
            case = EvaluationCase.model_validate(data)
            cases.append(case)

    # Check query_id uniqueness
    query_ids = [c.query_id for c in cases]
    assert len(query_ids) == len(set(query_ids)), f"Duplicate query IDs found in {track}"

    # Check track tag consistency
    for c in cases:
        assert c.track.value == track, f"Mismatched track tag '{c.track}' in {track}"


def test_track_a_control_integrity() -> None:
    # Track A must preserve all 14 historical control queries
    path = GOLDEN_DIR / "track_a.jsonl"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 14, f"Track A control must contain exactly 14 queries, found {len(lines)}"


def test_track_b_size_and_characteristics() -> None:
    path = GOLDEN_DIR / "track_b.jsonl"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 50, f"Track B expected >= 50 queries, found {len(lines)}"


def test_track_c_multi_format_coverage() -> None:
    path = GOLDEN_DIR / "track_c.jsonl"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [EvaluationCase.model_validate(json.loads(line)) for line in lines]
    assert len(cases) >= 25, f"Track C expected >= 25 queries, found {len(cases)}"

    # Check that table-requiring queries and adversarial queries exist
    table_cases = [c for c in cases if c.requires_table]
    assert len(table_cases) >= 5, "Track C should test table-lookup queries"

    adversarial_cases = [c for c in cases if c.category.value == "adversarial"]
    assert len(adversarial_cases) >= 5, "Track C should test adversarial / near-duplicate queries"


def test_track_d_unanswerable_characteristics() -> None:
    path = GOLDEN_DIR / "track_d.jsonl"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [EvaluationCase.model_validate(json.loads(line)) for line in lines]
    assert len(cases) >= 20, f"Track D expected >= 20 queries, found {len(cases)}"

    for c in cases:
        assert not c.answerable, f"Track D query {c.query_id} must be unanswerable"
        assert c.answerability.value == "unanswerable"
        assert c.relevant_chunk_ids == [], (
            f"Track D query {c.query_id} must have empty relevant chunks"
        )
        assert c.relevant_element_ids == [], (
            f"Track D query {c.query_id} must have empty relevant elements"
        )
        assert c.reference_answer is None, (
            f"Track D query {c.query_id} must have None reference answer"
        )


@pytest.mark.parametrize("track", TRACKS)
def test_dataset_corpus_validation_zero_errors(track: str) -> None:
    validator = DatasetValidator()
    ds_path = GOLDEN_DIR / f"{track}.jsonl"
    corpus_dir = CORPORA_DIR / track

    report = validator.validate_file(file_path=ds_path, corpus=corpus_dir)
    assert report.valid, f"Validation failed for {track}: {report.errors}"
    assert len(report.errors) == 0, f"Found validation errors in {track}: {report.errors}"
