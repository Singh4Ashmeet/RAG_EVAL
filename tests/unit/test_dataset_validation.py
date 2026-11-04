"""Comprehensive unit tests for dataset structural and corpus validation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from eval.datasets import (
    AnnotationProvenance,
    AnnotationStatus,
    Answerability,
    CorpusDocumentEntry,
    CorpusManifest,
    CorpusTrack,
    DatasetValidator,
    DiagnosticCode,
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    QuestionCategory,
    validate_dataset,
    validate_dataset_structure,
)


def _sample_provenance(
    annotator: str = "curator_1",
    source_documents: list[str] | None = None,
    evidence_references: list[str] | None = None,
) -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator=annotator,
        verified_at=datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC),
        source_documents=source_documents or ["doc_a.txt"],
        verification_notes="Verified against canonical technical documentation.",
        evidence_references=evidence_references or [],
    )


def _sample_case(**overrides: object) -> EvaluationCase:
    values: dict[str, object] = {
        "query_id": "q-1",
        "query": "How is chunking performed in RAG_EVAL?",
        "relevant_chunk_ids": [],
        "relevant_element_ids": [],
        "reference_answer": "Element-aware chunking preserves boundary metadata.",
        "status": AnnotationStatus.UNVERIFIED,
        "answerability": Answerability.ANSWERABLE,
        "category": QuestionCategory.DIRECT_FACTUAL,
        "difficulty": Difficulty.EASY,
        "track": CorpusTrack.TRACK_A,
    }
    values.update(overrides)
    return EvaluationCase.model_validate(values)


def _create_corpus_dir(tmp_path: Path, track: CorpusTrack = CorpusTrack.TRACK_A) -> Path:
    corpus_dir = tmp_path / f"corpus_{track.value}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    doc_content = "# System Architecture\n\nChunking preserves boundary metadata across documents."
    doc_path = corpus_dir / "architecture.md"
    doc_path.write_text(doc_content, encoding="utf-8")
    doc_sha256 = hashlib.sha256(doc_path.read_bytes()).hexdigest()

    manifest = CorpusManifest(
        track=track,
        manifest_version="1.0.0",
        documents=[
            CorpusDocumentEntry(
                filename="architecture.md",
                title="System Architecture",
                sha256=doc_sha256,
            )
        ],
    )
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return corpus_dir


# =========================================================================
# 1. Structural and JSONL Syntax Tests
# =========================================================================


class TestDatasetStructure:
    def test_valid_jsonl_file_passes(self, tmp_path: Path) -> None:
        case = _sample_case()
        file_path = tmp_path / "valid.jsonl"
        file_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is True
        assert len(report.errors) == 0
        assert report.cases_checked == 1

    def test_malformed_json_syntax_reports_line_number(self, tmp_path: Path) -> None:
        case1 = _sample_case(query_id="q-1")
        file_path = tmp_path / "malformed.jsonl"
        content = f"{case1.model_dump_json()}\n{{not valid json\n"
        file_path.write_text(content, encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.DATASET_INVALID_JSON]
        assert len(errors) == 1
        assert errors[0].line_number == 2

    def test_schema_violation_reports_line_and_field(self, tmp_path: Path) -> None:
        file_path = tmp_path / "schema_violation.jsonl"
        bad_record = {
            "query_id": "q-bad",
            # missing query string
            "reference_answer": "Valid answer",
        }
        file_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.DATASET_SCHEMA_ERROR]
        assert len(errors) >= 1
        assert errors[0].line_number == 1
        assert errors[0].field == "query"

    def test_duplicate_query_id_detected(self, tmp_path: Path) -> None:
        case1 = _sample_case(query_id="q-dup")
        case2 = _sample_case(query_id="q-dup")
        file_path = tmp_path / "duplicate_ids.jsonl"
        file_path.write_text(
            f"{case1.model_dump_json()}\n{case2.model_dump_json()}\n", encoding="utf-8"
        )

        report = validate_dataset_structure(file_path)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.DATASET_DUPLICATE_QUERY_ID]
        assert len(errors) == 1
        assert errors[0].query_id == "q-dup"
        assert errors[0].line_number == 2

    def test_empty_dataset_detected(self, tmp_path: Path) -> None:
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("\n\n   \n", encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.DATASET_EMPTY]
        assert len(errors) == 1

    def test_blank_lines_preserved_in_hash(self, tmp_path: Path) -> None:
        case = _sample_case()
        file_path = tmp_path / "blank_lines.jsonl"
        content = f"\n{case.model_dump_json()}\n\n"
        file_path.write_text(content, encoding="utf-8")

        raw_bytes = file_path.read_bytes()
        expected_hash = hashlib.sha256(raw_bytes).hexdigest()

        report = validate_dataset_structure(file_path, expected_hash=expected_hash)
        assert report.valid is True
        assert report.dataset_hash == expected_hash

    def test_legacy_records_default_safely_with_warnings(self, tmp_path: Path) -> None:
        legacy_data = {
            "query_id": "legacy-q1",
            "query": "What is the pipeline architecture?",
            "relevant_chunk_ids": ["c1"],
            "reference_answer": "Ingestion pipeline.",
        }
        file_path = tmp_path / "legacy.jsonl"
        file_path.write_text(json.dumps(legacy_data) + "\n", encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is True
        warnings = [d for d in report.warnings if d.code == DiagnosticCode.UNVERIFIED_CANDIDATE]
        assert len(warnings) == 1
        assert warnings[0].query_id == "legacy-q1"

    def test_multiple_line_errors_reported_together(self, tmp_path: Path) -> None:
        file_path = tmp_path / "multiple_errors.jsonl"
        content = (
            "{bad json 1\n"
            + json.dumps({"query_id": "q1"})  # missing query and ground truth
            + "\n{bad json 2\n"
        )
        file_path.write_text(content, encoding="utf-8")

        report = validate_dataset_structure(file_path)
        assert report.valid is False
        assert len(report.errors) >= 3


# =========================================================================
# 2. Hash and Digest Integrity Tests
# =========================================================================


class TestDatasetHashAndIntegrity:
    def test_dataset_hash_match(self, tmp_path: Path) -> None:
        case = _sample_case()
        file_path = tmp_path / "hash_match.jsonl"
        file_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        report = validate_dataset(file_path, expected_hash=actual_hash)
        assert report.valid is True
        assert not any(d.code == DiagnosticCode.DATASET_HASH_MISMATCH for d in report.diagnostics)

    def test_dataset_hash_mismatch(self, tmp_path: Path) -> None:
        case = _sample_case()
        file_path = tmp_path / "hash_mismatch.jsonl"
        file_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")

        wrong_hash = "0" * 64
        report = validate_dataset(file_path, expected_hash=wrong_hash)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.DATASET_HASH_MISMATCH]
        assert len(errors) == 1


# =========================================================================
# 3. Corpus Reference and Manifest Tests
# =========================================================================


class TestCorpusReferences:
    def test_valid_corpus_references_pass(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        validator = DatasetValidator()
        index, _ = validator._resolve_corpus_index(corpus_dir, CorpusTrack.TRACK_A)
        assert index is not None

        elem_id = next(iter(index.element_ids))
        chunk_id = next(iter(index.chunk_ids))

        case = _sample_case(
            query_id="q-real",
            relevant_element_ids=[elem_id],
            relevant_chunk_ids=[chunk_id],
            status=AnnotationStatus.VERIFIED,
            provenance=_sample_provenance(
                source_documents=["architecture.md"],
                evidence_references=[elem_id],
            ),
        )

        dataset = EvaluationDataset(
            dataset_id="golden_track_a",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validator.validate_dataset(
            dataset, corpus=index, expected_track=CorpusTrack.TRACK_A
        )
        assert report.valid is True
        assert len(report.errors) == 0
        assert report.documents_checked == 1
        assert report.elements_checked > 0
        assert report.chunks_checked > 0

    def test_unknown_element_id_detected(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case(
            query_id="q-unknown-elem",
            relevant_element_ids=["elem_nonexistent_12345"],
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.UNKNOWN_ELEMENT_REFERENCE]
        assert len(errors) == 1
        assert errors[0].element_id == "elem_nonexistent_12345"

    def test_unknown_chunk_id_detected(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case(
            query_id="q-unknown-chunk",
            relevant_chunk_ids=["chunk_nonexistent_99999"],
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.UNKNOWN_CHUNK_REFERENCE]
        assert len(errors) == 1
        assert errors[0].chunk_id == "chunk_nonexistent_99999"

    def test_chunk_element_mismatch_warning(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        validator = DatasetValidator()
        index, _ = validator._resolve_corpus_index(corpus_dir, CorpusTrack.TRACK_A)
        assert index is not None

        elem_id = next(iter(index.element_ids))
        chunk_id = next(iter(index.chunk_ids))

        # Intentionally tamper with index chunk-to-element mapping to simulate disjoint sets
        index.chunk_to_elements[chunk_id] = frozenset(["elem_other_disjoint"])

        case = _sample_case(
            query_id="q-mismatch",
            relevant_element_ids=[elem_id],
            relevant_chunk_ids=[chunk_id],
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validator.validate_dataset(
            dataset, corpus=index, expected_track=CorpusTrack.TRACK_A
        )
        warnings = [d for d in report.warnings if d.code == DiagnosticCode.CHUNK_ELEMENT_MISMATCH]
        assert len(warnings) == 1
        assert warnings[0].chunk_id == chunk_id

    def test_duplicate_manifest_document_entry(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        manifest_file = corpus_dir / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        # Duplicate document entry
        manifest_data["documents"].append(manifest_data["documents"][0])
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.CORPUS_DUPLICATE_DOCUMENT_ID]
        assert len(errors) == 1

    def test_missing_corpus_document_file(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        # Delete the actual document file
        (corpus_dir / "architecture.md").unlink()

        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.CORPUS_DOCUMENT_MISSING]
        assert len(errors) == 1

    def test_corpus_document_hash_mismatch(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        # Modify the document content without updating manifest
        (corpus_dir / "architecture.md").write_text("Modified content", encoding="utf-8")

        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [
            d for d in report.errors if d.code == DiagnosticCode.CORPUS_DOCUMENT_HASH_MISMATCH
        ]
        assert len(errors) == 1

    def test_unexpected_file_in_corpus_warning(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        # Add undeclared file
        (corpus_dir / "extra.md").write_text("Extra unindexed content", encoding="utf-8")

        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        warnings = [d for d in report.warnings if d.code == DiagnosticCode.CORPUS_UNEXPECTED_FILE]
        assert len(warnings) == 1


# =========================================================================
# 4. Track Isolation Tests
# =========================================================================


class TestTrackIsolation:
    def test_track_a_dataset_on_track_a_corpus_passes(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case(track=CorpusTrack.TRACK_A)
        dataset = EvaluationDataset(
            dataset_id="track_a_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is True
        assert len(report.errors) == 0

    def test_track_b_dataset_on_track_b_corpus_passes(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_B)
        case = _sample_case(track=CorpusTrack.TRACK_B)
        dataset = EvaluationDataset(
            dataset_id="track_b_ds",
            sha256="b" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_B,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_B)
        assert report.valid is True
        assert len(report.errors) == 0

    def test_track_mismatch_between_dataset_and_corpus(self, tmp_path: Path) -> None:
        corpus_a = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case_b = _sample_case(track=CorpusTrack.TRACK_B)
        dataset_b = EvaluationDataset(
            dataset_id="track_b_ds",
            sha256="b" * 64,
            cases=[case_b],
            track=CorpusTrack.TRACK_B,
        )

        report = validate_dataset(dataset_b, corpus=corpus_a, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.TRACK_MISMATCH]
        assert len(errors) >= 1

    def test_case_level_track_mismatch_detected(self, tmp_path: Path) -> None:
        case = _sample_case(query_id="q-cross", track=CorpusTrack.TRACK_B)
        dataset = EvaluationDataset(
            dataset_id="track_a_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.CROSS_TRACK_REFERENCE]
        assert len(errors) == 1
        assert errors[0].query_id == "q-cross"

    def test_missing_track_metadata_emits_warning(self) -> None:
        case = _sample_case(track=None)
        dataset = EvaluationDataset(
            dataset_id="legacy_ds",
            sha256="a" * 64,
            cases=[case],
            track=None,
        )

        report = validate_dataset(dataset, expected_track=CorpusTrack.TRACK_A)
        warnings = [d for d in report.warnings if d.code == DiagnosticCode.MISSING_TRACK_METADATA]
        assert len(warnings) == 1


# =========================================================================
# 5. Annotation Lifecycle and Provenance Tests
# =========================================================================


class TestAnnotationLifecycle:
    def test_provenance_referencing_missing_document(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case(
            query_id="q-missing-doc",
            status=AnnotationStatus.VERIFIED,
            provenance=_sample_provenance(source_documents=["missing_document.md"]),
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.PROVENANCE_DOCUMENT_MISSING]
        assert len(errors) == 1
        assert errors[0].document_id == "missing_document.md"

    def test_provenance_referencing_missing_evidence(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case(
            query_id="q-missing-ev",
            status=AnnotationStatus.VERIFIED,
            provenance=_sample_provenance(
                source_documents=["architecture.md"],
                evidence_references=["elem_nonexistent_888"],
            ),
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)
        assert report.valid is False
        errors = [d for d in report.errors if d.code == DiagnosticCode.PROVENANCE_EVIDENCE_MISSING]
        assert len(errors) == 1
        assert errors[0].element_id == "elem_nonexistent_888"

    def test_unanswerable_case_with_retrieval_ground_truth_emits_warning(self) -> None:
        case = _sample_case(
            query_id="q-unanswerable",
            answerability=Answerability.UNANSWERABLE,
            relevant_chunk_ids=["chunk-1"],
            reference_answer="Should not be answerable.",
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, expected_track=CorpusTrack.TRACK_A)
        warnings = [
            d for d in report.warnings if d.code == DiagnosticCode.ANNOTATION_STATE_INCONSISTENT
        ]
        assert len(warnings) == 1

    def test_answerable_case_missing_reference_answer_emits_warning(self) -> None:
        case = _sample_case(
            query_id="q-no-ans",
            answerability=Answerability.ANSWERABLE,
            relevant_chunk_ids=["chunk-1"],
            reference_answer=None,
        )
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        report = validate_dataset(dataset, expected_track=CorpusTrack.TRACK_A)
        warnings = [d for d in report.warnings if d.code == DiagnosticCode.MISSING_REFERENCE_ANSWER]
        assert len(warnings) == 1


# =========================================================================
# 6. Determinism and Immutability Tests
# =========================================================================


class TestDeterminismAndImmutability:
    def test_validator_does_not_mutate_inputs(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )
        orig_cases_json = dataset.model_dump_json()

        validator = DatasetValidator()
        validator.validate_dataset(dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A)

        assert dataset.model_dump_json() == orig_cases_json

    def test_repeated_validation_produces_identical_reports(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path, CorpusTrack.TRACK_A)
        case = _sample_case()
        dataset = EvaluationDataset(
            dataset_id="test_ds",
            sha256="a" * 64,
            cases=[case],
            track=CorpusTrack.TRACK_A,
        )

        validator = DatasetValidator()
        report1 = validator.validate_dataset(
            dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A
        )
        report2 = validator.validate_dataset(
            dataset, corpus=corpus_dir, expected_track=CorpusTrack.TRACK_A
        )

        assert report1.model_dump() == report2.model_dump()
