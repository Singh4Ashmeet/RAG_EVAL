"""Comprehensive tests for Track A and Track B golden corpora, manifests, and datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.ingest import create_default_ingestion_pipeline
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.reporting import generate_coverage_report
from eval.datasets.schema import AnnotationStatus, Answerability, CorpusTrack
from eval.datasets.validation import (
    CorpusManifest,
    DiagnosticCode,
    validate_dataset,
)

CORPUS_ROOT = Path("eval/corpora")
TRACK_A_DIR = CORPUS_ROOT / "track_a"
TRACK_B_DIR = CORPUS_ROOT / "track_b"
GOLDEN_ROOT = Path("eval/golden")
TRACK_A_DATASET = GOLDEN_ROOT / "track_a.jsonl"
TRACK_B_DATASET = GOLDEN_ROOT / "track_b.jsonl"


# =========================================================================
# 1. Corpus Integrity and Provenance Tests
# =========================================================================


class TestCorpusIntegrity:
    def test_both_tracks_have_manifests_and_provenance(self) -> None:
        assert (TRACK_A_DIR / "manifest.json").is_file()
        assert (TRACK_A_DIR / "PROVENANCE.json").is_file()
        assert (TRACK_B_DIR / "manifest.json").is_file()
        assert (TRACK_B_DIR / "PROVENANCE.json").is_file()

    def test_track_a_source_documents_match_hashes(self) -> None:
        manifest = CorpusManifest.model_validate_json(
            (TRACK_A_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest.track == CorpusTrack.TRACK_A
        assert len(manifest.documents) == 3

        for doc in manifest.documents:
            file_path = TRACK_A_DIR / doc.filename
            assert file_path.is_file()
            actual_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert actual_sha256 == doc.sha256

            # Verify byte-identical copy with canonical docs/ file
            canonical_path = Path("docs") / doc.filename
            assert canonical_path.is_file()
            assert canonical_path.read_bytes() == file_path.read_bytes()

    def test_track_b_source_document_matches_official_hash(self) -> None:
        manifest = CorpusManifest.model_validate_json(
            (TRACK_B_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest.track == CorpusTrack.TRACK_B
        assert len(manifest.documents) == 1

        rfc_doc = manifest.documents[0]
        assert rfc_doc.filename == "rfc8259.txt"
        file_path = TRACK_B_DIR / rfc_doc.filename
        assert file_path.is_file()
        actual_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_sha256 == rfc_doc.sha256
        assert actual_sha256 == "61a5378f4255c720beb2a4b4a63b29540147c140f36988bf086291989b4cd2d7"

    def test_ingestion_pipeline_ids_are_deterministic(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        for doc_file in (
            TRACK_A_DIR / "architecture.md",
            TRACK_B_DIR / "rfc8259.txt",
        ):
            res1 = pipeline.ingest(doc_file)
            res2 = pipeline.ingest(doc_file)
            assert res1.document is not None and res2.document is not None
            assert res1.document.metadata.document_id == res2.document.metadata.document_id
            assert [e.element_id for e in res1.document.elements] == [
                e.element_id for e in res2.document.elements
            ]
            assert [c.chunk_id for c in res1.chunks] == [c.chunk_id for c in res2.chunks]


# =========================================================================
# 2. Golden Dataset Integrity Tests
# =========================================================================


class TestDatasetIntegrity:
    def test_track_a_dataset_integrity(self) -> None:
        dataset = load_jsonl_dataset(TRACK_A_DATASET)
        assert dataset.track == CorpusTrack.TRACK_A
        assert 12 <= len(dataset.cases) <= 15
        query_ids = [c.query_id for c in dataset.cases]
        assert len(query_ids) == len(set(query_ids)), "Duplicate query IDs in Track A"

        for case in dataset.cases:
            assert case.track == CorpusTrack.TRACK_A
            assert case.status == AnnotationStatus.VERIFIED
            assert case.answerability == Answerability.ANSWERABLE
            assert case.provenance is not None
            assert case.reference_answer is not None and bool(case.reference_answer.strip())
            assert len(case.relevant_chunk_ids) > 0
            assert len(case.relevant_element_ids) > 0

    def test_track_b_dataset_integrity(self) -> None:
        dataset = load_jsonl_dataset(TRACK_B_DATASET)
        assert dataset.track == CorpusTrack.TRACK_B
        assert 10 <= len(dataset.cases) <= 12
        query_ids = [c.query_id for c in dataset.cases]
        assert len(query_ids) == len(set(query_ids)), "Duplicate query IDs in Track B"

        for case in dataset.cases:
            assert case.track == CorpusTrack.TRACK_B
            assert case.status == AnnotationStatus.VERIFIED
            assert case.answerability == Answerability.ANSWERABLE
            assert case.provenance is not None
            assert case.reference_answer is not None and bool(case.reference_answer.strip())
            assert len(case.relevant_chunk_ids) > 0
            assert len(case.relevant_element_ids) > 0


# =========================================================================
# 3. B2 Validation and Cross-Track Isolation Tests
# =========================================================================


class TestValidationAndIsolation:
    def test_track_a_validates_against_track_a_corpus(self) -> None:
        report = validate_dataset(
            dataset_or_path=TRACK_A_DATASET,
            corpus=TRACK_A_DIR,
            expected_track=CorpusTrack.TRACK_A,
        )
        assert report.valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert report.cases_checked == 14
        assert report.documents_checked == 3

    def test_track_b_validates_against_track_b_corpus(self) -> None:
        report = validate_dataset(
            dataset_or_path=TRACK_B_DATASET,
            corpus=TRACK_B_DIR,
            expected_track=CorpusTrack.TRACK_B,
        )
        assert report.valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert report.cases_checked == 11
        assert report.documents_checked == 1

    def test_track_a_against_track_b_corpus_fails_validation(self) -> None:
        report = validate_dataset(
            dataset_or_path=TRACK_A_DATASET,
            corpus=TRACK_B_DIR,
            expected_track=CorpusTrack.TRACK_B,
        )
        assert report.valid is False
        assert any(d.code == DiagnosticCode.TRACK_MISMATCH for d in report.errors)

    def test_track_b_against_track_a_corpus_fails_validation(self) -> None:
        report = validate_dataset(
            dataset_or_path=TRACK_B_DATASET,
            corpus=TRACK_A_DIR,
            expected_track=CorpusTrack.TRACK_A,
        )
        assert report.valid is False
        assert any(d.code == DiagnosticCode.TRACK_MISMATCH for d in report.errors)


# =========================================================================
# 4. B3 Coverage and Quality Report Tests
# =========================================================================


class TestCoverageReporting:
    def test_track_a_coverage_report(self) -> None:
        dataset = load_jsonl_dataset(TRACK_A_DATASET)
        val_report = validate_dataset(
            dataset_or_path=dataset,
            corpus=TRACK_A_DIR,
            expected_track=CorpusTrack.TRACK_A,
        )
        cov_report = generate_coverage_report(
            dataset=dataset,
            validation_report=val_report,
            expected_track=CorpusTrack.TRACK_A,
        )

        assert cov_report.total_cases == 14
        assert cov_report.validation_valid is True
        assert cov_report.manifest_verified is True
        assert cov_report.quality_ratios.verified_ratio == 1.0
        assert cov_report.quality_ratios.answerable_ratio == 1.0

        for metric in ("recall@5", "source_element_recall@5", "token_f1", "exact_match"):
            cov = cov_report.metric_coverage[metric]
            assert cov.eligible_cases == 14
            assert cov.excluded_cases == 0
            assert cov.coverage_ratio == 1.0
            assert cov.calculable is True

    def test_track_b_coverage_report(self) -> None:
        dataset = load_jsonl_dataset(TRACK_B_DATASET)
        val_report = validate_dataset(
            dataset_or_path=dataset,
            corpus=TRACK_B_DIR,
            expected_track=CorpusTrack.TRACK_B,
        )
        cov_report = generate_coverage_report(
            dataset=dataset,
            validation_report=val_report,
            expected_track=CorpusTrack.TRACK_B,
        )

        assert cov_report.total_cases == 11
        assert cov_report.validation_valid is True
        assert cov_report.manifest_verified is True
        assert cov_report.quality_ratios.verified_ratio == 1.0
        assert cov_report.quality_ratios.answerable_ratio == 1.0

        for metric in ("recall@5", "source_element_recall@5", "token_f1", "exact_match"):
            cov = cov_report.metric_coverage[metric]
            assert cov.eligible_cases == 11
            assert cov.excluded_cases == 0
            assert cov.coverage_ratio == 1.0
            assert cov.calculable is True

    def test_cross_track_coverage_rejected(self) -> None:
        dataset = load_jsonl_dataset(TRACK_A_DATASET)
        with pytest.raises(ValueError, match="contradicts expected track 'track_b'"):
            generate_coverage_report(
                dataset=dataset,
                expected_track=CorpusTrack.TRACK_B,
            )
