"""Comprehensive unit tests for evaluation dataset coverage and quality reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from eval.datasets import (
    AnnotationProvenance,
    AnnotationStatus,
    Answerability,
    CorpusTrack,
    DatasetValidationReport,
    DiagnosticCode,
    DiagnosticSeverity,
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    EvidenceCoverage,
    ExclusionReason,
    QuestionCategory,
    ValidationDiagnostic,
    calculate_metric_eligibility,
    generate_coverage_report,
)


def _sample_case(**overrides: Any) -> EvaluationCase:
    values: dict[str, Any] = {
        "query_id": "q-1",
        "query": "What is the chunking strategy?",
        "relevant_chunk_ids": ["chunk-1"],
        "relevant_element_ids": ["elem-1"],
        "reference_answer": "Element-aware chunking.",
        "status": AnnotationStatus.VERIFIED,
        "answerability": Answerability.ANSWERABLE,
        "category": QuestionCategory.DIRECT_FACTUAL,
        "difficulty": Difficulty.EASY,
        "provenance": AnnotationProvenance(
            annotator="tester",
            verified_at=datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC),
            source_documents=["doc_a.txt"],
            verification_notes="Verified against spec.",
            evidence_references=["elem-1"],
        ),
        "track": CorpusTrack.TRACK_A,
    }
    values.update(overrides)
    return EvaluationCase.model_validate(values)


def _sample_dataset(
    cases: list[EvaluationCase] | None = None, **overrides: Any
) -> EvaluationDataset:
    case_list = cases if cases is not None else [_sample_case()]
    values: dict[str, Any] = {
        "dataset_id": "synthetic_ds",
        "sha256": "f" * 64,
        "cases": case_list,
        "track": CorpusTrack.TRACK_A,
    }
    values.update(overrides)
    return EvaluationDataset.model_validate(values)


# =========================================================================
# 1. Dataset-Level Distributions
# =========================================================================


class TestDatasetDistributions:
    def test_status_and_answerability_distributions(self) -> None:
        c1 = _sample_case(query_id="q1", status=AnnotationStatus.VERIFIED)
        c2 = _sample_case(
            query_id="q2",
            status=AnnotationStatus.PARTIALLY_ANNOTATED,
            provenance=None,
        )
        c3 = _sample_case(
            query_id="q3",
            status=AnnotationStatus.UNVERIFIED,
            answerability=Answerability.UNANSWERABLE,
            provenance=None,
        )
        c4 = _sample_case(
            query_id="q4",
            status=AnnotationStatus.UNVERIFIED,
            answerability=Answerability.UNDETERMINED,
            provenance=None,
        )

        ds = _sample_dataset(cases=[c1, c2, c3, c4])
        report = generate_coverage_report(ds)

        assert report.total_cases == 4
        assert report.status_distribution[AnnotationStatus.VERIFIED] == 1
        assert report.status_distribution[AnnotationStatus.PARTIALLY_ANNOTATED] == 1
        assert report.status_distribution[AnnotationStatus.UNVERIFIED] == 2
        assert sum(report.status_distribution.values()) == 4

        assert report.answerability_distribution[Answerability.ANSWERABLE] == 2
        assert report.answerability_distribution[Answerability.UNANSWERABLE] == 1
        assert report.answerability_distribution[Answerability.UNDETERMINED] == 1
        assert sum(report.answerability_distribution.values()) == 4

    def test_all_question_categories_represented_even_when_zero(self) -> None:
        c1 = _sample_case(query_id="q1", category=QuestionCategory.DIRECT_FACTUAL)
        c2 = _sample_case(query_id="q2", category=QuestionCategory.CROSS_DOCUMENT)
        ds = _sample_dataset(cases=[c1, c2])
        report = generate_coverage_report(ds)

        # Invariant: Every QuestionCategory enum value must be a key in the distribution
        for cat in QuestionCategory:
            assert cat in report.category_distribution

        assert report.category_distribution[QuestionCategory.DIRECT_FACTUAL] == 1
        assert report.category_distribution[QuestionCategory.CROSS_DOCUMENT] == 1
        assert report.category_distribution[QuestionCategory.MISLEADING_TERMINOLOGY] == 0
        assert sum(report.category_distribution.values()) == 2

    def test_difficulty_distribution_and_missing_difficulty(self) -> None:
        c1 = _sample_case(query_id="q1", difficulty=Difficulty.EASY)
        c2 = _sample_case(query_id="q2", difficulty=Difficulty.HARD)
        c3 = _sample_case(query_id="q3", difficulty=None)
        ds = _sample_dataset(cases=[c1, c2, c3])
        report = generate_coverage_report(ds)

        for diff in Difficulty:
            assert diff in report.difficulty_distribution

        assert report.difficulty_distribution[Difficulty.EASY] == 1
        assert report.difficulty_distribution[Difficulty.HARD] == 1
        assert report.difficulty_distribution[Difficulty.MEDIUM] == 0
        assert report.missing_difficulty_count == 1
        assert sum(report.difficulty_distribution.values()) + report.missing_difficulty_count == 3


# =========================================================================
# 2. Evidence Coverage Tests
# =========================================================================


class TestEvidenceCoverage:
    def test_evidence_coverage_counts_and_distinct_references(self) -> None:
        c1 = _sample_case(
            query_id="q1",
            relevant_chunk_ids=["chunk_1", "chunk_2"],
            relevant_element_ids=["elem_1"],
            provenance=_sample_case().provenance,
        )
        c2 = _sample_case(
            query_id="q2",
            relevant_chunk_ids=["chunk_2", "chunk_3"],
            relevant_element_ids=["elem_1", "elem_2"],
            reference_answer="Another answer",
            provenance=AnnotationProvenance(
                annotator="tester",
                verified_at=datetime.now(UTC),
                source_documents=["doc_a.txt", "doc_b.txt"],
            ),
        )
        ds = _sample_dataset(cases=[c1, c2])
        report = generate_coverage_report(ds)

        ev: EvidenceCoverage = report.evidence_coverage
        assert ev.cases_with_chunk_ids == 2
        assert ev.cases_with_element_ids == 2
        assert ev.cases_with_reference_answer == 2
        assert ev.cases_with_provenance == 2

        assert ev.distinct_chunks_referenced == 3  # chunk_1, chunk_2, chunk_3
        assert ev.distinct_elements_referenced == 2  # elem_1, elem_2
        assert ev.distinct_documents_referenced == 2  # doc_a.txt, doc_b.txt

    def test_valid_vs_invalid_references_with_validation_report(self) -> None:
        c1 = _sample_case(
            query_id="q1",
            relevant_chunk_ids=["chunk_valid", "chunk_invalid"],
            relevant_element_ids=["elem_valid"],
            provenance=AnnotationProvenance(
                annotator="tester",
                verified_at=datetime.now(UTC),
                source_documents=["doc_missing.txt"],
            ),
        )
        ds = _sample_dataset(cases=[c1])

        # Synthetic B2 validation report with invalid references
        val_report = DatasetValidationReport(
            valid=False,
            dataset_id="synthetic_ds",
            dataset_hash="f" * 64,
            corpus_track=CorpusTrack.TRACK_A,
            diagnostics=[
                ValidationDiagnostic(
                    code=DiagnosticCode.UNKNOWN_CHUNK_REFERENCE,
                    severity=DiagnosticSeverity.ERROR,
                    message="Unknown chunk",
                    query_id="q1",
                    chunk_id="chunk_invalid",
                ),
                ValidationDiagnostic(
                    code=DiagnosticCode.PROVENANCE_DOCUMENT_MISSING,
                    severity=DiagnosticSeverity.ERROR,
                    message="Missing doc",
                    query_id="q1",
                    document_id="doc_missing.txt",
                ),
            ],
        )

        report = generate_coverage_report(ds, validation_report=val_report)
        ev = report.evidence_coverage
        assert ev.distinct_chunks_referenced == 2
        assert ev.valid_chunks_referenced == 1  # Only chunk_valid
        assert ev.valid_elements_referenced == 1  # elem_valid
        assert ev.valid_documents_referenced == 0  # doc_missing.txt was invalid


# =========================================================================
# 3. Transparent Quality Ratios Tests
# =========================================================================


class TestQualityRatios:
    def test_quality_ratios_calculated_correctly(self) -> None:
        c1 = _sample_case(query_id="q1", status=AnnotationStatus.VERIFIED)
        c2 = _sample_case(
            query_id="q2",
            status=AnnotationStatus.UNVERIFIED,
            answerability=Answerability.UNANSWERABLE,
            provenance=None,
        )
        ds = _sample_dataset(cases=[c1, c2])
        report = generate_coverage_report(ds)

        ratios = report.quality_ratios
        assert ratios.verified_ratio == pytest.approx(0.5)
        assert ratios.unverified_ratio == pytest.approx(0.5)
        assert ratios.partially_annotated_ratio == pytest.approx(0.0)
        assert ratios.answerable_ratio == pytest.approx(0.5)
        assert ratios.unanswerable_ratio == pytest.approx(0.5)

    def test_zero_total_cases_produces_none_ratios(self) -> None:
        ds = EvaluationDataset.model_construct(
            dataset_id="empty_ds",
            sha256="0" * 64,
            cases=[],
            track=CorpusTrack.TRACK_A,
        )
        report = generate_coverage_report(ds)

        ratios = report.quality_ratios
        assert ratios.verified_ratio is None
        assert ratios.partially_annotated_ratio is None
        assert ratios.unverified_ratio is None
        assert ratios.answerable_ratio is None
        assert ratios.unanswerable_ratio is None
        assert ratios.chunk_evidence_ratio is None
        assert ratios.element_evidence_ratio is None
        assert ratios.reference_answer_ratio is None
        assert ratios.provenance_ratio is None


# =========================================================================
# 4. Metric Eligibility & Denominator Invariants
# =========================================================================


class TestMetricEligibility:
    def test_verified_answerable_case_is_eligible(self) -> None:
        case = _sample_case()
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "recall@5")
        assert cov.eligible_cases == 1
        assert cov.excluded_cases == 0
        assert cov.total_cases == 1
        assert cov.coverage_ratio == 1.0
        assert cov.calculable is True

    def test_unverified_case_is_excluded(self) -> None:
        case = _sample_case(status=AnnotationStatus.UNVERIFIED, provenance=None)
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "precision@5")
        assert cov.eligible_cases == 0
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.UNVERIFIED_ANNOTATION] == 1
        assert cov.calculable is False

    def test_partially_annotated_case_is_excluded(self) -> None:
        case = _sample_case(status=AnnotationStatus.PARTIALLY_ANNOTATED, provenance=None)
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "mrr@5")
        assert cov.eligible_cases == 0
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.PARTIAL_ANNOTATION] == 1

    def test_unanswerable_case_excluded_from_retrieval(self) -> None:
        case = _sample_case(
            status=AnnotationStatus.VERIFIED,
            answerability=Answerability.UNANSWERABLE,
            relevant_chunk_ids=[],
            reference_answer="No evidence.",
        )
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "hit_rate@5")
        assert cov.eligible_cases == 0
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.UNANSWERABLE_NOT_SUPPORTED] == 1
        # Also missing relevant chunks
        assert cov.exclusion_reasons[ExclusionReason.MISSING_RELEVANT_CHUNK_IDS] == 1

    def test_undetermined_answerability_excluded(self) -> None:
        case = _sample_case(
            status=AnnotationStatus.UNVERIFIED,
            answerability=Answerability.UNDETERMINED,
            provenance=None,
        )
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "map@5")
        assert cov.eligible_cases == 0
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.UNDETERMINED_ANSWERABILITY] == 1

    def test_missing_relevant_chunk_ids_excluded_from_chunk_metrics(self) -> None:
        case = _sample_case(
            relevant_chunk_ids=[],
            relevant_element_ids=["elem-1"],
        )
        ds = _sample_dataset(cases=[case])

        chunk_cov = calculate_metric_eligibility(ds, "recall@5")
        assert chunk_cov.eligible_cases == 0
        assert chunk_cov.excluded_cases == 1
        assert chunk_cov.exclusion_reasons[ExclusionReason.MISSING_RELEVANT_CHUNK_IDS] == 1

        # But eligible for source element metrics!
        elem_cov = calculate_metric_eligibility(ds, "source_element_recall@5")
        assert elem_cov.eligible_cases == 1
        assert elem_cov.excluded_cases == 0

    def test_missing_relevant_element_ids_excluded_from_element_metrics(self) -> None:
        case = _sample_case(
            relevant_chunk_ids=["chunk-1"],
            relevant_element_ids=[],
        )
        ds = _sample_dataset(cases=[case])

        elem_cov = calculate_metric_eligibility(ds, "source_element_mrr@5")
        assert elem_cov.eligible_cases == 0
        assert elem_cov.excluded_cases == 1
        assert elem_cov.exclusion_reasons[ExclusionReason.MISSING_RELEVANT_ELEMENT_IDS] == 1

        # But eligible for chunk metrics!
        chunk_cov = calculate_metric_eligibility(ds, "mrr@5")
        assert chunk_cov.eligible_cases == 1
        assert chunk_cov.excluded_cases == 0

    def test_missing_reference_answer_excluded_from_answer_metrics(self) -> None:
        case = _sample_case(reference_answer=None)
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "token_f1")
        assert cov.eligible_cases == 0
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.MISSING_REFERENCE_ANSWER] == 1

    def test_multiple_exclusion_reasons_preserved_without_double_counting_cases(self) -> None:
        # Case has 3 reasons: unverified, undetermined, and missing reference answer
        case = _sample_case(
            status=AnnotationStatus.UNVERIFIED,
            answerability=Answerability.UNDETERMINED,
            reference_answer=None,
            provenance=None,
        )
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "exact_match")
        # Invariant: unique excluded count is 1!
        assert cov.excluded_cases == 1
        assert cov.eligible_cases == 0
        assert cov.total_cases == 1
        assert cov.eligible_cases + cov.excluded_cases == cov.total_cases

        # Reason breakdown records each occurrence
        assert cov.exclusion_reasons[ExclusionReason.UNVERIFIED_ANNOTATION] == 1
        assert cov.exclusion_reasons[ExclusionReason.UNDETERMINED_ANSWERABILITY] == 1
        assert cov.exclusion_reasons[ExclusionReason.MISSING_REFERENCE_ANSWER] == 1

    def test_denominator_invariant_holds_across_all_default_metrics(self) -> None:
        cases = [
            _sample_case(query_id="q1"),
            _sample_case(query_id="q2", status=AnnotationStatus.UNVERIFIED, provenance=None),
            _sample_case(query_id="q3", relevant_chunk_ids=[]),
            _sample_case(query_id="q4", reference_answer=None),
        ]
        ds = _sample_dataset(cases=cases)
        report = generate_coverage_report(ds)

        assert report.total_cases == 4
        for name, cov in report.metric_coverage.items():
            # Core Invariant: eligible + excluded == total
            assert cov.eligible_cases + cov.excluded_cases == cov.total_cases, (
                f"Invariant broken on {name}"
            )
            assert cov.total_cases == 4


# =========================================================================
# 5. Prediction-Aware Reporting Tests
# =========================================================================


class TestPredictionAwareReporting:
    def test_predictions_not_supplied_is_dataset_only(self) -> None:
        case = _sample_case()
        ds = _sample_dataset(cases=[case])

        cov = calculate_metric_eligibility(ds, "recall@5", predictions=None)
        assert cov.eligible_cases == 1
        assert ExclusionReason.MISSING_REQUIRED_PREDICTION not in cov.exclusion_reasons

    def test_all_predictions_present(self) -> None:
        case = _sample_case(query_id="q1")
        ds = _sample_dataset(cases=[case])

        predictions = {
            "q1": {
                "status": "success",
                "retrieved_chunk_ids": ["chunk-1"],
                "generated_answer": "Element-aware chunking.",
            }
        }

        retrieval_cov = calculate_metric_eligibility(ds, "recall@5", predictions=predictions)
        assert retrieval_cov.eligible_cases == 1

        answer_cov = calculate_metric_eligibility(ds, "token_f1", predictions=predictions)
        assert answer_cov.eligible_cases == 1

    def test_missing_query_prediction_excluded(self) -> None:
        c1 = _sample_case(query_id="q1")
        c2 = _sample_case(query_id="q2")
        ds = _sample_dataset(cases=[c1, c2])

        # Prediction only contains q1
        predictions = {
            "q1": {
                "status": "success",
                "retrieved_chunk_ids": ["chunk-1"],
                "generated_answer": "Answer 1",
            }
        }

        cov = calculate_metric_eligibility(ds, "recall@5", predictions=predictions)
        assert cov.total_cases == 2
        assert cov.eligible_cases == 1
        assert cov.excluded_cases == 1
        assert cov.exclusion_reasons[ExclusionReason.MISSING_REQUIRED_PREDICTION] == 1

    def test_empty_retrieved_results_is_valid_prediction_not_missing(self) -> None:
        case = _sample_case(query_id="q1")
        ds = _sample_dataset(cases=[case])

        # Query succeeded but retriever returned 0 chunks (empty list)
        predictions = {
            "q1": {
                "status": "success",
                "retrieved_chunk_ids": [],
            }
        }

        cov = calculate_metric_eligibility(ds, "recall@5", predictions=predictions)
        assert cov.eligible_cases == 1
        assert cov.excluded_cases == 0


# =========================================================================
# 6. B2 Validation Integration Tests
# =========================================================================


class TestB2ValidationIntegration:
    def test_case_level_invalid_chunk_reference_excluded(self) -> None:
        c1 = _sample_case(query_id="q1", relevant_chunk_ids=["chunk_bad"])
        c2 = _sample_case(query_id="q2", relevant_chunk_ids=["chunk_good"])
        ds = _sample_dataset(cases=[c1, c2])

        val_report = DatasetValidationReport(
            valid=False,
            dataset_id="test_ds",
            dataset_hash="a" * 64,
            corpus_track=CorpusTrack.TRACK_A,
            diagnostics=[
                ValidationDiagnostic(
                    code=DiagnosticCode.UNKNOWN_CHUNK_REFERENCE,
                    severity=DiagnosticSeverity.ERROR,
                    message="Chunk not in corpus",
                    query_id="q1",
                    chunk_id="chunk_bad",
                )
            ],
        )

        cov = calculate_metric_eligibility(ds, "recall@5", validation_report=val_report)
        assert cov.eligible_cases == 1  # Only q2 is eligible
        assert cov.excluded_cases == 1  # q1 is excluded
        assert cov.exclusion_reasons[ExclusionReason.INVALID_CORPUS_REFERENCE] == 1

    def test_manifest_verification_flag_preserved(self) -> None:
        ds = _sample_dataset()

        # Valid manifest case
        val_valid = DatasetValidationReport(
            valid=True,
            dataset_id="test_ds",
            dataset_hash="a" * 64,
            corpus_hash="manifest_hash_1234",
            diagnostics=[],
        )
        rep1 = generate_coverage_report(ds, validation_report=val_valid)
        assert rep1.manifest_verified is True
        assert rep1.corpus_hash == "manifest_hash_1234"

        # Missing manifest warning case
        val_unverified = DatasetValidationReport(
            valid=True,
            dataset_id="test_ds",
            dataset_hash="a" * 64,
            diagnostics=[
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_MANIFEST_MISSING,
                    severity=DiagnosticSeverity.WARNING,
                    message="No manifest found",
                )
            ],
        )
        rep2 = generate_coverage_report(ds, validation_report=val_unverified)
        assert rep2.manifest_verified is False


# =========================================================================
# 7. Track Isolation Tests
# =========================================================================


class TestTrackIsolation:
    def test_track_a_and_track_b_reports_remain_independent(self) -> None:
        case_a = _sample_case(query_id="qa", track=CorpusTrack.TRACK_A)
        ds_a = _sample_dataset(cases=[case_a], track=CorpusTrack.TRACK_A)
        rep_a = generate_coverage_report(ds_a, expected_track=CorpusTrack.TRACK_A)
        assert rep_a.track == CorpusTrack.TRACK_A

        case_b = _sample_case(query_id="qb", track=CorpusTrack.TRACK_B)
        ds_b = _sample_dataset(cases=[case_b], track=CorpusTrack.TRACK_B)
        rep_b = generate_coverage_report(ds_b, expected_track=CorpusTrack.TRACK_B)
        assert rep_b.track == CorpusTrack.TRACK_B

    def test_track_mismatch_raises_value_error(self) -> None:
        case_a = _sample_case(track=CorpusTrack.TRACK_A)
        ds_a = _sample_dataset(cases=[case_a], track=CorpusTrack.TRACK_A)

        with pytest.raises(ValueError, match="contradicts expected track 'track_b'"):
            generate_coverage_report(ds_a, expected_track=CorpusTrack.TRACK_B)

    def test_legacy_dataset_without_track_metadata(self) -> None:
        case = _sample_case(track=None)
        ds = _sample_dataset(cases=[case], track=None)

        rep = generate_coverage_report(ds)
        assert rep.track is None


# =========================================================================
# 8. Determinism and Immutability Tests
# =========================================================================


class TestDeterminismAndImmutability:
    def test_report_generation_does_not_mutate_inputs(self) -> None:
        case = _sample_case()
        ds = _sample_dataset(cases=[case])
        original_json = ds.model_dump_json()

        generate_coverage_report(ds)
        assert ds.model_dump_json() == original_json

    def test_repeated_runs_produce_identical_output(self) -> None:
        case = _sample_case()
        ds = _sample_dataset(cases=[case])

        rep1 = generate_coverage_report(ds)
        rep2 = generate_coverage_report(ds)
        assert rep1.model_dump() == rep2.model_dump()
