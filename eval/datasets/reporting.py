"""Evaluation dataset coverage, quality reporting, and metric eligibility analysis."""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eval.datasets.schema import (
    AnnotationStatus,
    Answerability,
    CorpusTrack,
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    QuestionCategory,
)
from eval.datasets.validation import (
    DatasetValidationReport,
    DiagnosticCode,
)


class ExclusionReason(StrEnum):
    """Stable, auditable reasons why an evaluation case is excluded from a metric."""

    UNVERIFIED_ANNOTATION = "unverified_annotation"
    PARTIAL_ANNOTATION = "partial_annotation"
    UNDETERMINED_ANSWERABILITY = "undetermined_answerability"
    UNANSWERABLE_NOT_SUPPORTED = "unanswerable_not_supported"
    MISSING_RELEVANT_CHUNK_IDS = "missing_relevant_chunk_ids"
    MISSING_RELEVANT_ELEMENT_IDS = "missing_relevant_element_ids"
    MISSING_REFERENCE_ANSWER = "missing_reference_answer"
    INVALID_CORPUS_REFERENCE = "invalid_corpus_reference"
    MISSING_REQUIRED_PREDICTION = "missing_required_prediction"
    UNSUPPORTED_METRIC_INPUT = "unsupported_metric_input"


class MetricCoverage(BaseModel):
    """Eligibility and exclusion breakdown for a single evaluation metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str
    total_cases: int
    eligible_cases: int
    excluded_cases: int
    coverage_ratio: float | None
    exclusion_reasons: dict[ExclusionReason, int]
    calculable: bool
    incalculable_reason: str | None = None


class EvidenceCoverage(BaseModel):
    """Corpus and annotation ground-truth evidence counts and distinct references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cases_with_chunk_ids: int
    cases_with_element_ids: int
    cases_with_reference_answer: int
    cases_with_provenance: int

    distinct_documents_referenced: int
    distinct_elements_referenced: int
    distinct_chunks_referenced: int

    valid_documents_referenced: int | None = None
    valid_elements_referenced: int | None = None
    valid_chunks_referenced: int | None = None


class DatasetQualityRatios(BaseModel):
    """Transparent quality and completeness ratios with explicit denominators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_ratio: float | None
    partially_annotated_ratio: float | None
    unverified_ratio: float | None

    answerable_ratio: float | None
    unanswerable_ratio: float | None

    chunk_evidence_ratio: float | None
    element_evidence_ratio: float | None
    reference_answer_ratio: float | None
    provenance_ratio: float | None


class DatasetCoverageReport(BaseModel):
    """Comprehensive, machine-readable dataset coverage and quality report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_sha256: str | None = None
    track: CorpusTrack | None = None
    total_cases: int

    validation_valid: bool | None = None
    validation_errors_count: int = 0
    validation_warnings_count: int = 0
    corpus_hash: str | None = None
    manifest_verified: bool = False

    status_distribution: dict[AnnotationStatus, int]
    answerability_distribution: dict[Answerability, int]
    category_distribution: dict[QuestionCategory, int]
    difficulty_distribution: dict[Difficulty, int]
    missing_difficulty_count: int

    evidence_coverage: EvidenceCoverage
    quality_ratios: DatasetQualityRatios

    metric_coverage: dict[str, MetricCoverage] = Field(default_factory=dict)


# Supported metric families
_CHUNK_RETRIEVAL_METRICS = frozenset({"recall", "precision", "mrr", "map", "hit_rate"})
_ELEMENT_RETRIEVAL_METRICS = frozenset(
    {"source_element_recall", "source_element_mrr", "source_element_hit_rate"}
)
_ANSWER_METRICS = frozenset({"exact_match", "token_f1"})

_DEFAULT_CUTOFFS = (1, 3, 5, 10)


def _classify_metric(metric_name: str) -> tuple[str, str | None]:
    """Normalize a metric name and return its family ('chunk', 'element', 'answer', or 'unknown').

    Returns (family, normalized_base_name).
    """
    clean_name = metric_name.strip()
    if clean_name.startswith("retrieval."):
        clean_name = clean_name[len("retrieval.") :]
    elif clean_name.startswith("answer."):
        clean_name = clean_name[len("answer.") :]

    # Strip @k cutoff if present
    base_name = re.sub(r"@\d+$", "", clean_name)

    if base_name in _CHUNK_RETRIEVAL_METRICS:
        return "chunk", base_name
    if base_name in _ELEMENT_RETRIEVAL_METRICS:
        return "element", base_name
    if base_name in _ANSWER_METRICS:
        return "answer", base_name
    return "unknown", None


def _extract_predictions_map(
    predictions: Any,
) -> dict[str, Any] | None:
    """Normalize prediction inputs into a query_id -> prediction lookup dict."""
    if predictions is None:
        return None

    # Handle EvaluationReport
    if hasattr(predictions, "queries"):
        return {q.query_id: q for q in predictions.queries}

    # Handle Sequence of QueryEvaluationResult or dicts
    if isinstance(predictions, (list, tuple)):
        result_map: dict[str, Any] = {}
        for item in predictions:
            if hasattr(item, "query_id"):
                result_map[item.query_id] = item
            elif isinstance(item, dict) and "query_id" in item:
                result_map[item["query_id"]] = item
        return result_map

    # Handle dict of query_id -> item
    if isinstance(predictions, dict):
        return predictions

    return None


def _check_prediction_presence(
    case: EvaluationCase,
    metric_family: str,
    predictions_map: dict[str, Any] | None,
) -> bool:
    """Check whether the required prediction is present for the case and metric family.

    Returns True if prediction is present/valid, False if missing.
    """
    if predictions_map is None:
        return True  # Dataset-only mode

    if case.query_id not in predictions_map:
        return False

    pred = predictions_map[case.query_id]

    if metric_family in {"chunk", "element"}:
        # In QueryEvaluationResult, check status != 'failed'
        if hasattr(pred, "status") and pred.status == "failed":
            return False
        if isinstance(pred, dict) and pred.get("status") == "failed":
            return False
        return True

    if metric_family == "answer":
        # Check generated answer presence
        if hasattr(pred, "generated_answer"):
            return pred.generated_answer is not None
        if isinstance(pred, dict):
            return pred.get("generated_answer") is not None
        return True

    return True


def _get_invalid_references(
    validation_report: DatasetValidationReport | None,
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str], set[str], set[str]]:
    """Extract sets and case-level maps of invalid references from a validation report.

    Returns:
        (case_invalid_chunks, case_invalid_elements, all_invalid_chunks,
         all_invalid_elements, all_invalid_docs)
    """
    case_invalid_chunks: dict[str, set[str]] = {}
    case_invalid_elements: dict[str, set[str]] = {}
    all_invalid_chunks: set[str] = set()
    all_invalid_elements: set[str] = set()
    all_invalid_docs: set[str] = set()

    if validation_report is None:
        return (
            case_invalid_chunks,
            case_invalid_elements,
            all_invalid_chunks,
            all_invalid_elements,
            all_invalid_docs,
        )

    for diag in validation_report.errors:
        if diag.code == DiagnosticCode.UNKNOWN_CHUNK_REFERENCE:
            if diag.chunk_id:
                all_invalid_chunks.add(diag.chunk_id)
            if diag.query_id and diag.chunk_id:
                case_invalid_chunks.setdefault(diag.query_id, set()).add(diag.chunk_id)
        elif diag.code == DiagnosticCode.UNKNOWN_ELEMENT_REFERENCE:
            if diag.element_id:
                all_invalid_elements.add(diag.element_id)
            if diag.query_id and diag.element_id:
                case_invalid_elements.setdefault(diag.query_id, set()).add(diag.element_id)
        elif diag.code in {
            DiagnosticCode.PROVENANCE_DOCUMENT_MISSING,
            DiagnosticCode.CORPUS_DOCUMENT_MISSING,
        }:
            if diag.document_id:
                all_invalid_docs.add(diag.document_id)

    return (
        case_invalid_chunks,
        case_invalid_elements,
        all_invalid_chunks,
        all_invalid_elements,
        all_invalid_docs,
    )


def calculate_metric_eligibility(
    dataset: EvaluationDataset,
    metric_name: str,
    validation_report: DatasetValidationReport | None = None,
    predictions: Any = None,
) -> MetricCoverage:
    """Determine case-by-case eligibility and exclusion reasons for a specific metric."""
    family, _ = _classify_metric(metric_name)
    total_cases = len(dataset.cases)

    if family == "unknown":
        return MetricCoverage(
            metric_name=metric_name,
            total_cases=total_cases,
            eligible_cases=0,
            excluded_cases=total_cases,
            coverage_ratio=0.0 if total_cases > 0 else None,
            exclusion_reasons=(
                {ExclusionReason.UNSUPPORTED_METRIC_INPUT: total_cases} if total_cases > 0 else {}
            ),
            calculable=False,
            incalculable_reason=f"Unsupported or unrecognized metric '{metric_name}'.",
        )

    pred_map = _extract_predictions_map(predictions)
    case_invalid_chunks, case_invalid_elements, _, _, _ = _get_invalid_references(validation_report)

    eligible_count = 0
    excluded_count = 0
    reason_counts: Counter[ExclusionReason] = Counter()

    for case in dataset.cases:
        case_reasons: list[ExclusionReason] = []

        # 1. Annotation status check
        if case.status == AnnotationStatus.UNVERIFIED:
            case_reasons.append(ExclusionReason.UNVERIFIED_ANNOTATION)
        elif case.status == AnnotationStatus.PARTIALLY_ANNOTATED:
            case_reasons.append(ExclusionReason.PARTIAL_ANNOTATION)

        # 2. Answerability check
        if case.answerability == Answerability.UNDETERMINED:
            case_reasons.append(ExclusionReason.UNDETERMINED_ANSWERABILITY)
        elif case.answerability == Answerability.UNANSWERABLE:
            case_reasons.append(ExclusionReason.UNANSWERABLE_NOT_SUPPORTED)

        # 3. Ground-truth presence and corpus validity per metric family
        if family == "chunk":
            if not case.relevant_chunk_ids:
                case_reasons.append(ExclusionReason.MISSING_RELEVANT_CHUNK_IDS)
            elif case.query_id in case_invalid_chunks:
                case_reasons.append(ExclusionReason.INVALID_CORPUS_REFERENCE)

        elif family == "element":
            if not case.relevant_element_ids:
                case_reasons.append(ExclusionReason.MISSING_RELEVANT_ELEMENT_IDS)
            elif case.query_id in case_invalid_elements:
                case_reasons.append(ExclusionReason.INVALID_CORPUS_REFERENCE)

        elif family == "answer":
            if case.reference_answer is None or not case.reference_answer.strip():
                case_reasons.append(ExclusionReason.MISSING_REFERENCE_ANSWER)

        # 4. Prediction presence check (when predictions are supplied)
        if not _check_prediction_presence(case, family, pred_map):
            case_reasons.append(ExclusionReason.MISSING_REQUIRED_PREDICTION)

        # Invariant: unique case classification
        if not case_reasons:
            eligible_count += 1
        else:
            excluded_count += 1
            for r in case_reasons:
                reason_counts[r] += 1

    coverage_ratio = (eligible_count / total_cases) if total_cases > 0 else None
    calculable = eligible_count > 0
    incalculable_reason = None
    if not calculable:
        if total_cases == 0:
            incalculable_reason = "Dataset contains zero cases."
        else:
            top_reason = (
                reason_counts.most_common(1)[0][0] if reason_counts else "no eligible cases"
            )
            incalculable_reason = (
                f"No cases eligible for '{metric_name}'. Primary exclusion: {top_reason}."
            )

    return MetricCoverage(
        metric_name=metric_name,
        total_cases=total_cases,
        eligible_cases=eligible_count,
        excluded_cases=excluded_count,
        coverage_ratio=coverage_ratio,
        exclusion_reasons=dict(reason_counts),
        calculable=calculable,
        incalculable_reason=incalculable_reason,
    )


def generate_coverage_report(
    dataset: EvaluationDataset,
    validation_report: DatasetValidationReport | None = None,
    expected_track: CorpusTrack | None = None,
    predictions: Any = None,
) -> DatasetCoverageReport:
    """Generate a typed dataset coverage and quality report across all distributions and metrics."""
    # 1. Track Isolation Verification
    effective_track = expected_track or dataset.track
    if expected_track is not None and dataset.track is not None and dataset.track != expected_track:
        raise ValueError(
            f"Dataset declares track '{dataset.track}' which contradicts "
            f"expected track '{expected_track}'."
        )

    total_cases = len(dataset.cases)

    # 2. Distributions
    status_counts: dict[AnnotationStatus, int] = {s: 0 for s in AnnotationStatus}
    answerability_counts: dict[Answerability, int] = {a: 0 for a in Answerability}
    category_counts: dict[QuestionCategory, int] = {c: 0 for c in QuestionCategory}
    difficulty_counts: dict[Difficulty, int] = {d: 0 for d in Difficulty}
    missing_difficulty_count = 0

    cases_with_chunk = 0
    cases_with_elem = 0
    cases_with_ans = 0
    cases_with_prov = 0

    all_chunks: set[str] = set()
    all_elements: set[str] = set()
    all_docs: set[str] = set()

    for case in dataset.cases:
        status_counts[case.status] = status_counts.get(case.status, 0) + 1
        answerability_counts[case.answerability] = (
            answerability_counts.get(case.answerability, 0) + 1
        )

        if case.category is not None:
            category_counts[case.category] = category_counts.get(case.category, 0) + 1

        if case.difficulty is not None:
            difficulty_counts[case.difficulty] = difficulty_counts.get(case.difficulty, 0) + 1
        else:
            missing_difficulty_count += 1

        if case.relevant_chunk_ids:
            cases_with_chunk += 1
            all_chunks.update(case.relevant_chunk_ids)

        if case.relevant_element_ids:
            cases_with_elem += 1
            all_elements.update(case.relevant_element_ids)

        if case.reference_answer is not None and case.reference_answer.strip():
            cases_with_ans += 1

        if case.provenance is not None:
            cases_with_prov += 1
            all_docs.update(case.provenance.source_documents)

    # 3. Evidence Coverage & Validation Mappings
    _, _, invalid_chunks, invalid_elements, invalid_docs = _get_invalid_references(
        validation_report
    )

    valid_chunks_count: int | None = None
    valid_elements_count: int | None = None
    valid_docs_count: int | None = None

    if validation_report is not None:
        valid_chunks_count = len(all_chunks - invalid_chunks)
        valid_elements_count = len(all_elements - invalid_elements)
        valid_docs_count = len(all_docs - invalid_docs)

    evidence_coverage = EvidenceCoverage(
        cases_with_chunk_ids=cases_with_chunk,
        cases_with_element_ids=cases_with_elem,
        cases_with_reference_answer=cases_with_ans,
        cases_with_provenance=cases_with_prov,
        distinct_documents_referenced=len(all_docs),
        distinct_elements_referenced=len(all_elements),
        distinct_chunks_referenced=len(all_chunks),
        valid_documents_referenced=valid_docs_count,
        valid_elements_referenced=valid_elements_count,
        valid_chunks_referenced=valid_chunks_count,
    )

    # 4. Transparent Quality Ratios
    def _ratio(count: int) -> float | None:
        return (count / total_cases) if total_cases > 0 else None

    quality_ratios = DatasetQualityRatios(
        verified_ratio=_ratio(status_counts[AnnotationStatus.VERIFIED]),
        partially_annotated_ratio=_ratio(status_counts[AnnotationStatus.PARTIALLY_ANNOTATED]),
        unverified_ratio=_ratio(status_counts[AnnotationStatus.UNVERIFIED]),
        answerable_ratio=_ratio(answerability_counts[Answerability.ANSWERABLE]),
        unanswerable_ratio=_ratio(answerability_counts[Answerability.UNANSWERABLE]),
        chunk_evidence_ratio=_ratio(cases_with_chunk),
        element_evidence_ratio=_ratio(cases_with_elem),
        reference_answer_ratio=_ratio(cases_with_ans),
        provenance_ratio=_ratio(cases_with_prov),
    )

    # 5. Manifest Verification Status
    manifest_verified = False
    validation_valid: bool | None = None
    val_errors = 0
    val_warnings = 0
    corpus_hash: str | None = None

    if validation_report is not None:
        validation_valid = validation_report.valid
        val_errors = len(validation_report.errors)
        val_warnings = len(validation_report.warnings)
        corpus_hash = validation_report.corpus_hash

        has_manifest_missing = any(
            w.code == DiagnosticCode.CORPUS_MANIFEST_MISSING for w in validation_report.warnings
        )
        has_manifest_invalid = any(
            e.code == DiagnosticCode.CORPUS_MANIFEST_INVALID for e in validation_report.errors
        )
        if not has_manifest_missing and not has_manifest_invalid and corpus_hash is not None:
            manifest_verified = True

    # 6. Default Metric Eligibility Set
    metric_coverage: dict[str, MetricCoverage] = {}
    default_metric_names: list[str] = []

    for cutoff in _DEFAULT_CUTOFFS:
        for m in ("recall", "precision", "mrr", "map", "hit_rate"):
            default_metric_names.append(f"{m}@{cutoff}")
        for m in ("recall", "mrr", "hit_rate"):
            default_metric_names.append(f"source_element_{m}@{cutoff}")

    default_metric_names.extend(["exact_match", "token_f1"])

    for m_name in default_metric_names:
        metric_coverage[m_name] = calculate_metric_eligibility(
            dataset=dataset,
            metric_name=m_name,
            validation_report=validation_report,
            predictions=predictions,
        )

    return DatasetCoverageReport(
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.sha256,
        track=effective_track,
        total_cases=total_cases,
        validation_valid=validation_valid,
        validation_errors_count=val_errors,
        validation_warnings_count=val_warnings,
        corpus_hash=corpus_hash,
        manifest_verified=manifest_verified,
        status_distribution=status_counts,
        answerability_distribution=answerability_counts,
        category_distribution=category_counts,
        difficulty_distribution=difficulty_counts,
        missing_difficulty_count=missing_difficulty_count,
        evidence_coverage=evidence_coverage,
        quality_ratios=quality_ratios,
        metric_coverage=metric_coverage,
    )
