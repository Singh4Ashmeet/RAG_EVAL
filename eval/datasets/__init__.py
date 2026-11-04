"""Evaluation dataset schemas and loading utilities."""

from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.reporting import (
    DatasetCoverageReport,
    DatasetQualityRatios,
    EvidenceCoverage,
    ExclusionReason,
    MetricCoverage,
    calculate_metric_eligibility,
    generate_coverage_report,
)
from eval.datasets.schema import (
    AnnotationProvenance,
    AnnotationStatus,
    Answerability,
    CorpusTrack,
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    QuestionCategory,
)
from eval.datasets.validation import (
    CorpusDocumentEntry,
    CorpusIndex,
    CorpusManifest,
    DatasetValidationReport,
    DatasetValidator,
    DiagnosticCode,
    DiagnosticSeverity,
    ValidationDiagnostic,
    validate_dataset,
    validate_dataset_structure,
)

__all__ = [
    "AnnotationProvenance",
    "AnnotationStatus",
    "Answerability",
    "CorpusDocumentEntry",
    "CorpusIndex",
    "CorpusManifest",
    "CorpusTrack",
    "DatasetCoverageReport",
    "DatasetQualityRatios",
    "DatasetValidationReport",
    "DatasetValidator",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "Difficulty",
    "EvaluationCase",
    "EvaluationDataset",
    "EvidenceCoverage",
    "ExclusionReason",
    "MetricCoverage",
    "QuestionCategory",
    "ValidationDiagnostic",
    "calculate_metric_eligibility",
    "generate_coverage_report",
    "load_jsonl_dataset",
    "validate_dataset",
    "validate_dataset_structure",
]
