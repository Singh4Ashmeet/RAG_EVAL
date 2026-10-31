"""Evaluation runner contracts and implementations."""

from eval.runners.evaluate import (
    EvaluationReport,
    EvaluationRunner,
    QueryEvaluationResult,
)
from eval.runners.run_config import EvaluationRunConfig
from eval.runners.run_sweep import (
    ExperimentConfig,
    ExperimentReport,
    ExperimentRunner,
    ExperimentVariant,
    PairedMetricComparison,
    VariantComparison,
    compare_evaluation_reports,
    paired_bootstrap_interval,
)

__all__ = [
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationRunConfig",
    "ExperimentConfig",
    "ExperimentReport",
    "ExperimentRunner",
    "ExperimentVariant",
    "PairedMetricComparison",
    "QueryEvaluationResult",
    "VariantComparison",
    "compare_evaluation_reports",
    "paired_bootstrap_interval",
]
