"""Repeatable experiment sweeps and paired per-query metric comparisons."""

from __future__ import annotations

import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from core.errors import EvaluationError, RAGEvalError
from core.generate.pipeline import GroundedAnswerGenerator
from core.retrieval.base import Retriever
from eval.datasets.schema import EvaluationDataset
from eval.runners.evaluate import EvaluationReport, EvaluationRunner
from eval.runners.run_config import EvaluationRunConfig

VariantStatus = Literal["complete", "partial", "failed"]
ComparisonStatus = Literal["complete", "unavailable"]
ExperimentStatus = Literal["complete", "partial", "failed"]


@dataclass(frozen=True)
class ExperimentVariant:
    """Runtime pipeline components plus a JSON-safe description of their config."""

    name: str
    pipeline_config: dict[str, Any]
    retriever: Retriever
    answerer: GroundedAnswerGenerator | None = None

    def __post_init__(self) -> None:
        cleaned_name = self.name.strip()
        if not cleaned_name:
            raise ValueError("Experiment variant name cannot be empty.")
        if not isinstance(self.pipeline_config, dict):
            raise ValueError("pipeline_config must be a JSON-compatible object.")
        try:
            encoded_config = json.dumps(
                self.pipeline_config,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            normalized_config = json.loads(encoded_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pipeline_config must contain only finite JSON values: {exc}"
            ) from exc
        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "pipeline_config", normalized_config)


class ExperimentConfig(BaseModel):
    """Reproducible settings for a baseline-versus-variants experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(..., min_length=1)
    baseline_variant: str = Field(..., min_length=1)
    evaluation: EvaluationRunConfig = Field(default_factory=EvaluationRunConfig)
    bootstrap_resamples: StrictInt = Field(default=2000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    random_seed: StrictInt = 0

    @field_validator("experiment_id", "baseline_variant")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty or whitespace-only.")
        return value.strip()


class PairedMetricComparison(BaseModel):
    """Paired means, delta, and bootstrap interval for one evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    paired_query_count: int = Field(..., ge=0)


class VariantComparison(BaseModel):
    """Comparison between one candidate and the declared baseline."""

    model_config = ConfigDict(extra="forbid")

    baseline_variant: str
    candidate_variant: str
    status: ComparisonStatus
    reason: str | None = None
    metrics: dict[str, PairedMetricComparison] = Field(default_factory=dict)


class ExperimentVariantResult(BaseModel):
    """Recorded configuration and evaluation outcome for one variant."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pipeline_config: dict[str, Any]
    status: VariantStatus
    evaluation_report: EvaluationReport | None = None
    error: str | None = None


class ExperimentReport(BaseModel):
    """Machine-readable record of all variants and baseline comparisons."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    dataset_id: str
    dataset_sha256: str
    created_at: datetime
    python_version: str
    platform: str
    config: ExperimentConfig
    status: ExperimentStatus
    variants: list[ExperimentVariantResult]
    comparisons: list[VariantComparison]


def paired_bootstrap_interval(
    differences: list[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the mean paired difference."""
    if not differences:
        raise ValueError("At least one paired difference is required.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Paired differences must be finite numbers.")
    if len(differences) == 1:
        return differences[0], differences[0]

    rng = random.Random(seed)
    sample_size = len(differences)
    bootstrap_means = sorted(
        fmean(rng.choices(differences, k=sample_size)) for _ in range(resamples)
    )
    tail = (1.0 - confidence_level) / 2.0
    return (
        _linear_quantile(bootstrap_means, tail),
        _linear_quantile(bootstrap_means, 1.0 - tail),
    )


def compare_evaluation_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    confidence_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, PairedMetricComparison]:
    """Compare matching reports by paired query IDs; positive deltas favor higher metrics.

    Latency metrics are included as measured values: a positive latency delta means
    the candidate was slower. This function does not label a variant as improved.
    """
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        raise EvaluationError("Cannot compare runs from different dataset byte digests.")
    if baseline.dataset_id != candidate.dataset_id:
        raise EvaluationError("Cannot compare runs with different dataset IDs.")

    baseline_queries = {query.query_id: query for query in baseline.queries}
    candidate_queries = {query.query_id: query for query in candidate.queries}
    if baseline_queries.keys() != candidate_queries.keys():
        raise EvaluationError("Cannot compare runs with different query ID sets.")

    metrics: dict[str, PairedMetricComparison] = {}
    metric_names = sorted(set(baseline.metrics) | set(candidate.metrics))
    for metric_name in metric_names:
        baseline_values: list[float] = []
        candidate_values: list[float] = []
        for query_id in sorted(baseline_queries):
            baseline_value = _query_metric(baseline_queries[query_id], metric_name)
            candidate_value = _query_metric(candidate_queries[query_id], metric_name)
            if baseline_value is not None and candidate_value is not None:
                baseline_values.append(baseline_value)
                candidate_values.append(candidate_value)

        if not baseline_values:
            metrics[metric_name] = PairedMetricComparison(
                baseline_mean=None,
                candidate_mean=None,
                delta=None,
                confidence_interval_low=None,
                confidence_interval_high=None,
                paired_query_count=0,
            )
            continue

        differences = [
            new - old for old, new in zip(baseline_values, candidate_values, strict=True)
        ]
        lower, upper = paired_bootstrap_interval(
            differences,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        )
        metrics[metric_name] = PairedMetricComparison(
            baseline_mean=fmean(baseline_values),
            candidate_mean=fmean(candidate_values),
            delta=fmean(differences),
            confidence_interval_low=lower,
            confidence_interval_high=upper,
            paired_query_count=len(differences),
        )
    return metrics


class ExperimentRunner:
    """Run named pipeline variants sequentially and persist comparable results."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(
        self,
        dataset: EvaluationDataset,
        variants: list[ExperimentVariant],
    ) -> ExperimentReport:
        """Evaluate each variant against the same dataset and configured cutoffs."""
        if not variants:
            raise EvaluationError("At least one experiment variant is required.")
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise EvaluationError("Experiment variant names must be unique.")
        if self.config.baseline_variant not in names:
            raise EvaluationError(
                f"Baseline variant '{self.config.baseline_variant}' was not provided."
            )

        runner = EvaluationRunner(self.config.evaluation)
        variant_results: list[ExperimentVariantResult] = []
        for variant in variants:
            try:
                evaluation_report = runner.run(dataset, variant.retriever, variant.answerer)
            except RAGEvalError as exc:
                variant_results.append(
                    ExperimentVariantResult(
                        name=variant.name,
                        pipeline_config=variant.pipeline_config,
                        status="failed",
                        error=str(exc),
                    )
                )
            else:
                variant_results.append(
                    ExperimentVariantResult(
                        name=variant.name,
                        pipeline_config=variant.pipeline_config,
                        status=evaluation_report.status,
                        evaluation_report=evaluation_report,
                    )
                )

        result_by_name = {result.name: result for result in variant_results}
        baseline_result = result_by_name[self.config.baseline_variant]
        comparisons: list[VariantComparison] = []
        for candidate in variant_results:
            if candidate.name == self.config.baseline_variant:
                continue
            if baseline_result.evaluation_report is None:
                comparisons.append(
                    VariantComparison(
                        baseline_variant=baseline_result.name,
                        candidate_variant=candidate.name,
                        status="unavailable",
                        reason="Baseline evaluation did not produce a report.",
                    )
                )
            elif candidate.evaluation_report is None:
                comparisons.append(
                    VariantComparison(
                        baseline_variant=baseline_result.name,
                        candidate_variant=candidate.name,
                        status="unavailable",
                        reason=candidate.error or "Candidate evaluation did not produce a report.",
                    )
                )
            else:
                comparisons.append(
                    VariantComparison(
                        baseline_variant=baseline_result.name,
                        candidate_variant=candidate.name,
                        status="complete",
                        metrics=compare_evaluation_reports(
                            baseline_result.evaluation_report,
                            candidate.evaluation_report,
                            confidence_level=self.config.confidence_level,
                            resamples=self.config.bootstrap_resamples,
                            seed=self.config.random_seed,
                        ),
                    )
                )

        if all(result.evaluation_report is None for result in variant_results):
            status: ExperimentStatus = "failed"
        elif all(result.status == "complete" for result in variant_results) and all(
            comparison.status == "complete" for comparison in comparisons
        ):
            status = "complete"
        else:
            status = "partial"

        return ExperimentReport(
            experiment_id=self.config.experiment_id,
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.sha256,
            created_at=datetime.now(UTC),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            config=self.config,
            status=status,
            variants=variant_results,
            comparisons=comparisons,
        )

    def write_report(self, report: ExperimentReport, file_path: str | Path) -> Path:
        """Write an indented UTF-8 JSON experiment record."""
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise EvaluationError(f"Failed to write experiment report '{path}': {exc}") from exc
        return path


def _query_metric(query: Any, metric_name: str) -> float | None:
    if metric_name.startswith("retrieval."):
        return query.retrieval_metrics.get(metric_name.removeprefix("retrieval."))
    if metric_name.startswith("answer."):
        return query.answer_metrics.get(metric_name.removeprefix("answer."))
    if metric_name == "latency.retrieval_ms":
        return query.retrieval_duration_ms
    if metric_name == "latency.generation_ms":
        return query.generation_duration_ms
    return None


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
