"""Evidence-based, deterministic diagnoses for evaluation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.errors import EvaluationError
from eval.runners.evaluate import EvaluationReport

DiagnosisCategory = Literal[
    "retrieval_error",
    "retrieval_miss",
    "retrieval_partial",
    "generation_error",
    "answer_unavailable",
    "low_lexical_overlap",
    "unresolved_citation",
]


class DiagnosisFinding(BaseModel):
    """A recurring observation tied to measured fields and affected queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: DiagnosisCategory
    affected_query_ids: list[str] = Field(..., min_length=1)
    query_count: int = Field(..., ge=1)
    dataset_query_count: int = Field(..., ge=1)
    rate: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(..., min_length=1)
    observed_metrics: dict[str, float] = Field(default_factory=dict)
    recommendation: str


class DiagnosisReport(BaseModel):
    """Diagnoses for one evaluation report, without making improvement claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_sha256: str
    evaluation_status: str
    query_count: int = Field(..., ge=1)
    findings: list[DiagnosisFinding]


class DiagnosisConfig(BaseModel):
    """Explicit thresholds for heuristic answer-quality diagnosis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low_token_f1_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationDiagnoser:
    """Group observed retrieval, answer, and citation issues from a run report."""

    def __init__(self, config: DiagnosisConfig | None = None) -> None:
        self.config = config or DiagnosisConfig()

    def diagnose(self, report: EvaluationReport) -> DiagnosisReport:
        """Return query-linked findings; thresholds are heuristics, not correctness tests."""
        query_count = len(report.queries)
        if query_count == 0:
            raise EvaluationError("Cannot diagnose an evaluation report with no queries.")

        groups: dict[DiagnosisCategory, list[tuple[str, list[str], dict[str, float]]]] = {
            category: []
            for category in (
                "retrieval_error",
                "retrieval_miss",
                "retrieval_partial",
                "generation_error",
                "answer_unavailable",
                "low_lexical_overlap",
                "unresolved_citation",
            )
        }
        max_cutoff = report.config.max_cutoff

        for query in report.queries:
            retrieval_errors = [error for error in query.errors if error.startswith("retrieval:")]
            generation_errors = [error for error in query.errors if error.startswith("generation:")]
            availability_errors = [
                error for error in query.errors if error.startswith("answer metrics unavailable:")
            ]

            if retrieval_errors:
                groups["retrieval_error"].append((query.query_id, retrieval_errors, {}))
            elif query.relevant_chunk_ids:
                metric_name = f"recall@{max_cutoff}"
                recall = query.retrieval_metrics.get(metric_name)
                if recall == 0.0:
                    groups["retrieval_miss"].append(
                        (
                            query.query_id,
                            [
                                f"{metric_name}=0.0; no relevant chunk retrieved "
                                "within configured depth"
                            ],
                            {metric_name: recall},
                        )
                    )
                elif recall is not None and recall < 1.0:
                    groups["retrieval_partial"].append(
                        (
                            query.query_id,
                            [f"{metric_name}={recall:.6g}; some relevant chunks were missed"],
                            {metric_name: recall},
                        )
                    )

            if generation_errors:
                groups["generation_error"].append((query.query_id, generation_errors, {}))
            if availability_errors:
                groups["answer_unavailable"].append((query.query_id, availability_errors, {}))

            token_f1_value = query.answer_metrics.get("token_f1")
            if token_f1_value is not None and token_f1_value < self.config.low_token_f1_threshold:
                groups["low_lexical_overlap"].append(
                    (
                        query.query_id,
                        [
                            "answer.token_f1="
                            f"{token_f1_value:.6g} is below configured heuristic threshold "
                            f"{self.config.low_token_f1_threshold:.6g}"
                        ],
                        {"answer.token_f1": token_f1_value},
                    )
                )

            if query.unresolved_citation_labels:
                groups["unresolved_citation"].append(
                    (
                        query.query_id,
                        [
                            "unresolved citation labels: "
                            + ", ".join(query.unresolved_citation_labels)
                        ],
                        {},
                    )
                )

        findings = [
            self._finding(category, entries, query_count)
            for category, entries in groups.items()
            if entries
        ]
        return DiagnosisReport(
            dataset_id=report.dataset_id,
            dataset_sha256=report.dataset_sha256,
            evaluation_status=report.status,
            query_count=query_count,
            findings=findings,
        )

    def write_report(self, report: DiagnosisReport, file_path: str | Path) -> Path:
        """Write a machine-readable UTF-8 JSON diagnosis report."""
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise EvaluationError(f"Failed to write diagnosis report '{path}': {exc}") from exc
        return path

    @staticmethod
    def _finding(
        category: DiagnosisCategory,
        entries: list[tuple[str, list[str], dict[str, float]]],
        dataset_query_count: int,
    ) -> DiagnosisFinding:
        recommendations: dict[DiagnosisCategory, str] = {
            "retrieval_error": (
                "Inspect the retriever error and local index/model availability before "
                "comparing retrieval quality."
            ),
            "retrieval_miss": (
                "Inspect the query and relevant chunk labels, then compare retrieval "
                "or chunking variants in a controlled sweep."
            ),
            "retrieval_partial": (
                "Inspect missed relevant chunks and compare retrieval depth or "
                "chunking variants on the same dataset."
            ),
            "generation_error": (
                "Inspect the generation error and provider availability; rerun only "
                "after the cause is understood."
            ),
            "answer_unavailable": (
                "Configure an answer generator and rerun if answer quality is in "
                "scope for this evaluation."
            ),
            "low_lexical_overlap": (
                "Review the answer against its reference and retrieved evidence; "
                "lexical overlap is a heuristic, not a factuality judgment."
            ),
            "unresolved_citation": (
                "Review citation labels and the prompt/output format; resolution "
                "checks label mapping, not whether a source supports a claim."
            ),
        }
        metrics: dict[str, list[float]] = {}
        for _, _, entry_metrics in entries:
            for name, value in entry_metrics.items():
                metrics.setdefault(name, []).append(value)
        return DiagnosisFinding(
            category=category,
            affected_query_ids=[query_id for query_id, _, _ in entries],
            query_count=len(entries),
            dataset_query_count=dataset_query_count,
            rate=len(entries) / dataset_query_count,
            evidence=[
                f"{query_id}: {observation}"
                for query_id, observations, _ in entries
                for observation in observations
            ],
            observed_metrics={name: sum(values) / len(values) for name, values in metrics.items()},
            recommendation=recommendations[category],
        )
