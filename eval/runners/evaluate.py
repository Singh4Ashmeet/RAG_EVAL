"""Evaluation runner producing per-query results and aggregate JSON metrics."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.errors import EvaluationError
from core.generate.pipeline import GroundedAnswerGenerator
from core.generate.schema import AnswerCitation
from core.llm.base import GenerationSettings
from core.retrieval.base import Retriever
from core.retrieval.schema import SearchResult
from eval.datasets.schema import EvaluationCase, EvaluationDataset
from eval.metrics.generation import exact_match, token_f1
from eval.metrics.retrieval import (
    average_precision_at_k,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    source_element_hit_rate_at_k,
    source_element_recall_at_k,
    source_element_reciprocal_rank_at_k,
)
from eval.runners.run_config import EvaluationRunConfig

QueryStatus = Literal["success", "partial", "failed"]
RunStatus = Literal["complete", "partial", "failed"]


class QueryEvaluationResult(BaseModel):
    """One query's retrieved evidence, applicable metrics, and failures."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    status: QueryStatus
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_element_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    retrieved_sources: list[SearchResult] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    retrieval_duration_ms: float | None = Field(default=None, ge=0.0)
    generated_answer: str | None = None
    generation_duration_ms: float | None = Field(default=None, ge=0.0)
    citations: list[AnswerCitation] = Field(default_factory=list)
    unresolved_citation_labels: list[str] = Field(default_factory=list)
    answer_metrics: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Machine-readable reproducibility record for one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_sha256: str
    created_at: datetime
    config: EvaluationRunConfig
    retrieval_model_id: str | None = None
    generation_model_id: str | None = None
    generation_settings: GenerationSettings | None = None
    status: RunStatus
    metrics: dict[str, float | None]
    metric_counts: dict[str, int]
    queries: list[QueryEvaluationResult]


class EvaluationRunner:
    """Evaluate retrieval and optional generated answers against a validated dataset."""

    def __init__(self, config: EvaluationRunConfig | None = None) -> None:
        self.config = config or EvaluationRunConfig()

    def run(
        self,
        dataset: EvaluationDataset,
        retriever: Retriever,
        answerer: GroundedAnswerGenerator | None = None,
        *,
        on_query_complete: Callable[[int, int, QueryEvaluationResult], None] | None = None,
    ) -> EvaluationReport:
        """Run each query once through retrieval and optionally generate from that result."""
        query_results: list[QueryEvaluationResult] = []
        metric_values: dict[str, list[float]] = {}
        total_cases = len(dataset.cases)

        for index, case in enumerate(dataset.cases, start=1):
            result = self._evaluate_case(case, retriever, answerer)
            query_results.append(result)
            if on_query_complete is not None:
                on_query_complete(index, total_cases, result)
            for name, value in result.retrieval_metrics.items():
                metric_values.setdefault(f"retrieval.{name}", []).append(value)
            for name, value in result.answer_metrics.items():
                metric_values.setdefault(f"answer.{name}", []).append(value)
            if result.retrieval_duration_ms is not None:
                metric_values.setdefault("latency.retrieval_ms", []).append(
                    result.retrieval_duration_ms
                )
            if result.generation_duration_ms is not None:
                metric_values.setdefault("latency.generation_ms", []).append(
                    result.generation_duration_ms
                )

        metric_names = (
            [
                f"retrieval.{metric}@{cutoff}"
                for cutoff in self.config.cutoffs
                for metric in ("recall", "precision", "mrr", "map", "hit_rate")
            ]
            + [
                f"retrieval.source_element_{metric}@{cutoff}"
                for cutoff in self.config.cutoffs
                for metric in ("recall", "mrr", "hit_rate")
            ]
            + [
                "answer.exact_match",
                "answer.token_f1",
                "latency.retrieval_ms",
                "latency.generation_ms",
            ]
        )
        metrics = {
            name: fmean(metric_values[name]) if metric_values.get(name) else None
            for name in metric_names
        }
        metric_counts = {name: len(metric_values.get(name, [])) for name in metric_names}

        errored_queries = sum(bool(query.errors) for query in query_results)
        if errored_queries == 0:
            status: RunStatus = "complete"
        elif all(query.status == "failed" for query in query_results):
            status = "failed"
        else:
            status = "partial"

        return EvaluationReport(
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.sha256,
            created_at=datetime.now(UTC),
            config=self.config,
            retrieval_model_id=self._retrieval_model_id(retriever),
            generation_model_id=answerer.provider.model_id if answerer is not None else None,
            generation_settings=answerer.settings if answerer is not None else None,
            status=status,
            metrics=metrics,
            metric_counts=metric_counts,
            queries=query_results,
        )

    def write_report(self, report: EvaluationReport, file_path: str | Path) -> Path:
        """Write an indented UTF-8 JSON report, creating its parent directory."""
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise EvaluationError(f"Failed to write evaluation report '{path}': {exc}") from exc
        return path

    def _evaluate_case(
        self,
        case: EvaluationCase,
        retriever: Retriever,
        answerer: GroundedAnswerGenerator | None,
    ) -> QueryEvaluationResult:
        errors: list[str] = []
        retrieval_metrics: dict[str, float] = {}
        answer_metrics: dict[str, float] = {}
        retrieved_sources: list[SearchResult] = []
        generated_answer: str | None = None
        citations: list[AnswerCitation] = []
        unresolved_citations: list[str] = []
        retrieval_duration_ms: float | None = None
        generation_duration_ms: float | None = None
        retrieval_succeeded = False

        retrieval_started = perf_counter()
        try:
            retrieved_sources = retriever.retrieve(case.query, top_k=self.config.max_cutoff)
            retrieval_succeeded = True
        except Exception as exc:
            errors.append(f"retrieval: {exc}")
        finally:
            retrieval_duration_ms = max(0.0, (perf_counter() - retrieval_started) * 1000.0)

        if retrieval_succeeded and case.relevant_chunk_ids:
            retrieved_ids = [result.chunk.chunk_id for result in retrieved_sources]
            for cutoff in self.config.cutoffs:
                retrieval_metrics.update(
                    {
                        f"recall@{cutoff}": recall_at_k(
                            retrieved_ids, case.relevant_chunk_ids, cutoff
                        ),
                        f"precision@{cutoff}": precision_at_k(
                            retrieved_ids, case.relevant_chunk_ids, cutoff
                        ),
                        f"mrr@{cutoff}": reciprocal_rank_at_k(
                            retrieved_ids, case.relevant_chunk_ids, cutoff
                        ),
                        f"map@{cutoff}": average_precision_at_k(
                            retrieved_ids, case.relevant_chunk_ids, cutoff
                        ),
                        f"hit_rate@{cutoff}": hit_rate_at_k(
                            retrieved_ids, case.relevant_chunk_ids, cutoff
                        ),
                    }
                )

        if retrieval_succeeded and case.relevant_element_ids:
            retrieved_chunks = [result.chunk for result in retrieved_sources]
            for cutoff in self.config.cutoffs:
                retrieval_metrics.update(
                    {
                        f"source_element_recall@{cutoff}": source_element_recall_at_k(
                            retrieved_chunks, case.relevant_element_ids, cutoff
                        ),
                        f"source_element_mrr@{cutoff}": source_element_reciprocal_rank_at_k(
                            retrieved_chunks, case.relevant_element_ids, cutoff
                        ),
                        f"source_element_hit_rate@{cutoff}": source_element_hit_rate_at_k(
                            retrieved_chunks, case.relevant_element_ids, cutoff
                        ),
                    }
                )

        if retrieval_succeeded and answerer is not None:
            generation_started = perf_counter()
            try:
                generated = answerer.answer_with_sources(case.query, retrieved_sources)
                generated_answer = generated.answer
                citations = generated.citations
                unresolved_citations = generated.unresolved_citation_labels
                if case.reference_answer is not None:
                    answer_metrics = {
                        "exact_match": exact_match(generated.answer, case.reference_answer),
                        "token_f1": token_f1(generated.answer, case.reference_answer),
                    }
            except Exception as exc:
                errors.append(f"generation: {exc}")
            finally:
                generation_duration_ms = max(0.0, (perf_counter() - generation_started) * 1000.0)
        elif retrieval_succeeded and case.reference_answer is not None:
            errors.append("answer metrics unavailable: no answer generator was configured.")

        if not retrieval_succeeded:
            status: QueryStatus = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return QueryEvaluationResult(
            query_id=case.query_id,
            query=case.query,
            status=status,
            relevant_chunk_ids=case.relevant_chunk_ids,
            relevant_element_ids=case.relevant_element_ids,
            reference_answer=case.reference_answer,
            retrieved_sources=retrieved_sources,
            retrieval_metrics=retrieval_metrics,
            retrieval_duration_ms=retrieval_duration_ms,
            generated_answer=generated_answer,
            generation_duration_ms=generation_duration_ms,
            citations=citations,
            unresolved_citation_labels=unresolved_citations,
            answer_metrics=answer_metrics,
            errors=errors,
        )

    @staticmethod
    def _retrieval_model_id(retriever: Retriever) -> str | None:
        model_id = getattr(retriever, "model_id", None)
        return model_id if isinstance(model_id, str) and model_id.strip() else None
