"""Tests for evaluation aggregation, per-query failures, and JSON report output."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import EvaluationError, RetrievalError
from core.generate import GroundedAnswerGenerator
from core.llm import GenerationSettings
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
from eval.datasets import EvaluationCase, EvaluationDataset
from eval.runners import EvaluationRunConfig, EvaluationRunner


@pytest.mark.parametrize("cutoffs", [(0,), (-1,), (True,), (1.5,), (1, 1)])
def test_run_config_rejects_invalid_or_duplicate_cutoffs(cutoffs: tuple[object, ...]) -> None:
    with pytest.raises(ValidationError):
        EvaluationRunConfig(cutoffs=cutoffs)  # type: ignore[arg-type]


def make_result(chunk_id: str, document_id: str, rank: int, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_index=rank,
        ),
        score=1.0 - rank * 0.1,
        rank=rank,
        strategy=RetrievalStrategy.DENSE,
    )


class FakeRetriever:
    model_id = "test:retrieval-v1"

    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []
        self.fail_queries: set[str] = set()

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if query in self.fail_queries:
            raise RetrievalError("index unavailable")
        return self.results_by_query.get(query, [])[:top_k]


class FakeProvider:
    model_id = "test:generation-v1"

    def __init__(self, answer: str = "The sky is blue [S1].") -> None:
        self.answer_text = answer

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        return self.answer_text


def make_dataset(*cases: EvaluationCase) -> EvaluationDataset:
    return EvaluationDataset(dataset_id="test-set", sha256="b" * 64, cases=list(cases))


def test_runner_computes_metrics_and_reuses_one_retrieval_for_generation() -> None:
    case = EvaluationCase(
        query_id="q-1",
        query="What color is the sky?",
        relevant_chunk_ids=["chunk-1"],
        reference_answer="The sky is blue.",
    )
    retrieved = [make_result("chunk-1", "doc-1", 0, "The sky is blue.")]
    retriever = FakeRetriever({case.query: retrieved})
    answerer = GroundedAnswerGenerator(retriever, FakeProvider())
    runner = EvaluationRunner(EvaluationRunConfig(cutoffs=(1, 3)))

    report = runner.run(make_dataset(case), retriever, answerer)

    assert report.status == "complete"
    assert report.dataset_sha256 == "b" * 64
    assert report.retrieval_model_id == "test:retrieval-v1"
    assert report.generation_model_id == "test:generation-v1"
    assert report.generation_settings == GenerationSettings(temperature=0.0, max_tokens=512)
    assert report.metrics["retrieval.recall@1"] == 1.0
    assert report.metrics["retrieval.precision@3"] == pytest.approx(1 / 3)
    assert report.metrics["answer.exact_match"] == 1.0
    assert report.metrics["answer.token_f1"] == 1.0
    assert report.metric_counts["answer.exact_match"] == 1
    assert retriever.calls == [(case.query, 3)]
    assert report.queries[0].generated_answer == "The sky is blue [S1]."
    assert report.queries[0].citations[0].chunk_id == "chunk-1"


def test_answer_only_case_runs_generation_without_retrieval_metrics() -> None:
    case = EvaluationCase(query_id="q-1", query="say blue", reference_answer="blue")
    retriever = FakeRetriever({case.query: [make_result("c1", "d1", 0, "blue evidence")]})
    answerer = GroundedAnswerGenerator(retriever, FakeProvider("blue"))

    report = EvaluationRunner(EvaluationRunConfig(cutoffs=(2,))).run(
        make_dataset(case), retriever, answerer
    )

    assert report.status == "complete"
    assert report.metrics["retrieval.recall@2"] is None
    assert report.metric_counts["retrieval.recall@2"] == 0
    assert report.metrics["answer.exact_match"] == 1.0


def test_missing_answerer_marks_answer_reference_as_incomplete() -> None:
    case = EvaluationCase(query_id="q-1", query="question", reference_answer="answer")
    retriever = FakeRetriever({case.query: []})

    report = EvaluationRunner().run(make_dataset(case), retriever)

    assert report.status == "partial"
    assert report.queries[0].status == "partial"
    assert report.metrics["answer.exact_match"] is None
    assert "no answer generator" in report.queries[0].errors[0]


def test_known_retrieval_failure_is_recorded_and_makes_run_failed() -> None:
    case = EvaluationCase(
        query_id="q-1",
        query="question",
        relevant_chunk_ids=["chunk-1"],
    )
    retriever = FakeRetriever({})
    retriever.fail_queries.add(case.query)

    report = EvaluationRunner().run(make_dataset(case), retriever)

    assert report.status == "failed"
    assert report.queries[0].status == "failed"
    assert report.queries[0].retrieval_metrics == {}
    assert report.queries[0].errors == ["retrieval: index unavailable"]


def test_write_report_creates_parent_and_round_trips_machine_readable_json(
    tmp_path: Path,
) -> None:
    case = EvaluationCase(query_id="q-1", query="question", relevant_chunk_ids=["c1"])
    retriever = FakeRetriever({case.query: [make_result("c1", "d1", 0, "evidence")]})
    runner = EvaluationRunner(EvaluationRunConfig(cutoffs=(1,)))
    report = runner.run(make_dataset(case), retriever)
    destination = tmp_path / "reports" / "run.json"

    returned_path = runner.write_report(report, destination)

    assert returned_path == destination
    assert destination.exists()
    assert '"dataset_sha256": "bbbb' in destination.read_text(encoding="utf-8")
    loaded_report = type(report).model_validate_json(destination.read_text(encoding="utf-8"))
    assert loaded_report.dataset_sha256 == report.dataset_sha256
    assert loaded_report.queries[0].retrieved_sources[0].chunk.chunk_id == "c1"


def test_report_write_failure_is_explicit(tmp_path: Path) -> None:
    case = EvaluationCase(query_id="q-1", query="question", relevant_chunk_ids=["c1"])
    report = EvaluationRunner().run(make_dataset(case), FakeRetriever({}))
    blocked_path = tmp_path / "file"
    blocked_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Failed to write"):
        EvaluationRunner().write_report(report, blocked_path / "run.json")
