"""Deterministic tests for evidence-linked evaluation diagnoses."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import RetrievalError
from core.generate import GroundedAnswerGenerator
from core.llm import GenerationSettings
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
from eval.datasets import EvaluationCase, EvaluationDataset
from eval.diagnostics import DiagnosisConfig, DiagnosisReport, EvaluationDiagnoser
from eval.runners import EvaluationRunConfig, EvaluationRunner


def result(chunk_id: str, rank: int = 0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            text="Evidence text.",
            chunk_index=rank,
        ),
        score=1.0 - rank * 0.1,
        rank=rank,
        strategy=RetrievalStrategy.DENSE,
    )


class FakeRetriever:
    model_id = "fake:retriever"

    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if query == "retrieval error":
            raise RetrievalError("fixture index failure")
        return self.results.get(query, [])[:top_k]


class FakeProvider:
    model_id = "fake:generator"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        if "generation error" in user_prompt:
            raise RuntimeError("fixture provider failure")
        return "unrelated words [S99]"


def build_report():
    cases = [
        EvaluationCase(
            query_id="q-miss",
            query="retrieval miss",
            relevant_chunk_ids=["wanted-miss"],
            reference_answer="expected response",
        ),
        EvaluationCase(
            query_id="q-partial",
            query="partial retrieval",
            relevant_chunk_ids=["wanted-a", "wanted-b"],
            reference_answer="expected response",
        ),
        EvaluationCase(
            query_id="q-generation",
            query="generation error",
            relevant_chunk_ids=["wanted-c"],
            reference_answer="expected response",
        ),
        EvaluationCase(
            query_id="q-error",
            query="retrieval error",
            relevant_chunk_ids=["wanted-d"],
            reference_answer="expected response",
        ),
    ]
    dataset = EvaluationDataset(
        dataset_id="diagnosis-fixture",
        sha256="e" * 64,
        cases=cases,
    )
    retriever = FakeRetriever(
        {
            "retrieval miss": [result("noise")],
            "partial retrieval": [result("wanted-a"), result("noise", 1)],
            "generation error": [result("wanted-c")],
        }
    )
    answerer = GroundedAnswerGenerator(retriever, FakeProvider())
    return EvaluationRunner(EvaluationRunConfig(cutoffs=(1, 2))).run(dataset, retriever, answerer)


def test_diagnoser_groups_observed_failures_and_links_query_evidence(
    tmp_path: Path,
) -> None:
    evaluation = build_report()

    diagnosis = EvaluationDiagnoser().diagnose(evaluation)

    findings = {finding.category: finding for finding in diagnosis.findings}
    assert diagnosis.dataset_sha256 == "e" * 64
    assert findings["retrieval_miss"].affected_query_ids == ["q-miss"]
    assert findings["retrieval_partial"].affected_query_ids == ["q-partial"]
    assert findings["retrieval_error"].affected_query_ids == ["q-error"]
    assert findings["generation_error"].affected_query_ids == ["q-generation"]
    assert findings["low_lexical_overlap"].query_count == 2
    assert findings["unresolved_citation"].query_count == 2
    assert findings["retrieval_partial"].observed_metrics == {"recall@2": 0.5}
    assert findings["retrieval_partial"].rate == pytest.approx(0.25)
    assert "heuristic" in findings["low_lexical_overlap"].recommendation

    destination = EvaluationDiagnoser().write_report(
        diagnosis, tmp_path / "reports" / "diagnosis.json"
    )
    loaded = DiagnosisReport.model_validate_json(destination.read_text(encoding="utf-8"))
    assert loaded == diagnosis


def test_answer_unavailable_is_reported_when_references_have_no_answerer() -> None:
    dataset = EvaluationDataset(
        dataset_id="answer-unavailable",
        sha256="f" * 64,
        cases=[EvaluationCase(query_id="q-1", query="question", reference_answer="answer")],
    )
    evaluation = EvaluationRunner(EvaluationRunConfig(cutoffs=(1,))).run(
        dataset, FakeRetriever({"question": []})
    )

    diagnosis = EvaluationDiagnoser().diagnose(evaluation)

    assert [finding.category for finding in diagnosis.findings] == ["answer_unavailable"]
    assert diagnosis.findings[0].affected_query_ids == ["q-1"]


def test_low_overlap_uses_explicit_configured_threshold() -> None:
    diagnosis = EvaluationDiagnoser(DiagnosisConfig(low_token_f1_threshold=0.0)).diagnose(
        build_report()
    )

    assert all(f.category != "low_lexical_overlap" for f in diagnosis.findings)


def test_diagnosis_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValidationError):
        DiagnosisConfig(low_token_f1_threshold=1.1)
