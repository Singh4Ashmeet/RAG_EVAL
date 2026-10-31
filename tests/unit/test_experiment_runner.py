"""Deterministic tests for experiment sweeps and paired metric comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import EvaluationError
from core.generate import GroundedAnswerGenerator
from core.llm import GenerationSettings
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
from eval.datasets import EvaluationCase, EvaluationDataset
from eval.runners import (
    ExperimentConfig,
    ExperimentReport,
    ExperimentRunner,
    ExperimentVariant,
    compare_evaluation_reports,
    paired_bootstrap_interval,
)


def make_result(chunk_id: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            text=f"Evidence {chunk_id}.",
            chunk_index=rank,
        ),
        score=1.0 - rank * 0.1,
        rank=rank,
        strategy=RetrievalStrategy.DENSE,
    )


class VariantRetriever:
    def __init__(self, model_id: str, results: dict[str, list[SearchResult]]) -> None:
        self.model_id = model_id
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results.get(query, [])[:top_k]


class FixedProvider:
    model_id = "test:generator-v1"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: GenerationSettings,
    ) -> str:
        return "The answer is supported [S1]."


def make_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset_id="experiment-fixture",
        sha256="c" * 64,
        cases=[
            EvaluationCase(
                query_id="q-1",
                query="question one",
                relevant_chunk_ids=["relevant-1"],
                reference_answer="The answer is supported.",
            ),
            EvaluationCase(
                query_id="q-2",
                query="question two",
                relevant_chunk_ids=["relevant-2"],
                reference_answer="The answer is supported.",
            ),
            EvaluationCase(
                query_id="q-3",
                query="question three",
                relevant_chunk_ids=["relevant-3"],
                reference_answer="The answer is supported.",
            ),
        ],
    )


def make_variant(
    name: str,
    results: dict[str, list[SearchResult]],
) -> tuple[ExperimentVariant, VariantRetriever]:
    retriever = VariantRetriever(f"test:{name}", results)
    answerer = GroundedAnswerGenerator(retriever, FixedProvider())
    return (
        ExperimentVariant(
            name=name,
            pipeline_config={"chunk_size": 500 if name == "baseline" else 300},
            retriever=retriever,
            answerer=answerer,
        ),
        retriever,
    )


def test_sweep_records_variants_configs_environment_and_paired_comparison(
    tmp_path: Path,
) -> None:
    baseline, baseline_retriever = make_variant(
        "baseline",
        {
            "question one": [make_result("noise", 0), make_result("relevant-1", 1)],
            "question two": [make_result("relevant-2", 0)],
            "question three": [make_result("noise", 0)],
        },
    )
    candidate, candidate_retriever = make_variant(
        "smaller-chunks",
        {
            "question one": [make_result("relevant-1", 0)],
            "question two": [make_result("relevant-2", 0)],
            "question three": [make_result("relevant-3", 0)],
        },
    )
    config = ExperimentConfig(
        experiment_id="chunk-size-comparison",
        baseline_variant="baseline",
        evaluation={"cutoffs": (1, 2)},
        bootstrap_resamples=300,
        random_seed=17,
    )
    runner = ExperimentRunner(config)

    report = runner.run(make_dataset(), [baseline, candidate])

    assert report.status == "complete"
    assert report.dataset_sha256 == "c" * 64
    assert report.python_version
    assert report.platform
    assert report.variants[1].pipeline_config == {"chunk_size": 300}
    assert report.variants[0].evaluation_report is not None
    assert report.variants[0].evaluation_report.generation_settings == GenerationSettings()
    assert baseline_retriever.calls == [
        ("question one", 2),
        ("question two", 2),
        ("question three", 2),
    ]
    assert candidate_retriever.calls == baseline_retriever.calls

    comparison = report.comparisons[0]
    assert comparison.status == "complete"
    recall = comparison.metrics["retrieval.recall@1"]
    assert recall.baseline_mean == pytest.approx(1 / 3)
    assert recall.candidate_mean == 1.0
    assert recall.delta == pytest.approx(2 / 3)
    assert recall.paired_query_count == 3
    assert recall.confidence_interval_low is not None
    assert recall.confidence_interval_high is not None
    assert "latency.retrieval_ms" in comparison.metrics
    assert "latency.generation_ms" in comparison.metrics

    destination = runner.write_report(report, tmp_path / "experiments" / "result.json")
    loaded = ExperimentReport.model_validate_json(destination.read_text(encoding="utf-8"))
    assert loaded.experiment_id == "chunk-size-comparison"
    assert loaded.comparisons[0].metrics["retrieval.recall@1"].delta == pytest.approx(2 / 3)


def test_bootstrap_interval_is_deterministic_and_handles_single_pair() -> None:
    differences = [0.5, -0.25, 0.75, 0.0]
    first = paired_bootstrap_interval(differences, resamples=500, seed=8)
    second = paired_bootstrap_interval(differences, resamples=500, seed=8)
    assert first == second
    assert paired_bootstrap_interval([0.25], resamples=100, seed=9) == (0.25, 0.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"resamples": 0},
        {"seed": True},
    ],
)
def test_bootstrap_configuration_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_interval([0.1, 0.2], **kwargs)  # type: ignore[arg-type]


def test_comparison_rejects_different_dataset_hashes() -> None:
    baseline, baseline_retriever = make_variant("baseline", {"question one": []})
    config = ExperimentConfig(
        experiment_id="same-data",
        baseline_variant="baseline",
        evaluation={"cutoffs": (1,)},
    )
    report = ExperimentRunner(config).run(make_dataset(), [baseline]).variants[0].evaluation_report
    assert report is not None
    other = report.model_copy(update={"dataset_sha256": "d" * 64})
    with pytest.raises(EvaluationError, match="different dataset"):
        compare_evaluation_reports(report, other)
    assert baseline_retriever.calls


def test_sweep_requires_unique_names_and_the_named_baseline() -> None:
    baseline, _ = make_variant("baseline", {})
    config = ExperimentConfig(experiment_id="bad", baseline_variant="baseline")
    runner = ExperimentRunner(config)
    with pytest.raises(EvaluationError, match="unique"):
        runner.run(make_dataset(), [baseline, baseline])
    with pytest.raises(EvaluationError, match="was not provided"):
        runner.run(make_dataset(), [make_variant("candidate", {})[0]])


def test_variant_configuration_must_be_json_safe() -> None:
    retriever = VariantRetriever("test:r", {})
    with pytest.raises(ValueError, match="JSON"):
        ExperimentVariant("bad", {"bad": object()}, retriever)


def test_failed_variant_is_reported_without_losing_other_results() -> None:
    baseline, _ = make_variant("baseline", {"question one": [make_result("relevant-1", 0)]})
    candidate = ExperimentVariant(
        name="candidate",
        pipeline_config={"strategy": "dense"},
        retriever=VariantRetriever("test:candidate", {}),
    )
    report = ExperimentRunner(
        ExperimentConfig(
            experiment_id="partial",
            baseline_variant="baseline",
            evaluation={"cutoffs": (1,)},
        )
    ).run(make_dataset(), [baseline, candidate])

    assert report.status == "partial"
    assert report.variants[0].evaluation_report is not None
    assert report.variants[1].status == "partial"
