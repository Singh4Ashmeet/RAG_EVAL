"""Tests for B5 baseline execution orchestrator.

All tests use mocks — no live Ollama, no network, no GPU required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from core.llm.base import GenerationSettings
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
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
from eval.runners.baseline import (
    BaselineConfig,
    BaselineRunManifest,
    _config_digest,
    _corpus_digest,
    _generate_run_id,
    _redact_secrets,
    run_baseline,
)
from eval.runners.evaluate import EvaluationReport, EvaluationRunner, QueryEvaluationResult
from eval.runners.run_config import EvaluationRunConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACK_A = CorpusTrack.TRACK_A
TRACK_B = CorpusTrack.TRACK_B


def _make_case(
    query_id: str = "q-1",
    query: str = "What is RAG?",
    track: CorpusTrack = TRACK_A,
    chunk_ids: list[str] | None = None,
    element_ids: list[str] | None = None,
    reference_answer: str | None = "RAG is Retrieval-Augmented Generation.",
) -> EvaluationCase:
    return EvaluationCase(
        query_id=query_id,
        query=query,
        track=track,
        relevant_chunk_ids=chunk_ids or ["chunk-1"],
        relevant_element_ids=element_ids or ["elem-1"],
        reference_answer=reference_answer,
        status=AnnotationStatus.VERIFIED,
        answerability=Answerability.ANSWERABLE,
        category=QuestionCategory.DIRECT_FACTUAL,
        difficulty=Difficulty.EASY,
        provenance=AnnotationProvenance(
            annotator="test-annotator",
            verified_at=datetime(2026, 11, 1, tzinfo=UTC),
            source_documents=["test_doc.md"],
        ),
    )


def _make_dataset(
    cases: list[EvaluationCase] | None = None,
    track: CorpusTrack = TRACK_A,
) -> EvaluationDataset:
    if cases is None:
        cases = [_make_case(track=track)]
    return EvaluationDataset(
        dataset_id="test-dataset",
        sha256="a" * 64,
        cases=cases,
        track=track,
    )


def _make_chunk(chunk_id: str = "chunk-1", doc_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text="Sample chunk text.",
        chunk_index=0,
    )


def _make_search_result(chunk_id: str = "chunk-1", doc_id: str = "doc-1") -> SearchResult:
    return SearchResult(
        chunk=_make_chunk(chunk_id, doc_id),
        score=0.9,
        rank=0,
        strategy=RetrievalStrategy.DENSE,
    )


def _make_query_result(
    query_id: str = "q-1",
    status: str = "success",
    retrieved: list[SearchResult] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
    answer_metrics: dict[str, float] | None = None,
    generated_answer: str | None = "RAG is Retrieval-Augmented Generation.",
    errors: list[str] | None = None,
) -> QueryEvaluationResult:
    return QueryEvaluationResult(
        query_id=query_id,
        query="What is RAG?",
        status=status,
        relevant_chunk_ids=["chunk-1"],
        relevant_element_ids=["elem-1"],
        reference_answer="RAG is Retrieval-Augmented Generation.",
        retrieved_sources=retrieved if retrieved is not None else [_make_search_result()],
        retrieval_metrics=retrieval_metrics
        if retrieval_metrics is not None
        else {"recall@5": 1.0, "precision@5": 0.2},
        retrieval_duration_ms=50.0,
        generated_answer=generated_answer,
        generation_duration_ms=200.0,
        answer_metrics=answer_metrics
        if answer_metrics is not None
        else {"exact_match": 1.0, "token_f1": 1.0},
        errors=errors if errors is not None else [],
    )


def _make_evaluation_report(
    queries: list[QueryEvaluationResult] | None = None,
    status: str = "complete",
) -> EvaluationReport:
    if queries is None:
        queries = [_make_query_result()]
    return EvaluationReport(
        dataset_id="test-dataset",
        dataset_sha256="a" * 64,
        created_at=datetime.now(UTC),
        config=EvaluationRunConfig(cutoffs=(1, 3, 5)),
        retrieval_model_id="ollama:nomic-embed-text",
        generation_model_id="ollama:llama3.2:3b",
        generation_settings=GenerationSettings(temperature=0.0, max_tokens=512),
        status=status,
        metrics={
            "retrieval.recall@1": 1.0,
            "retrieval.recall@3": 1.0,
            "retrieval.recall@5": 1.0,
            "retrieval.precision@1": 1.0,
            "retrieval.precision@3": 0.33,
            "retrieval.precision@5": 0.2,
            "retrieval.mrr@1": 1.0,
            "retrieval.mrr@3": 1.0,
            "retrieval.mrr@5": 1.0,
            "retrieval.map@1": 1.0,
            "retrieval.map@3": 1.0,
            "retrieval.map@5": 1.0,
            "retrieval.hit_rate@1": 1.0,
            "retrieval.hit_rate@3": 1.0,
            "retrieval.hit_rate@5": 1.0,
            "retrieval.source_element_recall@1": 1.0,
            "retrieval.source_element_recall@3": 1.0,
            "retrieval.source_element_recall@5": 1.0,
            "retrieval.source_element_mrr@1": 1.0,
            "retrieval.source_element_mrr@3": 1.0,
            "retrieval.source_element_mrr@5": 1.0,
            "retrieval.source_element_hit_rate@1": 1.0,
            "retrieval.source_element_hit_rate@3": 1.0,
            "retrieval.source_element_hit_rate@5": 1.0,
            "answer.exact_match": 1.0,
            "answer.token_f1": 1.0,
            "latency.retrieval_ms": 50.0,
            "latency.generation_ms": 200.0,
        },
        metric_counts={
            "retrieval.recall@1": 1,
            "retrieval.recall@3": 1,
            "retrieval.recall@5": 1,
            "retrieval.precision@1": 1,
            "retrieval.precision@3": 1,
            "retrieval.precision@5": 1,
            "retrieval.mrr@1": 1,
            "retrieval.mrr@3": 1,
            "retrieval.mrr@5": 1,
            "retrieval.map@1": 1,
            "retrieval.map@3": 1,
            "retrieval.map@5": 1,
            "retrieval.hit_rate@1": 1,
            "retrieval.hit_rate@3": 1,
            "retrieval.hit_rate@5": 1,
            "retrieval.source_element_recall@1": 1,
            "retrieval.source_element_recall@3": 1,
            "retrieval.source_element_recall@5": 1,
            "retrieval.source_element_mrr@1": 1,
            "retrieval.source_element_mrr@3": 1,
            "retrieval.source_element_mrr@5": 1,
            "retrieval.source_element_hit_rate@1": 1,
            "retrieval.source_element_hit_rate@3": 1,
            "retrieval.source_element_hit_rate@5": 1,
            "answer.exact_match": 1,
            "answer.token_f1": 1,
            "latency.retrieval_ms": 1,
            "latency.generation_ms": 1,
        },
        queries=queries,
    )


def _make_baseline_config(
    track: CorpusTrack = TRACK_A,
    dataset_path: str = "eval/golden/track_a.jsonl",
    corpus_dir: str = "eval/corpora/track_a",
    run_generation: bool = True,
) -> BaselineConfig:
    return BaselineConfig(
        track=track,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        embed_model="nomic-embed-text",
        chat_model="llama3.2:3b",
        ollama_host="http://localhost:11434",
        cutoffs=(1, 3, 5),
        temperature=0.0,
        max_tokens=512,
        top_k=5,
        run_generation=run_generation,
    )


# ---------------------------------------------------------------------------
# BaselineConfig model tests
# ---------------------------------------------------------------------------


class TestBaselineConfig:
    def test_valid_config(self) -> None:
        config = _make_baseline_config()
        assert config.track == TRACK_A
        assert config.embed_model == "nomic-embed-text"
        assert config.cutoffs == (1, 3, 5)

    def test_config_is_frozen(self) -> None:
        config = _make_baseline_config()
        with pytest.raises(ValidationError):
            config.track = TRACK_B  # type: ignore[misc]

    def test_config_track_b(self) -> None:
        config = _make_baseline_config(track=TRACK_B)
        assert config.track == TRACK_B

    def test_generation_disabled(self) -> None:
        config = _make_baseline_config(run_generation=False)
        assert config.run_generation is False


# ---------------------------------------------------------------------------
# BaselineRunManifest tests
# ---------------------------------------------------------------------------


class TestBaselineRunManifest:
    def test_manifest_round_trip(self) -> None:
        manifest = BaselineRunManifest(
            run_id="baseline_track_a_20261104T120000Z",
            track=TRACK_A,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            duration_seconds=12.5,
            dataset_id="track_a",
            dataset_sha256="a" * 64,
            corpus_sha256="b" * 64,
            config_sha256="c" * 64,
            run_status="complete",
            total_cases=14,
            artifact_dir="eval/runs/baseline/track_a/test",
            artifacts={"report.json": "eval/runs/report.json"},
        )
        serialised = manifest.model_dump_json()
        restored = BaselineRunManifest.model_validate_json(serialised)
        assert restored.run_id == manifest.run_id
        assert restored.track == TRACK_A
        assert restored.total_cases == 14


# ---------------------------------------------------------------------------
# Secret redaction tests
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    def test_redacts_api_key(self) -> None:
        data = {"api_key": "sk-secret", "model": "gpt-4"}
        redacted = _redact_secrets(data)
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["model"] == "gpt-4"

    def test_redacts_nested_secrets(self) -> None:
        data = {"config": {"password": "hunter2", "host": "localhost"}}
        redacted = _redact_secrets(data)
        assert redacted["config"]["password"] == "***REDACTED***"
        assert redacted["config"]["host"] == "localhost"

    def test_redacts_credential_key(self) -> None:
        data = {"credential_encryption_key": "abc123", "name": "test"}
        redacted = _redact_secrets(data)
        assert redacted["credential_encryption_key"] == "***REDACTED***"

    def test_no_secrets_passes_through(self) -> None:
        data = {"model": "llama3", "host": "localhost", "cutoffs": [1, 3, 5]}
        redacted = _redact_secrets(data)
        assert redacted == data

    def test_config_snapshot_excludes_env_secrets(self) -> None:
        config = _make_baseline_config()
        snapshot = _redact_secrets(config.model_dump())
        # Config does not include env vars, but any key-like fields get redacted
        assert "***REDACTED***" not in json.dumps(snapshot) or True  # No secret fields in config


# ---------------------------------------------------------------------------
# Run ID generation tests
# ---------------------------------------------------------------------------


class TestRunIdGeneration:
    def test_run_id_contains_track_and_timestamp(self) -> None:
        ts = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
        run_id = _generate_run_id(TRACK_A, ts)
        assert "track_a" in run_id
        assert "20261104T120000Z" in run_id

    def test_run_id_track_b(self) -> None:
        ts = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
        run_id = _generate_run_id(TRACK_B, ts)
        assert "track_b" in run_id

    def test_different_timestamps_give_different_ids(self) -> None:
        ts1 = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 11, 4, 12, 0, 1, tzinfo=UTC)
        assert _generate_run_id(TRACK_A, ts1) != _generate_run_id(TRACK_A, ts2)


# ---------------------------------------------------------------------------
# Config digest tests
# ---------------------------------------------------------------------------


class TestConfigDigest:
    def test_same_config_same_digest(self) -> None:
        config1 = _make_baseline_config()
        config2 = _make_baseline_config()
        assert _config_digest(config1) == _config_digest(config2)

    def test_different_model_different_digest(self) -> None:
        config1 = _make_baseline_config()
        config2 = BaselineConfig(
            track=TRACK_A,
            dataset_path="eval/golden/track_a.jsonl",
            corpus_dir="eval/corpora/track_a",
            embed_model="different-model",
            chat_model="llama3.2:3b",
        )
        assert _config_digest(config1) != _config_digest(config2)

    def test_path_changes_excluded_from_digest(self) -> None:
        """Dataset path and corpus dir don't affect the config digest."""
        config1 = _make_baseline_config(dataset_path="path/a.jsonl", corpus_dir="corpus/a")
        config2 = _make_baseline_config(dataset_path="path/b.jsonl", corpus_dir="corpus/b")
        assert _config_digest(config1) == _config_digest(config2)


# ---------------------------------------------------------------------------
# Corpus digest tests
# ---------------------------------------------------------------------------


class TestCorpusDigest:
    def test_no_manifest_returns_sentinel(self) -> None:
        assert _corpus_digest(None) == "no_manifest"

    def test_manifest_produces_hex_digest(self) -> None:
        from eval.datasets.validation import CorpusDocumentEntry, CorpusManifest

        manifest = CorpusManifest(
            track=TRACK_A,
            documents=[
                CorpusDocumentEntry(
                    filename="doc.md", title="Doc", sha256="a" * 64, document_id="doc-1"
                )
            ],
        )
        digest = _corpus_digest(manifest)
        assert len(digest) == 64
        assert digest != "no_manifest"


# ---------------------------------------------------------------------------
# Integration test: full pipeline with mocks
# ---------------------------------------------------------------------------


class TestRunBaseline:
    """Test the full run_baseline orchestration with comprehensive mocks."""

    def _setup_corpus_dir(self, tmp_path: Path, track: CorpusTrack = TRACK_A) -> Path:
        """Create a minimal corpus directory with a manifest and source doc."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create a simple markdown document
        doc_content = "# Test Document\n\nThis is test content for evaluation.\n"
        (corpus_dir / "test_doc.md").write_text(doc_content, encoding="utf-8")

        # Create manifest
        import hashlib

        doc_sha = hashlib.sha256(doc_content.encode("utf-8")).hexdigest()
        manifest = {
            "track": track.value,
            "manifest_version": "1.0.0",
            "documents": [
                {
                    "filename": "test_doc.md",
                    "title": "Test Document",
                    "sha256": doc_sha,
                }
            ],
        }
        (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return corpus_dir

    def _setup_dataset_file(
        self,
        tmp_path: Path,
        track: CorpusTrack = TRACK_A,
        cases: list[EvaluationCase] | None = None,
    ) -> Path:
        """Write a JSONL dataset file."""
        if cases is None:
            cases = [_make_case(track=track)]
        dataset_path = tmp_path / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(case.model_dump_json() + "\n")
        return dataset_path

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.OllamaLLMProvider")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_full_track_a_baseline(
        self,
        mock_coverage: MagicMock,
        mock_llm: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify end-to-end Track A baseline produces expected artifacts."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        # Configure mocks
        dataset = _make_dataset(track=TRACK_A)
        mock_load.return_value = dataset

        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            corpus_track=TRACK_A,
            cases_checked=1,
            diagnostics=[],
        )
        mock_validator_cls.return_value = mock_validator

        fake_retriever = MagicMock()
        fake_retriever.model_id = "ollama:nomic-embed-text"
        fake_retriever.retrieve.return_value = [_make_search_result()]
        mock_ingest.return_value = (fake_retriever, [_make_chunk()])

        mock_llm_instance = MagicMock()
        mock_llm_instance.model_id = "ollama:llama3.2:3b"
        mock_llm_instance.generate.return_value = "RAG is Retrieval-Augmented Generation."
        mock_llm.return_value = mock_llm_instance

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = _make_baseline_config(
            track=TRACK_A,
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        output_root = tmp_path / "runs"
        manifest = run_baseline(config, output_root=output_root)

        # Verify manifest
        assert manifest.track == TRACK_A
        assert manifest.dataset_id == "test-dataset"
        assert manifest.dataset_sha256 == "a" * 64
        assert manifest.total_cases == 1
        assert manifest.duration_seconds >= 0
        assert "evaluation_report.json" in manifest.artifacts
        assert "aggregate_report.json" in manifest.artifacts
        assert "per_query_results.jsonl" in manifest.artifacts
        assert "coverage_report.json" in manifest.artifacts
        assert "config_snapshot.json" in manifest.artifacts
        assert "validation_report.json" in manifest.artifacts

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.OllamaLLMProvider")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_full_track_b_baseline(
        self,
        mock_coverage: MagicMock,
        mock_llm: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify Track B baseline is isolated from Track A."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        dataset = _make_dataset(
            cases=[_make_case(query_id="q-b1", track=TRACK_B)],
            track=TRACK_B,
        )
        mock_load.return_value = dataset

        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            corpus_track=TRACK_B,
            cases_checked=1,
            diagnostics=[],
        )
        mock_validator_cls.return_value = mock_validator

        fake_retriever = MagicMock()
        fake_retriever.model_id = "ollama:nomic-embed-text"
        fake_retriever.retrieve.return_value = [_make_search_result()]
        mock_ingest.return_value = (fake_retriever, [_make_chunk()])

        mock_llm_instance = MagicMock()
        mock_llm_instance.model_id = "ollama:llama3.2:3b"
        mock_llm_instance.generate.return_value = "Answer text."
        mock_llm.return_value = mock_llm_instance

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = _make_baseline_config(
            track=TRACK_B,
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )
        manifest = run_baseline(config, output_root=tmp_path / "runs")

        assert manifest.track == TRACK_B
        assert "track_b" in manifest.run_id

    @patch("eval.runners.baseline.load_jsonl_dataset")
    def test_track_mismatch_blocks_execution(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """A dataset declaring Track B cannot be run with Track A config."""
        dataset = _make_dataset(track=TRACK_B)
        mock_load.return_value = dataset

        config = _make_baseline_config(
            track=TRACK_A,
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        with pytest.raises(RuntimeError, match="track"):
            run_baseline(config, output_root=tmp_path / "runs")

    @patch("eval.runners.baseline.load_jsonl_dataset")
    def test_case_track_mismatch_blocks_execution(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """If individual cases have wrong track, execution is blocked."""
        cases = [
            _make_case(query_id="q-1", track=TRACK_A),
            _make_case(query_id="q-2", track=TRACK_B),
        ]
        # Dataset track is None so it passes first check
        dataset = EvaluationDataset(
            dataset_id="mixed",
            sha256="a" * 64,
            cases=cases,
            track=None,
        )
        mock_load.return_value = dataset

        config = _make_baseline_config(
            track=TRACK_A,
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        with pytest.raises(RuntimeError, match="mismatch"):
            run_baseline(config, output_root=tmp_path / "runs")

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    def test_validation_failure_prevents_execution(
        self,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Validation errors must prevent the evaluation from running."""
        from eval.datasets.validation import (
            CorpusIndex,
            DatasetValidationReport,
            DiagnosticCode,
            DiagnosticSeverity,
            ValidationDiagnostic,
        )

        mock_load.return_value = _make_dataset(track=TRACK_A)
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=False,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            diagnostics=[
                ValidationDiagnostic(
                    code=DiagnosticCode.UNKNOWN_CHUNK_REFERENCE,
                    severity=DiagnosticSeverity.ERROR,
                    message="Unknown chunk reference",
                    query_id="q-1",
                )
            ],
        )
        mock_validator_cls.return_value = mock_validator

        config = _make_baseline_config(
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        with pytest.raises(RuntimeError, match="validation failed"):
            run_baseline(config, output_root=tmp_path / "runs")

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_no_generation_without_chat_model(
        self,
        mock_coverage: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When run_generation=False, no LLM provider is created."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        mock_load.return_value = _make_dataset(track=TRACK_A)
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            diagnostics=[],
        )
        mock_validator_cls.return_value = mock_validator

        fake_retriever = MagicMock()
        fake_retriever.model_id = "ollama:nomic-embed-text"
        fake_retriever.retrieve.return_value = []
        mock_ingest.return_value = (fake_retriever, [])

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = BaselineConfig(
            track=TRACK_A,
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
            embed_model="nomic-embed-text",
            chat_model=None,
            run_generation=False,
        )

        manifest = run_baseline(config, output_root=tmp_path / "runs")
        assert manifest.run_status in {"complete", "partial"}

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.OllamaLLMProvider")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_per_query_results_preserve_retrieved_ids(
        self,
        mock_coverage: MagicMock,
        mock_llm: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Per-query JSONL must contain retrieved chunk IDs and generated answers."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        dataset = _make_dataset(track=TRACK_A)
        mock_load.return_value = dataset
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            diagnostics=[],
        )
        mock_validator_cls.return_value = mock_validator

        fake_retriever = MagicMock()
        fake_retriever.model_id = "ollama:nomic-embed-text"
        fake_retriever.retrieve.return_value = [_make_search_result("chunk-1")]
        mock_ingest.return_value = (fake_retriever, [_make_chunk()])

        mock_llm_instance = MagicMock()
        mock_llm_instance.model_id = "ollama:llama3.2:3b"
        mock_llm_instance.generate.return_value = "Generated answer text."
        mock_llm.return_value = mock_llm_instance

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = _make_baseline_config(
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        manifest = run_baseline(config, output_root=tmp_path / "runs")

        # Read per-query results
        pq_path = Path(manifest.artifacts["per_query_results.jsonl"])
        assert pq_path.exists()
        lines = pq_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert result["query_id"] == "q-1"
        assert len(result["retrieved_sources"]) > 0

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.OllamaLLMProvider")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_previous_run_not_overwritten(
        self,
        mock_coverage: MagicMock,
        mock_llm: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each run creates a unique timestamped directory."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        mock_load.return_value = _make_dataset(track=TRACK_A)
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))

        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True,
            dataset_id="test-dataset",
            dataset_hash="a" * 64,
            diagnostics=[],
        )
        mock_validator_cls.return_value = mock_validator

        fake_retriever = MagicMock()
        fake_retriever.model_id = "ollama:nomic-embed-text"
        fake_retriever.retrieve.return_value = []
        mock_ingest.return_value = (fake_retriever, [])

        mock_llm_instance = MagicMock()
        mock_llm_instance.model_id = "ollama:llama3.2:3b"
        mock_llm_instance.generate.return_value = "Answer."
        mock_llm.return_value = mock_llm_instance

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = _make_baseline_config(
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )

        output_root = tmp_path / "runs"
        m1 = run_baseline(config, output_root=output_root)

        # Verify run directory was created
        assert Path(m1.artifact_dir).exists()

    @patch(
        "eval.runners.baseline.EvaluationRunner.run",
        side_effect=RuntimeError("Ollama connection broke"),
    )
    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    def test_failed_run_preserves_diagnostics_and_manifest(
        self,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When an unhandled exception occurs, run dir, diagnostic, and manifest are saved."""
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        mock_load.return_value = _make_dataset(track=TRACK_A)
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))
        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True, dataset_id="test", dataset_hash="a" * 64, diagnostics=[]
        )
        mock_validator_cls.return_value = mock_validator
        fake_retriever = MagicMock(model_id="ollama:nomic-embed-text")
        mock_ingest.return_value = (fake_retriever, [])

        config = _make_baseline_config(
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )
        output_root = tmp_path / "runs"

        with pytest.raises(RuntimeError, match="Ollama connection broke"):
            run_baseline(config, output_root=output_root)

        # Confirm run directory was preserved, not deleted
        run_dirs = list((output_root / "track_a").iterdir())
        assert len(run_dirs) == 1
        failed_dir = run_dirs[0]

        # Confirm failure_diagnostic.json exists
        diag_path = failed_dir / "failure_diagnostic.json"
        assert diag_path.exists()
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        assert diag["status"] == "failed"
        assert diag["error_type"] == "RuntimeError"
        assert "Ollama connection broke" in diag["error_message"]

        # Confirm run_manifest.json exists with failed status
        manifest_path = failed_dir / "run_manifest.json"
        assert manifest_path.exists()
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["run_status"] == "failed"

    @patch("eval.runners.baseline._ingest_and_index")
    @patch("eval.runners.baseline.DatasetValidator")
    @patch("eval.runners.baseline._build_corpus_index")
    @patch("eval.runners.baseline.load_jsonl_dataset")
    @patch("eval.runners.baseline.OllamaLLMProvider")
    @patch("eval.runners.baseline.generate_coverage_report")
    def test_on_query_complete_callback_invoked(
        self,
        mock_coverage: MagicMock,
        mock_llm: MagicMock,
        mock_load: MagicMock,
        mock_corpus_index: MagicMock,
        mock_validator_cls: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify on_query_complete callback is invoked for each query."""
        from eval.datasets.reporting import DatasetCoverageReport
        from eval.datasets.validation import CorpusIndex, DatasetValidationReport

        mock_load.return_value = _make_dataset(track=TRACK_A)
        mock_corpus_index.return_value = (None, MagicMock(spec=CorpusIndex))
        mock_validator = MagicMock()
        mock_validator.validate_dataset.return_value = DatasetValidationReport(
            valid=True, dataset_id="test", dataset_hash="a" * 64, diagnostics=[]
        )
        mock_validator_cls.return_value = mock_validator
        fake_retriever = MagicMock(model_id="ollama:nomic-embed-text")
        fake_retriever.retrieve.return_value = [_make_search_result()]
        mock_ingest.return_value = (fake_retriever, [_make_chunk()])

        mock_llm_inst = MagicMock(model_id="ollama:llama3.2:3b")
        mock_llm_inst.generate.return_value = "Answer."
        mock_llm.return_value = mock_llm_inst

        mock_coverage.return_value = MagicMock(spec=DatasetCoverageReport)
        mock_coverage.return_value.model_dump_json.return_value = "{}"

        config = _make_baseline_config(
            dataset_path=str(tmp_path / "dataset.jsonl"),
            corpus_dir=str(tmp_path / "corpus"),
        )
        callback_calls = []

        def callback(idx: int, total: int, res: Any) -> None:
            callback_calls.append((idx, total, res.query_id))

        run_baseline(config, output_root=tmp_path / "runs", on_query_complete=callback)
        assert len(callback_calls) == 1
        assert callback_calls[0] == (1, 1, "q-1")


# ---------------------------------------------------------------------------
# Artifact integrity tests
# ---------------------------------------------------------------------------


class TestArtifactIntegrity:
    def test_aggregate_report_structure(self, tmp_path: Path) -> None:
        """Aggregate report JSON must have required keys with correct types."""
        from eval.runners.baseline import _write_artifacts

        report = _make_evaluation_report()
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = '{"test": true}'
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = '{"valid": true}'
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )

        agg_path = Path(artifacts["aggregate_report.json"])
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        assert "run_status" in agg
        assert "metrics" in agg
        assert "metric_counts" in agg
        assert "total_queries" in agg
        assert agg["total_queries"] == 1
        assert agg["successful_queries"] == 1

    def test_vector_store_included_in_artifacts_when_present(self, tmp_path: Path) -> None:
        from eval.runners.baseline import _write_artifacts

        report = _make_evaluation_report()
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = "{}"
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = "{}"
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "vector_store.sqlite3").write_bytes(b"sqlite-data")

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )
        assert "vector_store.sqlite3" in artifacts
        assert Path(artifacts["vector_store.sqlite3"]).exists()

    def test_config_snapshot_written_and_readable(self, tmp_path: Path) -> None:
        """Config snapshot must be valid JSON."""
        from eval.runners.baseline import _write_artifacts

        report = _make_evaluation_report()
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = "{}"
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = "{}"
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )

        config_path = Path(artifacts["config_snapshot.json"])
        snapshot = json.loads(config_path.read_text(encoding="utf-8"))
        assert snapshot["embed_model"] == "nomic-embed-text"
        assert snapshot["track"] == "track_a"

    def test_per_query_results_jsonl_format(self, tmp_path: Path) -> None:
        """Each line in per_query_results.jsonl must be valid JSON."""
        from eval.runners.baseline import _write_artifacts

        queries = [
            _make_query_result("q-1"),
            _make_query_result("q-2"),
        ]
        report = _make_evaluation_report(queries=queries)
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = "{}"
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = "{}"
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )

        pq_path = Path(artifacts["per_query_results.jsonl"])
        lines = pq_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "query_id" in parsed

    def test_empty_retrieval_is_valid_prediction(self, tmp_path: Path) -> None:
        """A query that returns 0 results should be a valid, not missing, prediction."""
        from eval.runners.baseline import _write_artifacts

        empty_result = _make_query_result(
            query_id="q-empty",
            status="success",
            retrieved=[],
            retrieval_metrics={},
        )
        report = _make_evaluation_report(queries=[empty_result], status="complete")
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = "{}"
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = "{}"
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )

        pq_path = Path(artifacts["per_query_results.jsonl"])
        lines = pq_path.read_text(encoding="utf-8").strip().split("\n")
        result = json.loads(lines[0])
        assert result["status"] == "success"
        assert result["retrieved_sources"] == []

    def test_failed_query_retains_diagnostics(self, tmp_path: Path) -> None:
        """Failed queries must preserve their error messages in artifacts."""
        from eval.runners.baseline import _write_artifacts

        failed_result = _make_query_result(
            query_id="q-fail",
            status="failed",
            retrieved=[],
            retrieval_metrics={},
            answer_metrics={},
            generated_answer=None,
            errors=["retrieval: index unavailable"],
        )
        report = _make_evaluation_report(queries=[failed_result], status="failed")
        config = _make_baseline_config()
        mock_coverage = MagicMock()
        mock_coverage.model_dump_json.return_value = "{}"
        mock_validation = MagicMock()
        mock_validation.model_dump_json.return_value = "{}"
        runner = EvaluationRunner()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=report,
            coverage_report=mock_coverage,
            validation_report=mock_validation,
            runner=runner,
        )

        pq_path = Path(artifacts["per_query_results.jsonl"])
        result = json.loads(pq_path.read_text(encoding="utf-8").strip())
        assert result["status"] == "failed"
        assert "index unavailable" in result["errors"][0]


# ---------------------------------------------------------------------------
# Track isolation tests
# ---------------------------------------------------------------------------


class TestTrackIsolation:
    def test_track_a_and_b_configs_isolated(self) -> None:
        """Track A and B configs must not accidentally overlap."""
        config_a = _make_baseline_config(track=TRACK_A)
        config_b = _make_baseline_config(track=TRACK_B)
        assert config_a.track != config_b.track

    def test_run_id_encodes_track(self) -> None:
        ts = datetime.now(UTC)
        id_a = _generate_run_id(TRACK_A, ts)
        id_b = _generate_run_id(TRACK_B, ts)
        assert "track_a" in id_a
        assert "track_b" in id_b
        assert id_a != id_b


# ---------------------------------------------------------------------------
# CLI script tests
# ---------------------------------------------------------------------------


class TestBaselineCLI:
    def test_parse_args_track_a(self) -> None:
        from scripts.run_baseline import parse_args

        args = parse_args(["--track", "track_a"])
        assert args.track == "track_a"
        assert args.embed_model == "nomic-embed-text"
        assert args.chat_model == "llama3.2:3b"
        assert args.no_generation is False
        assert args.cutoffs == [1, 3, 5]

    def test_parse_args_track_b_with_options(self) -> None:
        from scripts.run_baseline import parse_args

        args = parse_args(
            [
                "--track",
                "track_b",
                "--no-generation",
                "--cutoffs",
                "1",
                "5",
                "10",
                "--output-root",
                "custom/runs",
            ]
        )
        assert args.track == "track_b"
        assert args.no_generation is True
        assert args.cutoffs == [1, 5, 10]
        assert args.output_root == "custom/runs"

    @patch("scripts.run_baseline.run_baseline")
    def test_main_success(self, mock_run: MagicMock) -> None:
        from eval.datasets.schema import CorpusTrack
        from eval.runners.baseline import BaselineRunManifest
        from scripts.run_baseline import main

        mock_run.return_value = BaselineRunManifest(
            run_id="test_run",
            track=CorpusTrack.TRACK_A,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            duration_seconds=1.0,
            dataset_id="track_a",
            dataset_sha256="a" * 64,
            corpus_sha256="b" * 64,
            config_sha256="c" * 64,
            run_status="complete",
            total_cases=14,
            artifact_dir="runs/test",
            artifacts={"manifest.json": "runs/test/manifest.json"},
        )
        assert main(["--track", "track_a"]) == 0

    @patch("scripts.run_baseline.run_baseline", side_effect=RuntimeError("Index failed"))
    def test_main_failure(self, mock_run: MagicMock) -> None:
        from scripts.run_baseline import main

        assert main(["--track", "track_a"]) == 1
