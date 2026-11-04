"""Dual-track baseline execution orchestrator for ground-truth evaluation.

Loads a validated golden dataset and its corpus, indexes documents into a fresh
vector store, executes the evaluation runner, generates a coverage report, and
serialises all artifacts to a timestamped run directory.

This module never downloads models, mutates golden datasets, pools tracks, or
optimises any parameter. It measures the current system exactly as configured.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.embed.ollama_embedder import OllamaEmbedder
from core.generate.pipeline import GroundedAnswerGenerator
from core.ingest import create_default_ingestion_pipeline
from core.llm.base import GenerationSettings
from core.llm.ollama_provider import OllamaLLMProvider
from core.normalize.schema import Chunk
from core.retrieval.dense import DenseRetriever
from core.storage.sqlite_store import SQLiteVectorStore
from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.reporting import generate_coverage_report
from eval.datasets.schema import CorpusTrack, EvaluationDataset
from eval.datasets.validation import (
    CorpusIndex,
    CorpusManifest,
    DatasetValidator,
)
from eval.runners.evaluate import EvaluationReport, EvaluationRunner, QueryEvaluationResult
from eval.runners.run_config import EvaluationRunConfig

# Patterns that indicate secrets to redact in config snapshots.
_SECRET_PATTERNS = re.compile(r"(key|secret|token|password|credential|api_key)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Configuration & manifest models
# ---------------------------------------------------------------------------


class BaselineConfig(BaseModel):
    """Frozen, serialisable snapshot of every parameter that affects a baseline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track: CorpusTrack
    dataset_path: str
    corpus_dir: str
    embed_model: str
    chat_model: str | None = None
    ollama_host: str = "http://localhost:11434"
    cutoffs: tuple[int, ...] = (1, 3, 5)
    temperature: float = 0.0
    max_tokens: int = 512
    top_k: int = 5
    run_generation: bool = True
    request_timeout: float = 180.0


class BaselineRunManifest(BaseModel):
    """Machine-readable record linking a baseline run to its inputs and outputs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    track: CorpusTrack
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = Field(ge=0.0)
    dataset_id: str
    dataset_sha256: str
    corpus_sha256: str
    config_sha256: str
    run_status: str
    total_cases: int
    artifact_dir: str
    artifacts: dict[str, str]


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------


def run_baseline(
    config: BaselineConfig,
    output_root: Path | str = Path("eval/runs/baseline"),
    on_query_complete: Callable[[int, int, QueryEvaluationResult], None] | None = None,
) -> BaselineRunManifest:
    """Execute a single-track baseline evaluation and persist all artifacts.

    Raises ``RuntimeError`` on track mismatch or validation failure. On unhandled
    exceptions or interrupts, preserves a diagnostic failure record in the run directory.
    """
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()

    output_root = Path(output_root)
    dataset_path = Path(config.dataset_path)
    corpus_dir = Path(config.corpus_dir)

    # 1. Load and validate dataset
    dataset = _load_and_validate_dataset(dataset_path, config.track)

    # 2. Build corpus index for validation
    corpus_manifest, corpus_index = _build_corpus_index(corpus_dir)

    # 3. Validate dataset against corpus
    validator = DatasetValidator()
    validation_report = validator.validate_dataset(dataset, corpus_index)
    validation_errors = [d for d in validation_report.diagnostics if d.severity == "error"]
    if validation_errors:
        error_summary = "; ".join(f"{d.code}: {d.message}" for d in validation_errors[:5])
        raise RuntimeError(
            f"Dataset validation failed with {len(validation_errors)} error(s): {error_summary}"
        )

    # 4. Ingest corpus and index documents
    run_id = _generate_run_id(config.track, started_at)
    run_dir = output_root / config.track.value / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "vector_store.sqlite3"

    try:
        retriever, _ = _ingest_and_index(
            corpus_dir=corpus_dir,
            db_path=db_path,
            embed_model=config.embed_model,
            ollama_host=config.ollama_host,
            timeout=config.request_timeout,
        )

        # 5. Optionally create answer generator
        answerer = None
        if config.run_generation and config.chat_model:
            provider = OllamaLLMProvider(
                config.chat_model, host=config.ollama_host, timeout=config.request_timeout
            )
            settings = GenerationSettings(
                temperature=config.temperature, max_tokens=config.max_tokens
            )
            answerer = GroundedAnswerGenerator(
                retriever, provider, top_k=config.top_k, settings=settings
            )

        # 6. Run evaluation
        eval_config = EvaluationRunConfig(cutoffs=config.cutoffs)
        runner = EvaluationRunner(eval_config)
        evaluation_report = runner.run(
            dataset, retriever, answerer, on_query_complete=on_query_complete
        )

        # 7. Generate coverage report
        coverage_report = generate_coverage_report(
            dataset, validation_report, expected_track=config.track
        )

        # 8. Serialise artifacts atomically
        artifacts = _write_artifacts(
            run_dir=run_dir,
            config=config,
            evaluation_report=evaluation_report,
            coverage_report=coverage_report,
            validation_report=validation_report,
            runner=runner,
        )

        completed_at = datetime.now(UTC)
        duration_seconds = max(0.0, time.perf_counter() - wall_start)

        # 9. Compute corpus digest from manifest
        corpus_sha256 = _corpus_digest(corpus_manifest)

        # 10. Build and write manifest
        manifest_path = run_dir / "run_manifest.json"
        artifacts["run_manifest.json"] = str(manifest_path)

        manifest = BaselineRunManifest(
            run_id=run_id,
            track=config.track,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(duration_seconds, 3),
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.sha256,
            corpus_sha256=corpus_sha256,
            config_sha256=_config_digest(config),
            run_status=evaluation_report.status,
            total_cases=len(dataset.cases),
            artifact_dir=str(run_dir),
            artifacts=artifacts,
        )

        _atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
        return manifest

    except BaseException as exc:
        failed_at = datetime.now(UTC)
        duration_seconds = max(0.0, time.perf_counter() - wall_start)
        is_interrupt = isinstance(exc, KeyboardInterrupt)
        status = "interrupted" if is_interrupt else "failed"

        # Preserve failure diagnostic evidence
        tb_str = traceback.format_exc() if not is_interrupt else "Execution interrupted by user"
        failure_diag = {
            "status": status,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb_str,
            "failed_at": failed_at.isoformat(),
            "duration_seconds": round(duration_seconds, 3),
        }
        diag_path = run_dir / "failure_diagnostic.json"
        _atomic_write_text(diag_path, json.dumps(failure_diag, indent=2) + "\n")

        # Write partial/failed manifest so run remains identifiable and diagnosable
        failed_artifacts: dict[str, str] = {
            "failure_diagnostic.json": str(diag_path),
        }
        if db_path.exists():
            failed_artifacts["vector_store.sqlite3"] = str(db_path)

        partial_manifest = BaselineRunManifest(
            run_id=run_id,
            track=config.track,
            started_at=started_at,
            completed_at=failed_at,
            duration_seconds=round(duration_seconds, 3),
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.sha256,
            corpus_sha256=_corpus_digest(corpus_manifest),
            config_sha256=_config_digest(config),
            run_status=status,
            total_cases=len(dataset.cases),
            artifact_dir=str(run_dir),
            artifacts=failed_artifacts,
        )
        _atomic_write_text(
            run_dir / "run_manifest.json", partial_manifest.model_dump_json(indent=2) + "\n"
        )
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_and_validate_dataset(
    dataset_path: Path, expected_track: CorpusTrack
) -> EvaluationDataset:
    """Load a JSONL dataset and verify its track matches the expected track."""
    dataset = load_jsonl_dataset(dataset_path)

    # Verify track consistency
    if dataset.track is not None and dataset.track != expected_track:
        raise RuntimeError(
            f"Dataset declares track '{dataset.track}' but baseline "
            f"is configured for track '{expected_track}'."
        )

    # Verify all cases belong to the expected track
    mismatched = [
        c.query_id for c in dataset.cases if c.track is not None and c.track != expected_track
    ]
    if mismatched:
        raise RuntimeError(
            f"Dataset contains {len(mismatched)} case(s) with track mismatch: {mismatched[:5]}"
        )

    return dataset


def _build_corpus_index(corpus_dir: Path) -> tuple[CorpusManifest | None, CorpusIndex]:
    """Load corpus manifest and build a corpus index for validation."""
    manifest_path = corpus_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = CorpusManifest.model_validate(manifest_data)
        except Exception:
            pass  # Validation will catch manifest issues

    # Ingest all documents to build the index
    pipeline = create_default_ingestion_pipeline()
    documents = []
    all_chunks: list[Chunk] = []

    # Find source documents in corpus directory
    source_files = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix in {".md", ".txt"} and p.name != "manifest.json"
    )

    for source_file in source_files:
        result = pipeline.ingest(source_file)
        if result.document is not None:
            documents.append(result.document)
        all_chunks.extend(result.chunks)

    corpus_track = manifest.track if manifest is not None else None
    return manifest, CorpusIndex(
        documents=documents,
        chunks=all_chunks,
        track=corpus_track,
        manifest=manifest,
    )


def _ingest_and_index(
    *,
    corpus_dir: Path,
    db_path: Path,
    embed_model: str,
    ollama_host: str,
    timeout: float = 120.0,
) -> tuple[DenseRetriever, list[Chunk]]:
    """Ingest corpus documents and index them into a fresh SQLite vector store."""
    pipeline = create_default_ingestion_pipeline()
    embedder = OllamaEmbedder(embed_model, host=ollama_host, timeout=timeout)
    vector_store = SQLiteVectorStore(db_path)
    retriever = DenseRetriever(embedder, vector_store)

    all_chunks: list[Chunk] = []
    source_files = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix in {".md", ".txt"} and p.name != "manifest.json"
    )

    for source_file in source_files:
        result = pipeline.ingest(source_file)
        if result.document is not None and result.chunks:
            retriever.index_document(result.document.metadata.document_id, result.chunks)
            all_chunks.extend(result.chunks)

    return retriever, all_chunks


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text content by writing to a temporary file then renaming."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_jsonl(path: Path, lines: Iterable[str]) -> None:
    """Atomically write JSONL lines by writing to a temporary file then renaming."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    tmp_path.replace(path)


def _write_artifacts(
    *,
    run_dir: Path,
    config: BaselineConfig,
    evaluation_report: EvaluationReport,
    coverage_report: Any,
    validation_report: Any,
    runner: EvaluationRunner,
) -> dict[str, str]:
    """Serialise all baseline run artifacts atomically and return a name→path mapping."""
    artifacts: dict[str, str] = {}

    db_path = run_dir / "vector_store.sqlite3"
    if db_path.exists():
        artifacts["vector_store.sqlite3"] = str(db_path)

    # Config snapshot (secrets redacted)
    config_snapshot = _redact_secrets(config.model_dump())
    config_path = run_dir / "config_snapshot.json"
    _atomic_write_text(config_path, json.dumps(config_snapshot, indent=2, default=str) + "\n")
    artifacts["config_snapshot.json"] = str(config_path)

    # Evaluation report (full)
    report_path = run_dir / "evaluation_report.json"
    _atomic_write_text(report_path, evaluation_report.model_dump_json(indent=2) + "\n")
    artifacts["evaluation_report.json"] = str(report_path)

    # Aggregate metrics (compact summary)
    aggregate = {
        "run_status": evaluation_report.status,
        "dataset_id": evaluation_report.dataset_id,
        "dataset_sha256": evaluation_report.dataset_sha256,
        "retrieval_model_id": evaluation_report.retrieval_model_id,
        "generation_model_id": evaluation_report.generation_model_id,
        "metrics": evaluation_report.metrics,
        "metric_counts": evaluation_report.metric_counts,
        "total_queries": len(evaluation_report.queries),
        "successful_queries": sum(1 for q in evaluation_report.queries if q.status == "success"),
        "failed_queries": sum(1 for q in evaluation_report.queries if q.status == "failed"),
        "partial_queries": sum(1 for q in evaluation_report.queries if q.status == "partial"),
    }
    aggregate_path = run_dir / "aggregate_report.json"
    _atomic_write_text(aggregate_path, json.dumps(aggregate, indent=2, default=str) + "\n")
    artifacts["aggregate_report.json"] = str(aggregate_path)

    # Per-query results (JSONL, one per line)
    per_query_path = run_dir / "per_query_results.jsonl"
    _atomic_write_jsonl(per_query_path, (q.model_dump_json() for q in evaluation_report.queries))
    artifacts["per_query_results.jsonl"] = str(per_query_path)

    # Coverage report
    coverage_path = run_dir / "coverage_report.json"
    _atomic_write_text(coverage_path, coverage_report.model_dump_json(indent=2) + "\n")
    artifacts["coverage_report.json"] = str(coverage_path)

    # Validation report
    validation_path = run_dir / "validation_report.json"
    _atomic_write_text(validation_path, validation_report.model_dump_json(indent=2) + "\n")
    artifacts["validation_report.json"] = str(validation_path)

    return artifacts


def _redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values whose keys match secret patterns."""
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _SECRET_PATTERNS.search(key):
            redacted[key] = "***REDACTED***"
        elif isinstance(value, dict):
            redacted[key] = _redact_secrets(value)
        else:
            redacted[key] = value
    return redacted


def _generate_run_id(track: CorpusTrack, timestamp: datetime) -> str:
    """Produce a human-readable, collision-resistant run identifier."""
    ts = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"baseline_{track.value}_{ts}"


def _config_digest(config: BaselineConfig) -> str:
    """Produce a SHA-256 digest of the serialised configuration."""
    raw = config.model_dump_json(exclude={"dataset_path", "corpus_dir"})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _corpus_digest(manifest: CorpusManifest | None) -> str:
    """Produce a composite digest from the corpus manifest entries."""
    if manifest is None:
        return "no_manifest"
    parts = []
    for doc in sorted(manifest.documents, key=lambda d: d.document_id or d.filename):
        parts.append(f"{doc.document_id or doc.filename}:{doc.sha256}")
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
