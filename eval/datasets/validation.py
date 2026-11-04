"""Deterministic evaluation dataset and corpus validation layer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk, NormalizedDocument
from eval.datasets.schema import (
    AnnotationStatus,
    Answerability,
    CorpusTrack,
    EvaluationCase,
    EvaluationDataset,
)


class DiagnosticSeverity(StrEnum):
    """Severity classification for validation diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DiagnosticCode(StrEnum):
    """Stable, documented codes for all validation findings."""

    # Dataset structural & JSONL codes
    DATASET_INVALID_JSON = "DATASET_INVALID_JSON"
    DATASET_SCHEMA_ERROR = "DATASET_SCHEMA_ERROR"
    DATASET_DUPLICATE_QUERY_ID = "DATASET_DUPLICATE_QUERY_ID"
    DATASET_EMPTY = "DATASET_EMPTY"
    DATASET_HASH_MISMATCH = "DATASET_HASH_MISMATCH"
    DATASET_READ_ERROR = "DATASET_READ_ERROR"

    # Corpus manifest & document codes
    CORPUS_MANIFEST_MISSING = "CORPUS_MANIFEST_MISSING"
    CORPUS_MANIFEST_INVALID = "CORPUS_MANIFEST_INVALID"
    CORPUS_DOCUMENT_MISSING = "CORPUS_DOCUMENT_MISSING"
    CORPUS_DOCUMENT_HASH_MISMATCH = "CORPUS_DOCUMENT_HASH_MISMATCH"
    CORPUS_DUPLICATE_DOCUMENT_ID = "CORPUS_DUPLICATE_DOCUMENT_ID"
    CORPUS_UNEXPECTED_FILE = "CORPUS_UNEXPECTED_FILE"
    CORPUS_INGESTION_ERROR = "CORPUS_INGESTION_ERROR"

    # Track isolation codes
    TRACK_MISMATCH = "TRACK_MISMATCH"
    CROSS_TRACK_REFERENCE = "CROSS_TRACK_REFERENCE"
    MISSING_TRACK_METADATA = "MISSING_TRACK_METADATA"

    # Reference integrity codes
    UNKNOWN_ELEMENT_REFERENCE = "UNKNOWN_ELEMENT_REFERENCE"
    UNKNOWN_CHUNK_REFERENCE = "UNKNOWN_CHUNK_REFERENCE"
    CHUNK_ELEMENT_MISMATCH = "CHUNK_ELEMENT_MISMATCH"
    DUPLICATE_RELEVANCE_ID = "DUPLICATE_RELEVANCE_ID"

    # Annotation lifecycle & provenance codes
    VERIFIED_WITHOUT_PROVENANCE = "VERIFIED_WITHOUT_PROVENANCE"
    PROVENANCE_DOCUMENT_MISSING = "PROVENANCE_DOCUMENT_MISSING"
    PROVENANCE_EVIDENCE_MISSING = "PROVENANCE_EVIDENCE_MISSING"
    ANNOTATION_STATE_INCONSISTENT = "ANNOTATION_STATE_INCONSISTENT"
    MISSING_REFERENCE_ANSWER = "MISSING_REFERENCE_ANSWER"
    MISSING_DIFFICULTY = "MISSING_DIFFICULTY"
    UNVERIFIED_CANDIDATE = "UNVERIFIED_CANDIDATE"
    PARTIAL_ANNOTATION = "PARTIAL_ANNOTATION"


class ValidationDiagnostic(BaseModel):
    """One deterministic validation observation with contextual coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    line_number: int | None = None
    query_id: str | None = None
    document_id: str | None = None
    element_id: str | None = None
    chunk_id: str | None = None
    field: str | None = None


class DatasetValidationReport(BaseModel):
    """Complete results of dataset structural and corpus-aware validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    dataset_id: str
    dataset_hash: str
    corpus_track: CorpusTrack | None = None
    corpus_hash: str | None = None
    cases_checked: int = 0
    documents_checked: int = 0
    elements_checked: int = 0
    chunks_checked: int = 0
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationDiagnostic]:
        """All blocking error diagnostics that make the dataset invalid."""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationDiagnostic]:
        """Non-blocking warning diagnostics."""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING]

    @property
    def info(self) -> list[ValidationDiagnostic]:
        """Informational diagnostics."""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.INFO]


class CorpusDocumentEntry(BaseModel):
    """Metadata describing one canonical source document in a corpus manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    document_id: str | None = None
    source_url: str | None = None
    license_or_reuse_terms: str | None = None
    retrieved_at: datetime | None = None


class CorpusManifest(BaseModel):
    """Manifest describing a version-controlled golden corpus track."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track: CorpusTrack
    manifest_version: str = Field(default="1.0.0", min_length=1)
    documents: list[CorpusDocumentEntry] = Field(..., min_length=1)
    description: str | None = None


class CorpusIndex:
    """Pre-indexed, immutable reference set of documents, elements, and chunks."""

    def __init__(
        self,
        documents: Sequence[NormalizedDocument],
        chunks: Sequence[Chunk],
        track: CorpusTrack | None = None,
        manifest: CorpusManifest | None = None,
        manifest_hash: str | None = None,
        filename_by_doc_id: dict[str, str] | None = None,
    ) -> None:
        self.documents = tuple(documents)
        self.chunks = tuple(chunks)
        self.track = track
        self.manifest = manifest
        self.manifest_hash = manifest_hash

        # Index sets and lookup mappings
        self.document_ids = frozenset(doc.metadata.document_id for doc in self.documents)
        self.element_ids = frozenset(
            elem.element_id for doc in self.documents for elem in doc.elements
        )
        self.chunk_ids = frozenset(chunk.chunk_id for chunk in self.chunks)

        self.element_to_doc: dict[str, str] = {
            elem.element_id: doc.metadata.document_id
            for doc in self.documents
            for elem in doc.elements
        }
        self.chunk_to_doc: dict[str, str] = {
            chunk.chunk_id: chunk.document_id for chunk in self.chunks
        }
        self.chunk_to_elements: dict[str, frozenset[str]] = {
            chunk.chunk_id: frozenset(chunk.source_element_ids) for chunk in self.chunks
        }
        self.filename_by_doc_id: dict[str, str] = filename_by_doc_id or {
            doc.metadata.document_id: doc.metadata.filename for doc in self.documents
        }
        self.filenames = frozenset(self.filename_by_doc_id.values())


class DatasetValidator:
    """Deterministic, side-effect-free evaluation dataset validator."""

    def __init__(self, pipeline=None) -> None:
        self._pipeline = pipeline or create_default_ingestion_pipeline()

    def validate_dataset(
        self,
        dataset: EvaluationDataset,
        corpus: CorpusIndex | Path | str | None = None,
        expected_track: CorpusTrack | None = None,
        expected_hash: str | None = None,
    ) -> DatasetValidationReport:
        """Validate an in-memory EvaluationDataset against structural rules and corpus."""
        diagnostics: list[ValidationDiagnostic] = []

        # 1. Dataset hash verification if requested
        if expected_hash is not None and dataset.sha256 != expected_hash:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.DATASET_HASH_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Dataset SHA-256 digest '{dataset.sha256}' does not match "
                        f"expected '{expected_hash}'."
                    ),
                    field="sha256",
                )
            )

        # 2. Track consistency
        effective_track = expected_track or dataset.track
        if (
            expected_track is not None
            and dataset.track is not None
            and dataset.track != expected_track
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.TRACK_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Dataset declares track '{dataset.track}' but expected "
                        f"track is '{expected_track}'."
                    ),
                    field="track",
                )
            )
        elif effective_track is not None and dataset.track is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.MISSING_TRACK_METADATA,
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "Dataset does not declare an explicit track; validating against "
                        "specified track context."
                    ),
                    field="track",
                )
            )

        # 3. Structural checks on cases
        seen_query_ids: set[str] = set()
        for idx, case in enumerate(dataset.cases, start=1):
            if case.query_id in seen_query_ids:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.DATASET_DUPLICATE_QUERY_ID,
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Duplicate query_id detected: '{case.query_id}'.",
                        query_id=case.query_id,
                        line_number=idx,
                        field="query_id",
                    )
                )
            seen_query_ids.add(case.query_id)

            # Case-level track isolation
            if (
                case.track is not None
                and effective_track is not None
                and case.track != effective_track
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CROSS_TRACK_REFERENCE,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Case declares track '{case.track}' which contradicts "
                            f"dataset track '{effective_track}'."
                        ),
                        query_id=case.query_id,
                        line_number=idx,
                        field="track",
                    )
                )

            # Annotation lifecycle checks
            self._validate_case_annotations(case, idx, diagnostics)

        # 4. Corpus-aware validation if corpus is provided
        corpus_index: CorpusIndex | None = None
        if corpus is not None:
            corpus_index, corpus_diags = self._resolve_corpus_index(corpus, effective_track)
            diagnostics.extend(corpus_diags)

            if corpus_index is not None:
                self._validate_corpus_references(
                    dataset.cases, corpus_index, effective_track, diagnostics
                )

        is_valid = not any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)
        return DatasetValidationReport(
            valid=is_valid,
            dataset_id=dataset.dataset_id,
            dataset_hash=dataset.sha256,
            corpus_track=effective_track,
            corpus_hash=corpus_index.manifest_hash if corpus_index else None,
            cases_checked=len(dataset.cases),
            documents_checked=len(corpus_index.documents) if corpus_index else 0,
            elements_checked=len(corpus_index.element_ids) if corpus_index else 0,
            chunks_checked=len(corpus_index.chunk_ids) if corpus_index else 0,
            diagnostics=diagnostics,
        )

    def validate_file(
        self,
        file_path: Path | str,
        corpus: CorpusIndex | Path | str | None = None,
        expected_track: CorpusTrack | None = None,
        expected_hash: str | None = None,
    ) -> DatasetValidationReport:
        """Validate a JSONL dataset file, gathering line-by-line syntax and schema diagnostics."""
        path = Path(file_path)
        diagnostics: list[ValidationDiagnostic] = []

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.DATASET_READ_ERROR,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Failed to read dataset file '{path}': {exc}",
                )
            )
            return DatasetValidationReport(
                valid=False,
                dataset_id=path.stem,
                dataset_hash="",
                diagnostics=diagnostics,
            )

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.DATASET_INVALID_JSON,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Dataset file '{path}' is not valid UTF-8: {exc}",
                )
            )
            return DatasetValidationReport(
                valid=False,
                dataset_id=path.stem,
                dataset_hash=hashlib.sha256(raw_bytes).hexdigest(),
                diagnostics=diagnostics,
            )

        computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if expected_hash is not None and computed_sha256 != expected_hash:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.DATASET_HASH_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"File content SHA-256 '{computed_sha256}' does not match "
                        f"expected '{expected_hash}'."
                    ),
                    field="sha256",
                )
            )

        cases: list[EvaluationCase] = []
        non_empty_lines = 0
        seen_query_ids: dict[str, int] = {}
        inferred_track: CorpusTrack | None = None

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            non_empty_lines += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.DATASET_INVALID_JSON,
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Invalid JSON at line {line_number}: {exc}",
                        line_number=line_number,
                    )
                )
                continue

            if not isinstance(data, dict):
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.DATASET_SCHEMA_ERROR,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Expected JSON object at line {line_number}, "
                            f"got {type(data).__name__}."
                        ),
                        line_number=line_number,
                    )
                )
                continue

            qid = data.get("query_id")
            if isinstance(qid, str) and qid.strip():
                if qid in seen_query_ids:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code=DiagnosticCode.DATASET_DUPLICATE_QUERY_ID,
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Duplicate query_id detected: '{qid}'.",
                            query_id=qid,
                            line_number=line_number,
                            field="query_id",
                        )
                    )
                else:
                    seen_query_ids[qid] = line_number

            try:
                case = EvaluationCase.model_validate(data)
                cases.append(case)
                if inferred_track is None and case.track is not None:
                    inferred_track = case.track
                self._validate_case_annotations(case, line_number, diagnostics)
            except ValidationError as exc:
                for err in exc.errors(include_url=False):
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    diagnostics.append(
                        ValidationDiagnostic(
                            code=DiagnosticCode.DATASET_SCHEMA_ERROR,
                            severity=DiagnosticSeverity.ERROR,
                            message=(
                                f"Schema violation at line {line_number} in "
                                f"'{field_name}': {err['msg']}"
                            ),
                            line_number=line_number,
                            query_id=data.get("query_id"),
                            field=field_name,
                        )
                    )

        if non_empty_lines == 0:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.DATASET_EMPTY,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Dataset file '{path}' contains no records.",
                )
            )

        effective_track = expected_track or inferred_track
        if (
            expected_track is not None
            and inferred_track is not None
            and inferred_track != expected_track
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.TRACK_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Dataset declares track '{inferred_track}' "
                        f"but expected track is '{expected_track}'."
                    ),
                    field="track",
                )
            )
        elif effective_track is not None and inferred_track is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.MISSING_TRACK_METADATA,
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "Dataset does not declare an explicit track; validating against "
                        "specified track context."
                    ),
                    field="track",
                )
            )

        for idx, case in enumerate(cases, start=1):
            if (
                case.track is not None
                and effective_track is not None
                and case.track != effective_track
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CROSS_TRACK_REFERENCE,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Case declares track '{case.track}' which contradicts "
                            f"dataset track '{effective_track}'."
                        ),
                        query_id=case.query_id,
                        line_number=idx,
                        field="track",
                    )
                )

        corpus_index: CorpusIndex | None = None
        if corpus is not None:
            corpus_index, corpus_diags = self._resolve_corpus_index(corpus, effective_track)
            diagnostics.extend(corpus_diags)
            if corpus_index is not None:
                self._validate_corpus_references(cases, corpus_index, effective_track, diagnostics)

        is_valid = not any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)
        return DatasetValidationReport(
            valid=is_valid,
            dataset_id=path.stem,
            dataset_hash=computed_sha256,
            corpus_track=effective_track,
            corpus_hash=corpus_index.manifest_hash if corpus_index else None,
            cases_checked=len(cases),
            documents_checked=len(corpus_index.documents) if corpus_index else 0,
            elements_checked=len(corpus_index.element_ids) if corpus_index else 0,
            chunks_checked=len(corpus_index.chunk_ids) if corpus_index else 0,
            diagnostics=diagnostics,
        )

    def _validate_case_annotations(
        self,
        case: EvaluationCase,
        line_number: int,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        """Validate annotation status, answerability, and provenance consistency."""
        # Verified state checks
        if case.status == AnnotationStatus.VERIFIED:
            if case.provenance is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.VERIFIED_WITHOUT_PROVENANCE,
                        severity=DiagnosticSeverity.ERROR,
                        message="Case is marked VERIFIED but lacks required AnnotationProvenance.",
                        query_id=case.query_id,
                        line_number=line_number,
                        field="provenance",
                    )
                )
        elif case.status == AnnotationStatus.UNVERIFIED:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.UNVERIFIED_CANDIDATE,
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "Case is unverified; it will be excluded from official baseline scores."
                    ),
                    query_id=case.query_id,
                    line_number=line_number,
                    field="status",
                )
            )
        elif case.status == AnnotationStatus.PARTIALLY_ANNOTATED:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.PARTIAL_ANNOTATION,
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "Case is partially annotated; ensure remaining ground-truth "
                        "labels are completed."
                    ),
                    query_id=case.query_id,
                    line_number=line_number,
                    field="status",
                )
            )

        # Answerability consistency
        if case.answerability == Answerability.UNANSWERABLE:
            if case.relevant_chunk_ids or case.relevant_element_ids:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.ANNOTATION_STATE_INCONSISTENT,
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            "Query is marked UNANSWERABLE but includes relevant chunk or "
                            "element ground truth."
                        ),
                        query_id=case.query_id,
                        line_number=line_number,
                        field="answerability",
                    )
                )
        elif case.answerability == Answerability.ANSWERABLE:
            if case.reference_answer is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.MISSING_REFERENCE_ANSWER,
                        severity=DiagnosticSeverity.WARNING,
                        message="Query is answerable but provides no reference answer string.",
                        query_id=case.query_id,
                        line_number=line_number,
                        field="reference_answer",
                    )
                )

        if case.difficulty is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.MISSING_DIFFICULTY,
                    severity=DiagnosticSeverity.WARNING,
                    message="Case does not declare a difficulty level.",
                    query_id=case.query_id,
                    line_number=line_number,
                    field="difficulty",
                )
            )

    def _resolve_corpus_index(
        self,
        corpus: CorpusIndex | Path | str,
        expected_track: CorpusTrack | None,
    ) -> tuple[CorpusIndex | None, list[ValidationDiagnostic]]:
        """Resolve a corpus input into a fully indexed CorpusIndex, recording manifest findings."""
        if isinstance(corpus, CorpusIndex):
            return corpus, []

        corpus_path = Path(corpus)
        diagnostics: list[ValidationDiagnostic] = []

        if not corpus_path.exists() or not corpus_path.is_dir():
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_DOCUMENT_MISSING,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Corpus path '{corpus_path}' does not exist or is not a directory.",
                )
            )
            return None, diagnostics

        manifest_file = corpus_path / "manifest.json"
        provenance_file = corpus_path / "PROVENANCE.json"

        manifest: CorpusManifest | None = None
        manifest_hash: str | None = None

        if manifest_file.exists():
            manifest, manifest_hash, m_diags = self._load_corpus_manifest(
                manifest_file, corpus_path
            )
            diagnostics.extend(m_diags)
        elif provenance_file.exists():
            manifest, manifest_hash, p_diags = self._load_provenance_manifest(
                provenance_file, corpus_path
            )
            diagnostics.extend(p_diags)
        else:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_MANIFEST_MISSING,
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"No manifest.json or PROVENANCE.json found in corpus "
                        f"directory '{corpus_path}'."
                    ),
                )
            )

        if manifest is not None and expected_track is not None and manifest.track != expected_track:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.TRACK_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Corpus manifest declares track '{manifest.track}' "
                        f"but expected '{expected_track}'."
                    ),
                    field="track",
                )
            )

        # Ingest and index corpus documents
        documents: list[NormalizedDocument] = []
        chunks: list[Chunk] = []
        filename_by_doc_id: dict[str, str] = {}

        candidate_files = [
            f
            for f in corpus_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in {".txt", ".md"}
            and f.name not in {"manifest.json", "PROVENANCE.json"}
        ]

        # Check for unexpected files if manifest exists
        if manifest is not None:
            manifest_filenames = {entry.filename for entry in manifest.documents}
            for candidate in candidate_files:
                if candidate.name not in manifest_filenames:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code=DiagnosticCode.CORPUS_UNEXPECTED_FILE,
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                f"Found file '{candidate.name}' in corpus directory "
                                "not declared in manifest."
                            ),
                            document_id=candidate.name,
                        )
                    )

        for doc_file in candidate_files:
            ingest_result = self._pipeline.ingest(doc_file)
            if ingest_result.document is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CORPUS_INGESTION_ERROR,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Failed to parse document '{doc_file.name}': {ingest_result.errors}"
                        ),
                        document_id=doc_file.name,
                    )
                )
                continue

            documents.append(ingest_result.document)
            chunks.extend(ingest_result.chunks)
            filename_by_doc_id[ingest_result.document.metadata.document_id] = doc_file.name

        index = CorpusIndex(
            documents=documents,
            chunks=chunks,
            track=manifest.track if manifest else expected_track,
            manifest=manifest,
            manifest_hash=manifest_hash,
            filename_by_doc_id=filename_by_doc_id,
        )
        return index, diagnostics

    def _load_corpus_manifest(
        self,
        manifest_path: Path,
        corpus_dir: Path,
    ) -> tuple[CorpusManifest | None, str | None, list[ValidationDiagnostic]]:
        """Load and verify a manifest.json file and its document hashes."""
        diagnostics: list[ValidationDiagnostic] = []
        raw_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(raw_bytes).hexdigest()

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            manifest = CorpusManifest.model_validate(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_MANIFEST_INVALID,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Invalid corpus manifest at '{manifest_path}': {exc}",
                )
            )
            return None, manifest_hash, diagnostics

        seen_filenames: set[str] = set()
        for doc_entry in manifest.documents:
            if doc_entry.filename in seen_filenames:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CORPUS_DUPLICATE_DOCUMENT_ID,
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Duplicate document entry '{doc_entry.filename}' in manifest.",
                        document_id=doc_entry.filename,
                    )
                )
            seen_filenames.add(doc_entry.filename)

            target_path = corpus_dir / doc_entry.filename
            if not target_path.exists():
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CORPUS_DOCUMENT_MISSING,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Manifest references document '{doc_entry.filename}' "
                            f"which does not exist in '{corpus_dir}'."
                        ),
                        document_id=doc_entry.filename,
                    )
                )
                continue

            actual_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
            if actual_sha256.lower() != doc_entry.sha256.lower():
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CORPUS_DOCUMENT_HASH_MISMATCH,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Content SHA-256 for '{doc_entry.filename}' ({actual_sha256}) "
                            f"does not match manifest ({doc_entry.sha256})."
                        ),
                        document_id=doc_entry.filename,
                    )
                )

        return manifest, manifest_hash, diagnostics

    def _load_provenance_manifest(
        self,
        prov_path: Path,
        corpus_dir: Path,
    ) -> tuple[CorpusManifest | None, str | None, list[ValidationDiagnostic]]:
        """Convert a Track B PROVENANCE.json file into a verified CorpusManifest representation."""
        diagnostics: list[ValidationDiagnostic] = []
        raw_bytes = prov_path.read_bytes()
        prov_hash = hashlib.sha256(raw_bytes).hexdigest()

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_MANIFEST_INVALID,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Invalid PROVENANCE.json at '{prov_path}': {exc}",
                )
            )
            return None, prov_hash, diagnostics

        # Determine target file (e.g. rfc8259_json.txt)
        filename = data.get("filename") or f"{data.get('document_id', 'doc')}.txt"
        target_path = corpus_dir / filename
        if not target_path.exists():
            # Check if there is an alternative matching txt/md file
            matching = [
                f for f in corpus_dir.iterdir() if f.is_file() and f.suffix in {".txt", ".md"}
            ]
            if len(matching) == 1:
                target_path = matching[0]
                filename = target_path.name
            else:
                diagnostics.append(
                    ValidationDiagnostic(
                        code=DiagnosticCode.CORPUS_DOCUMENT_MISSING,
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Provenance points to document '{filename}' which was not "
                            f"found in '{corpus_dir}'."
                        ),
                        document_id=filename,
                    )
                )
                return None, prov_hash, diagnostics

        actual_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
        expected_sha256 = data.get("content_sha256", "")
        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            diagnostics.append(
                ValidationDiagnostic(
                    code=DiagnosticCode.CORPUS_DOCUMENT_HASH_MISMATCH,
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Document '{filename}' SHA-256 ({actual_sha256}) does not match "
                        f"provenance ({expected_sha256})."
                    ),
                    document_id=filename,
                )
            )

        manifest = CorpusManifest(
            track=CorpusTrack.TRACK_B,
            manifest_version="1.0.0",
            documents=[
                CorpusDocumentEntry(
                    filename=filename,
                    title=data.get("title", filename),
                    sha256=expected_sha256 or actual_sha256,
                    document_id=data.get("document_id"),
                    source_url=data.get("source_url"),
                    license_or_reuse_terms=data.get("license_or_reuse_terms"),
                )
            ],
            description=data.get("title"),
        )
        return manifest, prov_hash, diagnostics

    def _validate_corpus_references(
        self,
        cases: Sequence[EvaluationCase],
        index: CorpusIndex,
        effective_track: CorpusTrack | None,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        """Cross-reference evaluation case element and chunk IDs against the indexed corpus."""
        for case in cases:
            # 1. Element ID reference checks
            for elem_id in case.relevant_element_ids:
                if elem_id not in index.element_ids:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code=DiagnosticCode.UNKNOWN_ELEMENT_REFERENCE,
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Case references unknown source element ID: '{elem_id}'.",
                            query_id=case.query_id,
                            element_id=elem_id,
                            field="relevant_element_ids",
                        )
                    )

            # 2. Chunk ID reference checks
            for chunk_id in case.relevant_chunk_ids:
                if chunk_id not in index.chunk_ids:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code=DiagnosticCode.UNKNOWN_CHUNK_REFERENCE,
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Case references unknown chunk ID: '{chunk_id}'.",
                            query_id=case.query_id,
                            chunk_id=chunk_id,
                            field="relevant_chunk_ids",
                        )
                    )

            # 3. Chunk-to-Element alignment checks
            if case.relevant_chunk_ids and case.relevant_element_ids:
                case_elem_set = set(case.relevant_element_ids)
                for chunk_id in case.relevant_chunk_ids:
                    source_elems = index.chunk_to_elements.get(chunk_id)
                    if source_elems is not None and not (source_elems & case_elem_set):
                        diagnostics.append(
                            ValidationDiagnostic(
                                code=DiagnosticCode.CHUNK_ELEMENT_MISMATCH,
                                severity=DiagnosticSeverity.WARNING,
                                message=(
                                    f"Chunk '{chunk_id}' does not contain any of the "
                                    "case's declared relevant elements."
                                ),
                                query_id=case.query_id,
                                chunk_id=chunk_id,
                                field="relevant_chunk_ids",
                            )
                        )

            # 4. Provenance source document and evidence reference checks
            if case.provenance is not None:
                for src_doc in case.provenance.source_documents:
                    if src_doc not in index.filenames and src_doc not in index.document_ids:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code=DiagnosticCode.PROVENANCE_DOCUMENT_MISSING,
                                severity=DiagnosticSeverity.ERROR,
                                message=(
                                    f"Provenance source_document '{src_doc}' does not "
                                    "exist in the referenced corpus."
                                ),
                                query_id=case.query_id,
                                document_id=src_doc,
                                field="provenance.source_documents",
                            )
                        )

                for ev_ref in case.provenance.evidence_references:
                    if ev_ref not in index.element_ids and ev_ref not in index.chunk_ids:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code=DiagnosticCode.PROVENANCE_EVIDENCE_MISSING,
                                severity=DiagnosticSeverity.ERROR,
                                message=(
                                    f"Provenance evidence reference '{ev_ref}' not "
                                    "found among corpus elements or chunks."
                                ),
                                query_id=case.query_id,
                                element_id=ev_ref,
                                field="provenance.evidence_references",
                            )
                        )


def validate_dataset_structure(
    dataset_or_path: EvaluationDataset | Path | str,
    expected_hash: str | None = None,
) -> DatasetValidationReport:
    """Convenience entry point for pure structural validation of a dataset or JSONL file."""
    validator = DatasetValidator()
    if isinstance(dataset_or_path, (Path, str)):
        return validator.validate_file(dataset_or_path, expected_hash=expected_hash)
    return validator.validate_dataset(dataset_or_path, expected_hash=expected_hash)


def validate_dataset(
    dataset_or_path: EvaluationDataset | Path | str,
    corpus: CorpusIndex | Path | str | None = None,
    expected_track: CorpusTrack | None = None,
    expected_hash: str | None = None,
) -> DatasetValidationReport:
    """Primary public validator API for evaluation datasets and corpus consistency."""
    validator = DatasetValidator()
    if isinstance(dataset_or_path, (Path, str)):
        return validator.validate_file(
            dataset_or_path,
            corpus=corpus,
            expected_track=expected_track,
            expected_hash=expected_hash,
        )
    return validator.validate_dataset(
        dataset_or_path,
        corpus=corpus,
        expected_track=expected_track,
        expected_hash=expected_hash,
    )
