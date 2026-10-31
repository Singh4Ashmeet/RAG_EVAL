"""Unit tests for M2.1 document ingestion contracts and domain exceptions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import (
    CorruptedFileError,
    ParserError,
    RAGEvalError,
    UnsupportedFormatError,
)
from core.ingest import IngestionResult, IngestionStatus
from core.normalize.schema import Chunk, DocumentMetadata, NormalizedDocument


class TestIngestionExceptionHierarchy:
    """Verify that new ingestion domain exceptions inherit correctly."""

    def test_inheritance(self) -> None:
        assert issubclass(UnsupportedFormatError, ParserError)
        assert issubclass(UnsupportedFormatError, RAGEvalError)
        assert issubclass(CorruptedFileError, ParserError)
        assert issubclass(CorruptedFileError, RAGEvalError)

    def test_catchable_as_parser_and_rag_eval_error(self) -> None:
        with pytest.raises(ParserError):
            raise UnsupportedFormatError("Format .xyz not supported")

        with pytest.raises(RAGEvalError):
            raise UnsupportedFormatError("Format .xyz not supported")

        with pytest.raises(ParserError):
            raise CorruptedFileError("File header is corrupted")

        with pytest.raises(RAGEvalError):
            raise CorruptedFileError("File header is corrupted")

    def test_exception_messages(self) -> None:
        err1 = UnsupportedFormatError("Unsupported format")
        err2 = CorruptedFileError("Cannot decode UTF-8")
        assert str(err1) == "Unsupported format"
        assert str(err2) == "Cannot decode UTF-8"


class TestIngestionStatusEnum:
    """Verify IngestionStatus enumeration members and behavior."""

    def test_enum_members_and_values(self) -> None:
        assert IngestionStatus.SUCCESS == "success"
        assert IngestionStatus.PARTIAL_SUCCESS == "partial_success"
        assert IngestionStatus.FAILED == "failed"

    def test_is_str_enum(self) -> None:
        assert isinstance(IngestionStatus.SUCCESS, str)
        assert isinstance(IngestionStatus.PARTIAL_SUCCESS, str)
        assert isinstance(IngestionStatus.FAILED, str)
        assert set(IngestionStatus) == {
            IngestionStatus.SUCCESS,
            IngestionStatus.PARTIAL_SUCCESS,
            IngestionStatus.FAILED,
        }


class TestIngestionResultSchema:
    """Verify IngestionResult validation, defaults, and serialization."""

    @pytest.fixture
    def sample_document(self) -> NormalizedDocument:
        metadata = DocumentMetadata(
            document_id="doc-test-1",
            filename="sample.txt",
        )
        return NormalizedDocument(metadata=metadata, elements=[])

    @pytest.fixture
    def sample_chunk(self) -> Chunk:
        return Chunk(
            chunk_id="chunk-test-1",
            document_id="doc-test-1",
            text="This is a test chunk.",
            chunk_index=0,
        )

    def test_valid_success_construction(
        self,
        sample_document: NormalizedDocument,
        sample_chunk: Chunk,
    ) -> None:
        result = IngestionResult(
            document_id="doc-test-1",
            status=IngestionStatus.SUCCESS,
            source_path=Path("docs/sample.txt"),
            document=sample_document,
            chunks=[sample_chunk],
            errors=[],
            duration_ms=42.5,
        )
        assert result.document_id == "doc-test-1"
        assert result.status == IngestionStatus.SUCCESS
        assert result.source_path == Path("docs/sample.txt")
        assert result.document == sample_document
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "chunk-test-1"
        assert result.errors == []
        assert result.duration_ms == 42.5

    def test_valid_failed_construction_without_document(self) -> None:
        result = IngestionResult(
            document_id="doc-failed-1",
            status=IngestionStatus.FAILED,
            source_path=Path("docs/bad.bin"),
            document=None,
            chunks=[],
            errors=["CorruptedFileError: Unable to parse binary file"],
            duration_ms=1.2,
        )
        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "CorruptedFileError" in result.errors[0]

    def test_valid_partial_success(
        self,
        sample_document: NormalizedDocument,
        sample_chunk: Chunk,
    ) -> None:
        result = IngestionResult(
            document_id="doc-partial-1",
            status=IngestionStatus.PARTIAL_SUCCESS,
            source_path=Path("docs/multipage.pdf"),
            document=sample_document,
            chunks=[sample_chunk],
            errors=["Page 3 OCR extraction timed out"],
            duration_ms=150.0,
        )
        assert result.status == IngestionStatus.PARTIAL_SUCCESS
        assert result.document is not None
        assert len(result.errors) == 1

    def test_defaults_for_optional_fields(self) -> None:
        result = IngestionResult(
            document_id="doc-defaults",
            status=IngestionStatus.SUCCESS,
            source_path=Path("test.txt"),
            duration_ms=5.0,
        )
        assert result.document is None
        assert result.chunks == []
        assert result.errors == []

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            IngestionResult.model_validate({})
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert {"document_id", "status", "source_path", "duration_ms"}.issubset(missing_fields)

    def test_invalid_document_id(self) -> None:
        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=1.0,
            )

        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="   ",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=1.0,
            )

    def test_invalid_duration_ms(self) -> None:
        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="doc-1",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=-0.1,
            )

        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="doc-1",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=float("nan"),
            )

        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="doc-1",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=float("inf"),
            )

    def test_source_path_coercion_from_string(self) -> None:
        result = IngestionResult(
            document_id="doc-str-path",
            status=IngestionStatus.SUCCESS,
            source_path="data/raw.txt",  # type: ignore[arg-type]
            duration_ms=10.0,
        )
        assert isinstance(result.source_path, Path)
        assert result.source_path == Path("data/raw.txt")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            IngestionResult(
                document_id="doc-1",
                status=IngestionStatus.SUCCESS,
                source_path=Path("test.txt"),
                duration_ms=1.0,
                unknown_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_serialization_roundtrip(
        self,
        sample_document: NormalizedDocument,
        sample_chunk: Chunk,
    ) -> None:
        original = IngestionResult(
            document_id="doc-roundtrip",
            status=IngestionStatus.SUCCESS,
            source_path=Path("corpus/sample.md"),
            document=sample_document,
            chunks=[sample_chunk],
            errors=["Minor parsing warning"],
            duration_ms=88.4,
        )
        json_data = original.model_dump_json()
        restored = IngestionResult.model_validate_json(json_data)
        assert restored == original
        assert restored.source_path == original.source_path
        assert restored.status == IngestionStatus.SUCCESS


class TestPublicExports:
    """Verify package initializers expose expected public symbols."""

    def test_core_ingest_exports(self) -> None:
        from core.ingest import IngestionResult, IngestionStatus

        assert IngestionResult is not None
        assert IngestionStatus is not None

    def test_core_errors_exports(self) -> None:
        from core.errors import CorruptedFileError, UnsupportedFormatError

        assert CorruptedFileError is not None
        assert UnsupportedFormatError is not None
