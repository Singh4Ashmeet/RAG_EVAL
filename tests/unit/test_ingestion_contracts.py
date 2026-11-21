"""Unit tests for M2.1 document ingestion contracts and domain exceptions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import (
    CorruptedFileError,
    EncryptedDocumentError,
    OCRError,
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
        assert issubclass(EncryptedDocumentError, ParserError)
        assert issubclass(EncryptedDocumentError, RAGEvalError)
        assert issubclass(OCRError, ParserError)
        assert issubclass(OCRError, RAGEvalError)

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
        from core.errors import (
            CorruptedFileError,
            EncryptedDocumentError,
            OCRError,
            UnsupportedFormatError,
        )

        assert CorruptedFileError is not None
        assert EncryptedDocumentError is not None
        assert OCRError is not None
        assert UnsupportedFormatError is not None


class TestMultiFormatContracts:
    """Contract tests verifying the unified document representation across all parsers."""

    FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"

    @pytest.mark.parametrize(
        ("parser_cls_name", "filename"),
        [
            ("PlainTextParser", "sample.txt"),
            ("PlainTextParser", "sample.md"),
            ("PDFParser", "sample.pdf"),
            ("DOCXParser", "sample.docx"),
        ],
    )
    def test_parser_protocol_and_deterministic_identity(
        self, parser_cls_name: str, filename: str
    ) -> None:
        import hashlib

        from core.parsers import DocumentParser, DOCXParser, PDFParser, PlainTextParser

        parsers = {
            "PlainTextParser": PlainTextParser,
            "PDFParser": PDFParser,
            "DOCXParser": DOCXParser,
        }
        file_path = self.FIXTURES_DIR / filename
        assert file_path.exists()

        parser = parsers[parser_cls_name]()
        assert isinstance(parser, DocumentParser)

        doc = parser.parse(file_path)
        assert isinstance(doc, NormalizedDocument)

        raw_bytes = file_path.read_bytes()
        expected_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
        expected_id = f"doc_{expected_sha[:16]}"
        assert doc.metadata.document_id == expected_id
        assert doc.metadata.content_sha256 == expected_sha
        assert doc.metadata.byte_size == len(raw_bytes)
        assert doc.metadata.filename == filename

        assert len(doc.elements) > 0
        for idx, elem in enumerate(doc.elements):
            assert elem.order_index == idx
            assert elem.element_id == f"{expected_id}_e{idx:04d}"
            assert isinstance(elem.content, str)
            assert len(elem.content.strip()) > 0

    @pytest.mark.parametrize(
        "filename",
        ["sample.txt", "sample.md", "sample.pdf", "sample.docx"],
    )
    def test_chunking_contract_across_all_formats(self, filename: str) -> None:
        from core.ingest.pipeline import create_default_ingestion_pipeline

        file_path = self.FIXTURES_DIR / filename
        pipeline = create_default_ingestion_pipeline(chunk_size=500, chunk_overlap=100)

        result = pipeline.ingest(file_path)
        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert len(result.chunks) > 0
        assert result.errors == []

        doc_id = result.document.metadata.document_id
        for c_idx, chunk in enumerate(result.chunks):
            assert chunk.document_id == doc_id
            assert chunk.chunk_id == f"{doc_id}_c{c_idx:04d}"
            assert len(chunk.text.strip()) > 0
            assert len(chunk.source_element_ids) > 0

    def test_pdf_provenance_flows_to_chunks(self) -> None:
        from core.ingest.pipeline import create_default_ingestion_pipeline

        pdf_path = self.FIXTURES_DIR / "sample.pdf"
        pipeline = create_default_ingestion_pipeline()
        result = pipeline.ingest(pdf_path)

        assert result.status == IngestionStatus.SUCCESS
        assert len(result.chunks) >= 1

        chunk_pages = [loc.page_number for chunk in result.chunks for loc in chunk.locations]
        assert 1 in chunk_pages
        assert 2 in chunk_pages

    def test_image_parser_protocol_with_ocr_engine(self) -> None:
        from PIL import Image

        from core.chunking.chunker import ElementAwareChunker
        from core.parsers.base import DocumentParser
        from core.parsers.image_parser import ImageParser

        img_path = self.FIXTURES_DIR / "sample.png"

        def mock_ocr(_: Image.Image) -> str:
            return "Sample OCR Line"

        parser = ImageParser(ocr_engine=mock_ocr)
        assert isinstance(parser, DocumentParser)
        doc = parser.parse(img_path)

        assert isinstance(doc, NormalizedDocument)
        chunker = ElementAwareChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Sample OCR Line"
        assert len(chunks[0].locations) == 1
        assert chunks[0].locations[0].page_number == 1
