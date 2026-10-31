"""Unit tests for the M2.1 Step 4 IngestionPipeline orchestrator."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.chunking import ElementAwareChunker
from core.errors import ChunkerError, ParserError
from core.ingest import IngestionPipeline, IngestionResult, IngestionStatus
from core.normalize.schema import (
    Chunk,
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
)
from core.parsers import ParserRegistry, PlainTextParser


class CountingParser:
    """Stub parser that counts invocations and can optionally raise an exception."""

    def __init__(
        self,
        return_doc: NormalizedDocument | None = None,
        exc_to_raise: Exception | None = None,
    ) -> None:
        self.call_count = 0
        self.last_path: Path | None = None
        self.return_doc = return_doc
        self.exc_to_raise = exc_to_raise

    def parse(self, file_path: Path) -> NormalizedDocument:
        self.call_count += 1
        self.last_path = file_path
        if self.exc_to_raise is not None:
            raise self.exc_to_raise
        assert self.return_doc is not None
        return self.return_doc


class CountingChunker:
    """Stub chunker that counts invocations and can optionally raise an exception."""

    def __init__(
        self,
        return_chunks: list[Chunk] | None = None,
        exc_to_raise: Exception | None = None,
    ) -> None:
        self.call_count = 0
        self.last_doc: NormalizedDocument | None = None
        self.return_chunks = return_chunks if return_chunks is not None else []
        self.exc_to_raise = exc_to_raise

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        self.call_count += 1
        self.last_doc = document
        if self.exc_to_raise is not None:
            raise self.exc_to_raise
        return self.return_chunks


@pytest.fixture
def standard_pipeline() -> tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker]:
    registry = ParserRegistry()
    parser = PlainTextParser()
    registry.register([".txt", ".md"], parser)
    chunker = ElementAwareChunker(chunk_size=100, chunk_overlap=20)
    pipeline = IngestionPipeline(parser_registry=registry, chunker=chunker)
    return pipeline, registry, chunker


class TestIngestionPipelineSuccess:
    """Tests for successful ingestion workflows."""

    def test_successful_text_file_ingestion(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        text_file = tmp_path / "sample.txt"
        text_file.write_text("This is paragraph one.\n\nThis is paragraph two.", encoding="utf-8")

        result = pipeline.ingest(text_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.source_path == text_file
        assert result.document is not None
        assert result.document_id == result.document.metadata.document_id
        assert len(result.document.elements) == 2
        assert len(result.chunks) > 0
        assert result.errors == []
        assert result.duration_ms >= 0.0

    def test_successful_markdown_ingestion(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        md_file = tmp_path / "guide.md"
        md_file.write_text(
            "# Overview\n\nIntroductory text here.\n\n- Feature A\n- Feature B",
            encoding="utf-8",
        )

        result = pipeline.ingest(md_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert len(result.document.elements) == 4
        # Verify markdown element types extracted
        elem_types = [e.element_type for e in result.document.elements]
        assert ElementType.HEADER in elem_types
        assert ElementType.LIST_ITEM in elem_types
        assert len(result.chunks) > 0
        assert result.errors == []

    def test_string_file_path_coercion(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        file = tmp_path / "string_path.txt"
        file.write_text("Testing path coercion.", encoding="utf-8")

        # Pass as string instead of Path
        result = pipeline.ingest(str(file))

        assert result.status == IngestionStatus.SUCCESS
        assert isinstance(result.source_path, Path)
        assert result.source_path == file

    def test_empty_document_ingestion(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        result = pipeline.ingest(empty_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert result.document.elements == []
        assert result.chunks == []
        assert result.errors == []
        assert result.document_id == result.document.metadata.document_id

    def test_whitespace_only_document_ingestion(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        ws_file = tmp_path / "whitespace.txt"
        ws_file.write_text("   \n\n\t\t\n   ", encoding="utf-8")

        result = pipeline.ingest(ws_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert result.document.elements == []
        assert result.chunks == []
        assert result.errors == []


class TestIngestionPipelineRecoverableFailures:
    """Tests for expected, recoverable failure scenarios."""

    def test_unsupported_file_extension(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        unsupported = tmp_path / "data.unregistered"
        unsupported.write_text("Some content", encoding="utf-8")

        result = pipeline.ingest(unsupported)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Unsupported file format" in result.errors[0]
        assert result.document_id.startswith("doc_unprocessed_")

    def test_missing_file_handling(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        missing = tmp_path / "does_not_exist.txt"

        result = pipeline.ingest(missing)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Failed to read file" in result.errors[0]
        assert result.document_id.startswith("doc_unprocessed_")

    def test_corrupted_invalid_utf8_input(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        corrupted = tmp_path / "corrupted.txt"
        # Write invalid UTF-8 byte sequences
        corrupted.write_bytes(b"\x80\x81\xff\xfe")

        result = pipeline.ingest(corrupted)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Failed to decode" in result.errors[0]
        assert result.document_id.startswith("doc_unprocessed_")

    def test_parser_failure_handling(self, tmp_path: Path) -> None:
        file = tmp_path / "doc.txt"
        file.write_text("Valid text", encoding="utf-8")

        registry = ParserRegistry()
        stub_parser = CountingParser(exc_to_raise=ParserError("Simulated parser extraction fault"))
        registry.register(".txt", stub_parser)
        stub_chunker = CountingChunker()

        pipeline = IngestionPipeline(registry, stub_chunker)
        result = pipeline.ingest(file)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Simulated parser extraction fault" in result.errors[0]
        assert stub_parser.call_count == 1
        assert stub_chunker.call_count == 0  # Chunker must not be invoked on parser error

    def test_chunker_failure_handling(self, tmp_path: Path) -> None:
        file = tmp_path / "doc.txt"
        file.write_text("Valid text", encoding="utf-8")

        doc = NormalizedDocument(
            metadata=DocumentMetadata(document_id="doc_test_123", filename="doc.txt"),
            elements=[
                ExtractedElement(
                    element_id="doc_test_123_e0000",
                    element_type=ElementType.TEXT,
                    content="Hello world",
                    order_index=0,
                )
            ],
        )

        registry = ParserRegistry()
        stub_parser = CountingParser(return_doc=doc)
        registry.register(".txt", stub_parser)
        stub_chunker = CountingChunker(exc_to_raise=ChunkerError("Simulated chunker failure"))

        pipeline = IngestionPipeline(registry, stub_chunker)
        result = pipeline.ingest(file)

        assert result.status == IngestionStatus.FAILED
        # Document was parsed, so it should be preserved along with its canonical ID
        assert result.document is doc
        assert result.document_id == "doc_test_123"
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Simulated chunker failure" in result.errors[0]
        assert stub_parser.call_count == 1
        assert stub_chunker.call_count == 1


class TestIngestionPipelineContractsAndProvenance:
    """Tests asserting schema validity, provenance, timing, and exception propagation."""

    def test_document_id_propagation_and_chunk_provenance(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        file = tmp_path / "provenance.txt"
        file.write_text("Element one content.\n\nElement two content.", encoding="utf-8")

        result = pipeline.ingest(file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        canonical_doc_id = result.document.metadata.document_id
        assert result.document_id == canonical_doc_id

        doc_elem_ids = {e.element_id for e in result.document.elements}

        # Check every chunk preserves provenance
        for chunk in result.chunks:
            assert chunk.document_id == canonical_doc_id
            assert set(chunk.source_element_ids).issubset(doc_elem_ids)
            assert chunk.chunk_id.startswith(f"{canonical_doc_id}_c")

    def test_nonnegative_finite_duration(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        file = tmp_path / "timing.txt"
        file.write_text("Simple timing test.", encoding="utf-8")

        result = pipeline.ingest(file)

        assert result.duration_ms >= 0.0
        assert not math.isnan(result.duration_ms)
        assert not math.isinf(result.duration_ms)

    def test_unexpected_exception_propagates(self, tmp_path: Path) -> None:
        file = tmp_path / "unexpected.txt"
        file.write_text("Some text", encoding="utf-8")

        registry = ParserRegistry()
        stub_parser = CountingParser(exc_to_raise=TypeError("Unexpected type bug in parser"))
        registry.register(".txt", stub_parser)
        stub_chunker = CountingChunker()

        pipeline = IngestionPipeline(registry, stub_chunker)

        # Programming errors must NOT be swallowed or converted to IngestionStatus.FAILED
        with pytest.raises(TypeError, match="Unexpected type bug in parser"):
            pipeline.ingest(file)

    def test_unexpected_chunker_exception_propagates(self, tmp_path: Path) -> None:
        file = tmp_path / "unexpected_chunker.txt"
        file.write_text("Some text", encoding="utf-8")

        doc = NormalizedDocument(
            metadata=DocumentMetadata(document_id="doc_999", filename="test.txt"),
            elements=[],
        )
        registry = ParserRegistry()
        stub_parser = CountingParser(return_doc=doc)
        registry.register(".txt", stub_parser)
        stub_chunker = CountingChunker(exc_to_raise=KeyError("Uncaught key error in chunker"))

        pipeline = IngestionPipeline(registry, stub_chunker)

        with pytest.raises(KeyError, match="Uncaught key error in chunker"):
            pipeline.ingest(file)

    def test_result_schema_validation(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        pipeline, _, _ = standard_pipeline
        file = tmp_path / "schema_test.txt"
        file.write_text("Text for schema validation test.", encoding="utf-8")

        result = pipeline.ingest(file)

        # Ensure result passes Pydantic model validation and serialization roundtrip
        dumped = result.model_dump()
        reconstructed = IngestionResult.model_validate(dumped)
        assert reconstructed == result

        json_str = result.model_dump_json()
        restored = IngestionResult.model_validate_json(json_str)
        assert restored == result

    def test_zero_external_dependencies(
        self,
        standard_pipeline: tuple[IngestionPipeline, ParserRegistry, ElementAwareChunker],
        tmp_path: Path,
    ) -> None:
        # Pipeline operates purely in-memory and local filesystem without DB/network
        pipeline, _, _ = standard_pipeline
        file = tmp_path / "isolated.txt"
        file.write_text("Completely local and offline.", encoding="utf-8")

        result = pipeline.ingest(file)
        assert result.status == IngestionStatus.SUCCESS


class TestPublicExports:
    """Verify core.ingest public symbol exports."""

    def test_core_ingest_exports(self) -> None:
        from core.ingest import IngestionPipeline, IngestionResult, IngestionStatus

        assert IngestionPipeline is not None
        assert IngestionResult is not None
        assert IngestionStatus is not None
