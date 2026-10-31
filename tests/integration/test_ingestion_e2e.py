"""End-to-end integration tests for default ingestion factory and CLI integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from apps.cli.ingest import main
from core.chunking import ElementAwareChunker
from core.ingest import (
    IngestionPipeline,
    IngestionResult,
    IngestionStatus,
    create_default_ingestion_pipeline,
)
from core.normalize.schema import Chunk, DocumentMetadata, NormalizedDocument


class TestDefaultIngestionPipelineFactory:
    """Tests for create_default_ingestion_pipeline factory behavior and configuration."""

    def test_factory_returns_ingestion_pipeline(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        assert isinstance(pipeline, IngestionPipeline)

    def test_factory_registers_txt_extension(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        supported = pipeline.parser_registry.supported_extensions()
        assert ".txt" in supported

    def test_factory_registers_md_extension(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        supported = pipeline.parser_registry.supported_extensions()
        assert ".md" in supported

    def test_factory_supported_extensions_exact(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        supported = pipeline.parser_registry.supported_extensions()
        # PlainTextParser only supports .txt and .md
        assert supported == {".txt", ".md"}

    def test_factory_uses_element_aware_chunker_with_defaults(self) -> None:
        pipeline = create_default_ingestion_pipeline()
        assert isinstance(pipeline.chunker, ElementAwareChunker)
        assert pipeline.chunker.chunk_size == 1000
        assert pipeline.chunker.chunk_overlap == 200

    def test_core_ingest_exports_factory(self) -> None:
        from core.ingest import create_default_ingestion_pipeline as factory_fn

        assert callable(factory_fn)


class TestEndToEndPipelineIngestion:
    """End-to-end ingestion tests using the factory-constructed pipeline."""

    def test_e2e_valid_text_file(self, tmp_path: Path) -> None:
        pipeline = create_default_ingestion_pipeline()
        text_file = tmp_path / "integration_doc.txt"
        text_file.write_text(
            "First section of text.\n\nSecond section with more details.\n\nConclusion.",
            encoding="utf-8",
        )

        result = pipeline.ingest(text_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.source_path == text_file
        assert result.document is not None
        assert result.document_id == result.document.metadata.document_id
        assert len(result.document.elements) == 3
        assert len(result.chunks) > 0
        assert result.errors == []
        assert result.duration_ms >= 0.0

    def test_e2e_valid_markdown_file(self, tmp_path: Path) -> None:
        pipeline = create_default_ingestion_pipeline()
        md_file = tmp_path / "integration_doc.md"
        md_file.write_text(
            "# Chapter 1\n\nIntroduction paragraph.\n\n- Bullet item 1\n- Bullet item 2",
            encoding="utf-8",
        )

        result = pipeline.ingest(md_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert len(result.document.elements) == 4
        assert len(result.chunks) > 0
        assert result.errors == []

    def test_e2e_empty_file(self, tmp_path: Path) -> None:
        pipeline = create_default_ingestion_pipeline()
        empty_file = tmp_path / "empty_doc.txt"
        empty_file.write_text("", encoding="utf-8")

        result = pipeline.ingest(empty_file)

        assert result.status == IngestionStatus.SUCCESS
        assert result.document is not None
        assert result.document.elements == []
        assert result.chunks == []
        assert result.errors == []

    def test_e2e_unsupported_extension(self, tmp_path: Path) -> None:
        pipeline = create_default_ingestion_pipeline()
        pdf_file = tmp_path / "unsupported.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy binary content")

        result = pipeline.ingest(pdf_file)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Unsupported file format '.pdf'" in result.errors[0]

    def test_e2e_missing_file(self, tmp_path: Path) -> None:
        pipeline = create_default_ingestion_pipeline()
        missing = tmp_path / "non_existent.txt"

        result = pipeline.ingest(missing)

        assert result.status == IngestionStatus.FAILED
        assert result.document is None
        assert result.chunks == []
        assert len(result.errors) == 1
        assert "Failed to read file" in result.errors[0]


class TestCommandLineInterface:
    """Tests for apps/cli/ingest.py entry point and exit code semantics."""

    def test_cli_success_output_and_exit_code(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file = tmp_path / "cli_sample.txt"
        file.write_text("Text for CLI test.\n\nSecond paragraph.", encoding="utf-8")

        exit_code = main([str(file)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Source Path:" in captured.out
        assert "cli_sample.txt" in captured.out
        assert "Ingestion Status:  success" in captured.out
        assert "Document ID:       doc_" in captured.out
        assert "Number of Chunks:" in captured.out
        assert "Duration:" in captured.out
        assert "Errors:" not in captured.out

    def test_cli_failure_output_and_exit_code(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file = tmp_path / "bad_ext.xyz"
        file.write_text("Unsupported format", encoding="utf-8")

        exit_code = main([str(file)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Ingestion Status:  failed" in captured.out
        assert "Errors:" in captured.out
        assert "Unsupported file format '.xyz'" in captured.out

    def test_cli_partial_success_returns_exit_code_1(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file = tmp_path / "partial.txt"
        file.write_text("Content", encoding="utf-8")

        partial_result = IngestionResult(
            document_id="doc_partial_001",
            status=IngestionStatus.PARTIAL_SUCCESS,
            source_path=file,
            document=NormalizedDocument(
                metadata=DocumentMetadata(document_id="doc_partial_001", filename="partial.txt"),
                elements=[],
            ),
            chunks=[
                Chunk(
                    chunk_id="doc_partial_001_c0000",
                    document_id="doc_partial_001",
                    text="Content",
                    chunk_index=0,
                )
            ],
            errors=["Non-fatal extraction warning"],
            duration_ms=12.5,
        )

        with patch.object(IngestionPipeline, "ingest", return_value=partial_result):
            exit_code = main([str(file)])

        captured = capsys.readouterr()
        # Condition 2: Return 0 ONLY for SUCCESS; return 1 for FAILED and PARTIAL_SUCCESS
        assert exit_code == 1
        assert "Ingestion Status:  partial_success" in captured.out
        assert "Errors:" in captured.out
        assert "Non-fatal extraction warning" in captured.out

    def test_cli_argument_parsing_missing_arguments(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower() or "usage:" in captured.out.lower()

    def test_cli_unexpected_exception_propagates(self, tmp_path: Path) -> None:
        file = tmp_path / "crash.txt"
        file.write_text("Text", encoding="utf-8")

        with patch.object(
            IngestionPipeline,
            "ingest",
            side_effect=RuntimeError("Unexpected kernel crash"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected kernel crash"):
                main([str(file)])

    def test_no_external_dependencies_e2e(self, tmp_path: Path) -> None:
        file = tmp_path / "offline.txt"
        file.write_text("Offline verification.", encoding="utf-8")

        pipeline = create_default_ingestion_pipeline()
        result = pipeline.ingest(file)

        assert result.status == IngestionStatus.SUCCESS
