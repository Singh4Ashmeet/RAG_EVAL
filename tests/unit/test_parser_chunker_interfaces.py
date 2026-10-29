"""Unit tests for DocumentParser and DocumentChunker protocols and shared exceptions."""

from pathlib import Path

from core.chunking import DocumentChunker
from core.errors import ChunkerError, ParserError, RAGEvalError
from core.normalize.schema import (
    Chunk,
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
)
from core.parsers import DocumentParser


class DummyParser:
    """Minimal conforming implementation of DocumentParser."""

    def parse(self, file_path: Path) -> NormalizedDocument:
        meta = DocumentMetadata(
            document_id=f"doc-{file_path.stem}",
            filename=file_path.name,
        )
        elem = ExtractedElement(
            element_id="elem-001",
            element_type=ElementType.TEXT,
            content="Parsed content from dummy parser.",
            order_index=0,
            location=PageLocation(page_number=1),
        )
        return NormalizedDocument(metadata=meta, elements=[elem], page_count=1)


class DummyChunker:
    """Minimal conforming implementation of DocumentChunker."""

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, elem in enumerate(document.elements):
            chunk = Chunk(
                chunk_id=f"{document.metadata.document_id}-chunk-{idx}",
                document_id=document.metadata.document_id,
                text=elem.content,
                chunk_index=idx,
                source_element_ids=[elem.element_id],
                locations=[elem.location] if elem.location else [],
            )
            chunks.append(chunk)
        return chunks


class IncompleteParser:
    """Class missing parse method."""

    def read(self, file_path: Path) -> str:
        return "content"


class IncompleteChunker:
    """Class missing chunk method."""

    def split(self, text: str) -> list[str]:
        return [text]


# ---------------------------------------------------------------------------
# Exception Hierarchy Tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_inheritance(self):
        assert issubclass(RAGEvalError, Exception)
        assert issubclass(ParserError, RAGEvalError)
        assert issubclass(ChunkerError, RAGEvalError)

    def test_catchable_as_rag_eval_error(self):
        try:
            raise ParserError("parsing failed")
        except RAGEvalError as exc:
            assert str(exc) == "parsing failed"

        try:
            raise ChunkerError("chunking failed")
        except RAGEvalError as exc:
            assert str(exc) == "chunking failed"


# ---------------------------------------------------------------------------
# DocumentParser Protocol Tests
# ---------------------------------------------------------------------------


class TestDocumentParserProtocol:
    def test_conforming_parser_satisfies_protocol(self):
        parser = DummyParser()
        assert isinstance(parser, DocumentParser)

    def test_incomplete_parser_rejected_by_protocol(self):
        obj = IncompleteParser()
        assert not isinstance(obj, DocumentParser)

    def test_parser_returns_valid_normalized_document(self):
        parser = DummyParser()
        doc = parser.parse(Path("sample_report.pdf"))

        assert isinstance(doc, NormalizedDocument)
        assert doc.metadata.document_id == "doc-sample_report"
        assert doc.metadata.filename == "sample_report.pdf"
        assert len(doc.elements) == 1
        assert doc.elements[0].content == "Parsed content from dummy parser."
        assert doc.page_count == 1


# ---------------------------------------------------------------------------
# DocumentChunker Protocol Tests
# ---------------------------------------------------------------------------


class TestDocumentChunkerProtocol:
    def test_conforming_chunker_satisfies_protocol(self):
        chunker = DummyChunker()
        assert isinstance(chunker, DocumentChunker)

    def test_incomplete_chunker_rejected_by_protocol(self):
        obj = IncompleteChunker()
        assert not isinstance(obj, DocumentChunker)

    def test_chunker_returns_valid_chunks(self):
        parser = DummyParser()
        doc = parser.parse(Path("test_doc.pdf"))
        chunker = DummyChunker()

        chunks = chunker.chunk(doc)

        assert isinstance(chunks, list)
        assert len(chunks) == 1

        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.document_id == "doc-test_doc"
        assert chunk.chunk_id == "doc-test_doc-chunk-0"
        assert chunk.chunk_index == 0
        assert chunk.text == "Parsed content from dummy parser."
        assert chunk.source_element_ids == ["elem-001"]
        assert len(chunk.locations) == 1
        assert chunk.locations[0].page_number == 1


# ---------------------------------------------------------------------------
# Public Export Tests
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_public_imports(self):
        from core.chunking import DocumentChunker as ImportedChunker
        from core.errors import ChunkerError as ImportedChunkerError
        from core.errors import ParserError as ImportedParserError
        from core.errors import RAGEvalError as ImportedRAGEvalError
        from core.parsers import DocumentParser as ImportedParser

        assert ImportedParser is DocumentParser
        assert ImportedChunker is DocumentChunker
        assert ImportedRAGEvalError is RAGEvalError
        assert ImportedParserError is ParserError
        assert ImportedChunkerError is ChunkerError
