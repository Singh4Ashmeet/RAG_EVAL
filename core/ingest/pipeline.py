"""Document ingestion pipeline orchestrating parsing and chunking."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from core.chunking.base import DocumentChunker
from core.chunking.chunker import ElementAwareChunker
from core.errors import ChunkerError, ParserError
from core.ingest.schema import IngestionResult, IngestionStatus
from core.normalize.schema import NormalizedDocument
from core.parsers.registry import ParserRegistry
from core.parsers.text_parser import PlainTextParser


class IngestionPipeline:
    """Synchronous pipeline orchestrating document parsing and chunking."""

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: DocumentChunker,
    ) -> None:
        """Initialize the pipeline with a parser registry and document chunker.

        Args:
            parser_registry: Registry mapping file extensions to DocumentParser instances.
            chunker: Concrete instance implementing the DocumentChunker protocol.
        """
        self.parser_registry = parser_registry
        self.chunker = chunker

    def ingest(self, file_path: Path | str) -> IngestionResult:
        """Ingest a source document from the filesystem, parsing and chunking it.

        Args:
            file_path: Filesystem path to the source file to ingest.

        Returns:
            An IngestionResult capturing status, document, chunks, errors, and duration.
        """
        start_time = time.perf_counter()
        path = Path(file_path)

        document: NormalizedDocument | None = None

        try:
            # 1. Resolve parser via registry
            parser = self.parser_registry.get_parser(path)

            # 2. Parse file into canonical NormalizedDocument
            document = parser.parse(path)

            # 3. Chunk the document
            chunks = self.chunker.chunk(document)

            duration_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
            return IngestionResult(
                document_id=document.metadata.document_id,
                status=IngestionStatus.SUCCESS,
                source_path=path,
                document=document,
                chunks=chunks,
                errors=[],
                duration_ms=duration_ms,
            )

        except (ParserError, ChunkerError) as exc:
            duration_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)

            # If parsing succeeded and chunking failed, preserve the document identity
            if document is not None:
                doc_id = document.metadata.document_id
            else:
                doc_id = self._generate_fallback_document_id(path)

            return IngestionResult(
                document_id=doc_id,
                status=IngestionStatus.FAILED,
                source_path=path,
                document=document,
                chunks=[],
                errors=[str(exc)],
                duration_ms=duration_ms,
            )

    @staticmethod
    def _generate_fallback_document_id(path: Path) -> str:
        """Generate a deterministic fallback document ID for unparsed files."""
        path_str = path.as_posix()
        digest = hashlib.sha256(path_str.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"doc_unprocessed_{digest}"


def create_default_ingestion_pipeline() -> IngestionPipeline:
    """Create a default IngestionPipeline with registered PlainTextParser and
    ElementAwareChunker.
    """
    registry = ParserRegistry()
    text_parser = PlainTextParser()
    registry.register([".txt", ".md"], text_parser)
    chunker = ElementAwareChunker()
    return IngestionPipeline(parser_registry=registry, chunker=chunker)
