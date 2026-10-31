"""Unit tests for M2.1 Step 2: ParserRegistry and PlainTextParser."""

import hashlib
from datetime import UTC
from pathlib import Path

import pytest

from core.errors import CorruptedFileError, UnsupportedFormatError
from core.normalize.schema import ElementType, NormalizedDocument
from core.parsers import DocumentParser, ParserRegistry, PlainTextParser


class TestParserRegistry:
    """Verify registry routing, extension normalization, and error handling."""

    def test_register_and_get_single_extension(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register(".txt", parser)

        resolved = registry.get_parser("document.txt")
        assert resolved is parser

    def test_register_multiple_extensions(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register([".txt", ".md"], parser)

        assert registry.get_parser("note.txt") is parser
        assert registry.get_parser("readme.md") is parser

    def test_case_insensitive_and_dotless_extensions(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register("TXT", parser)
        registry.register(".MD", parser)

        assert registry.get_parser("UPPER.TXT") is parser
        assert registry.get_parser("mixed.Txt") is parser
        assert registry.get_parser("lower.txt") is parser
        assert registry.get_parser("UPPER.MD") is parser
        assert registry.get_parser("lower.md") is parser

    def test_unsupported_extension_raises_unsupported_format_error(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register(".txt", parser)

        with pytest.raises(UnsupportedFormatError) as exc_info:
            registry.get_parser("report.pdf")
        assert ".pdf" in str(exc_info.value)

    def test_file_without_extension_raises_unsupported_format_error(self) -> None:
        registry = ParserRegistry()
        with pytest.raises(UnsupportedFormatError) as exc_info:
            registry.get_parser("LICENSE")
        assert "<none>" in str(exc_info.value)

    def test_supported_extensions_set(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register([".txt", "MD"], parser)

        assert registry.supported_extensions() == {".txt", ".md"}

    def test_routes_txt_and_md_to_same_parser(self) -> None:
        registry = ParserRegistry()
        parser = PlainTextParser()
        registry.register([".txt", ".md"], parser)

        p1 = registry.get_parser(Path("test.txt"))
        p2 = registry.get_parser(Path("test.md"))
        assert p1 is p2 is parser


class TestPlainTextParserProtocolConformance:
    """Verify PlainTextParser implements DocumentParser protocol."""

    def test_runtime_protocol_check(self) -> None:
        parser = PlainTextParser()
        assert isinstance(parser, DocumentParser)


class TestPlainTextParserExecution:
    """Verify PlainTextParser behavior across various input formats and conditions."""

    def test_parse_plain_text_file(self, tmp_path: Path) -> None:
        content = (
            "First paragraph line 1.\n"
            "First paragraph line 2.\n\n"
            "Second paragraph.\n\n"
            "Third paragraph."
        )
        file_path = tmp_path / "sample.txt"
        file_path.write_bytes(content.encode("utf-8"))

        parser = PlainTextParser()
        doc = parser.parse(file_path)

        assert isinstance(doc, NormalizedDocument)
        assert doc.metadata.filename == "sample.txt"
        assert doc.metadata.mime_type == "text/plain"
        assert doc.metadata.byte_size == len(content.encode("utf-8"))
        assert doc.metadata.content_sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert doc.metadata.processed_at is not None
        assert doc.metadata.processed_at.tzinfo == UTC
        assert doc.page_count is None

        # 3 paragraphs
        assert len(doc.elements) == 3
        for idx, elem in enumerate(doc.elements):
            assert elem.order_index == idx
            assert elem.element_type == ElementType.TEXT
            assert elem.location is None
            assert elem.element_id == f"{doc.metadata.document_id}_e{idx:04d}"

        assert doc.elements[0].content == "First paragraph line 1.\nFirst paragraph line 2."
        assert doc.elements[1].content == "Second paragraph."
        assert doc.elements[2].content == "Third paragraph."

    def test_parse_markdown_file(self, tmp_path: Path) -> None:
        content = (
            "# Document Title\n\n"
            "Introductory text.\n\n"
            "## Section 1\n\n"
            "- First bullet item\n"
            "- Second bullet item\n"
            "1. Numbered item\n\n"
            "Concluding remarks."
        )
        file_path = tmp_path / "guide.md"
        file_path.write_bytes(content.encode("utf-8"))

        parser = PlainTextParser()
        doc = parser.parse(file_path)

        assert doc.metadata.filename == "guide.md"
        assert doc.metadata.mime_type == "text/markdown"

        types = [e.element_type for e in doc.elements]
        assert types == [
            ElementType.HEADER,  # # Document Title
            ElementType.TEXT,  # Introductory text.
            ElementType.HEADER,  # ## Section 1
            ElementType.LIST_ITEM,  # - First bullet item
            ElementType.LIST_ITEM,  # - Second bullet item
            ElementType.LIST_ITEM,  # 1. Numbered item
            ElementType.TEXT,  # Concluding remarks.
        ]

    def test_empty_file_behavior(self, tmp_path: Path) -> None:
        file_path = tmp_path / "empty.txt"
        file_path.write_bytes(b"")

        parser = PlainTextParser()
        doc = parser.parse(file_path)

        assert doc.metadata.byte_size == 0
        assert doc.metadata.content_sha256 == hashlib.sha256(b"").hexdigest()
        assert doc.elements == []
        assert doc.page_count is None

    def test_blank_lines_behavior(self, tmp_path: Path) -> None:
        content = "\n\n   \n\t\nOnly one valid paragraph.\n\n   \n\n"
        file_path = tmp_path / "spaced.txt"
        file_path.write_bytes(content.encode("utf-8"))

        parser = PlainTextParser()
        doc = parser.parse(file_path)

        assert len(doc.elements) == 1
        assert doc.elements[0].content == "Only one valid paragraph."
        assert doc.elements[0].order_index == 0

    def test_stable_element_ids_for_unchanged_input(self, tmp_path: Path) -> None:
        content = "Line one.\n\nLine two."
        file_path = tmp_path / "stable.txt"
        file_path.write_bytes(content.encode("utf-8"))

        parser = PlainTextParser()
        doc1 = parser.parse(file_path)
        doc2 = parser.parse(file_path)

        assert doc1.metadata.document_id == doc2.metadata.document_id
        assert doc1.metadata.content_sha256 == doc2.metadata.content_sha256
        assert [e.element_id for e in doc1.elements] == [e.element_id for e in doc2.elements]

    def test_different_content_produces_different_identities(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_bytes(b"Alpha content.")
        file_b.write_bytes(b"Beta content.")

        parser = PlainTextParser()
        doc_a = parser.parse(file_a)
        doc_b = parser.parse(file_b)

        assert doc_a.metadata.content_sha256 != doc_b.metadata.content_sha256
        assert doc_a.metadata.document_id != doc_b.metadata.document_id
        assert doc_a.elements[0].element_id != doc_b.elements[0].element_id

    def test_crlf_line_endings_handling(self, tmp_path: Path) -> None:
        content_lf = b"First line.\nSecond line.\n\nThird line."
        content_crlf = b"First line.\r\nSecond line.\r\n\r\nThird line."

        file_lf = tmp_path / "lf.txt"
        file_crlf = tmp_path / "crlf.txt"
        file_lf.write_bytes(content_lf)
        file_crlf.write_bytes(content_crlf)

        parser = PlainTextParser()
        doc_lf = parser.parse(file_lf)
        doc_crlf = parser.parse(file_crlf)

        # Both produce 2 elements with matching trimmed content
        assert len(doc_lf.elements) == len(doc_crlf.elements) == 2
        assert doc_lf.elements[0].content == doc_crlf.elements[0].content
        assert doc_lf.elements[1].content == doc_crlf.elements[1].content

    def test_missing_file_raises_corrupted_file_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "non_existent.txt"
        parser = PlainTextParser()

        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse(missing)
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_invalid_utf8_raises_corrupted_file_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "invalid_utf8.txt"
        # 0xff is not a valid UTF-8 start byte
        bad_file.write_bytes(b"Valid text\xff\xfeInvalid bytes")

        parser = PlainTextParser()
        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse(bad_file)
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


class TestPublicExports:
    """Verify exports in core.parsers package."""

    def test_core_parsers_exports(self) -> None:
        from core.parsers import DocumentParser, ParserRegistry, PlainTextParser

        assert DocumentParser is not None
        assert ParserRegistry is not None
        assert PlainTextParser is not None
