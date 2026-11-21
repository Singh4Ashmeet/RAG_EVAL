"""Document parser interfaces and implementations."""

from core.parsers.base import DocumentParser
from core.parsers.docx_parser import DOCXParser
from core.parsers.image_parser import ImageParser
from core.parsers.pdf_scanned import ScannedPDFParser
from core.parsers.pdf_text import PDFParser
from core.parsers.registry import ParserRegistry
from core.parsers.tables import serialize_table
from core.parsers.text_parser import PlainTextParser

__all__ = [
    "DOCXParser",
    "DocumentParser",
    "ImageParser",
    "PDFParser",
    "ParserRegistry",
    "PlainTextParser",
    "ScannedPDFParser",
    "serialize_table",
]
