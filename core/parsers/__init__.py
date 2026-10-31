"""Document parser interfaces and implementations."""

from core.parsers.base import DocumentParser
from core.parsers.registry import ParserRegistry
from core.parsers.text_parser import PlainTextParser

__all__ = [
    "DocumentParser",
    "ParserRegistry",
    "PlainTextParser",
]
