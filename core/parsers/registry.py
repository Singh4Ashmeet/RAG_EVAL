"""Parser registry for routing files to matching DocumentParser instances."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from core.errors import UnsupportedFormatError
from core.parsers.base import DocumentParser


class ParserRegistry:
    """Synchronous registry mapping file extensions to DocumentParser implementations."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}

    def register(
        self,
        extensions: str | Iterable[str],
        parser: DocumentParser,
    ) -> None:
        """Register a parser for one or more file extensions.

        Args:
            extensions: Single extension (e.g. '.txt' or 'md') or an iterable of extensions.
            parser: Concrete instance implementing DocumentParser protocol.
        """
        ext_list = [extensions] if isinstance(extensions, str) else list(extensions)
        for ext in ext_list:
            normalized = self._normalize_extension(ext)
            if normalized:
                self._parsers[normalized] = parser

    def get_parser(self, file_path: Path | str) -> DocumentParser:
        """Retrieve the registered parser for the given file's extension.

        Args:
            file_path: Filesystem path or filename to resolve.

        Returns:
            Registered DocumentParser instance.

        Raises:
            UnsupportedFormatError: If the extension is not registered or file has no extension.
        """
        path = Path(file_path)
        ext = self._normalize_extension(path.suffix)
        if not ext or ext not in self._parsers:
            display_ext = path.suffix if path.suffix else "<none>"
            raise UnsupportedFormatError(
                f"Unsupported file format '{display_ext}' for file '{path.name}'."
            )
        return self._parsers[ext]

    def supported_extensions(self) -> set[str]:
        """Return the set of normalized registered extensions."""
        return set(self._parsers.keys())

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """Normalize extension to lowercase with leading dot (e.g. 'TXT' -> '.txt')."""
        cleaned = extension.strip().lower()
        if not cleaned:
            return ""
        if not cleaned.startswith("."):
            return f".{cleaned}"
        return cleaned
