"""Base parser interfaces and protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from core.normalize.schema import NormalizedDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for document parsers converting source files into NormalizedDocuments."""

    def parse(self, file_path: Path) -> NormalizedDocument:
        """Parse a source document from the filesystem into a canonical NormalizedDocument.

        Args:
            file_path: Absolute or relative filesystem Path to the source file.

        Returns:
            A canonical NormalizedDocument containing metadata and extracted elements.
        """
        ...
