"""Canonical document, element, provenance, and chunk schemas.

This module defines the foundational Pydantic models for document representation
throughout the RAG_EVAL pipeline:
- DocumentMetadata: Source file identity, hashes, sizes, and timestamps.
- BoundingBox & PageLocation: Normalized spatial coordinates and page provenance.
- ExtractedElement: Granular content units (text, tables, headings, images) with ordering.
- NormalizedDocument: Canonical representation of an ingested document.
- Chunk: Discrete text segments derived from a NormalizedDocument for embedding/retrieval.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_HEX_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class ElementType(StrEnum):
    """Semantic type of an extracted document element."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Metadata describing a source document and its ingestion state."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        min_length=1,
        description="Stable canonical identifier for this document.",
    )
    filename: str = Field(
        ...,
        min_length=1,
        description="Original source filename (e.g., 'report_2024.pdf').",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type if known (e.g., 'application/pdf').",
    )
    content_sha256: str | None = Field(
        default=None,
        description="64-character hex-encoded SHA-256 checksum of original file bytes.",
    )
    byte_size: int | None = Field(
        default=None,
        ge=0,
        description="Size of source file in bytes; must be non-negative.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Original creation timestamp of document, must be timezone-aware.",
    )
    processed_at: datetime | None = Field(
        default=None,
        description="Timestamp when document was parsed/ingested, must be timezone-aware.",
    )
    source_uri: str | None = Field(
        default=None,
        description="URI or storage path of original source file.",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value metadata from user or pipeline.",
    )

    @field_validator("document_id", "filename")
    @classmethod
    def validate_non_whitespace_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("String identifier cannot be empty or whitespace-only.")
        return v

    @field_validator("content_sha256")
    @classmethod
    def validate_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _SHA256_HEX_PATTERN.match(v):
            raise ValueError(
                f"content_sha256 must be a 64-character hexadecimal string, got: {v!r}"
            )
        return v.lower()

    @field_validator("created_at", "processed_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("Datetime must be timezone-aware (tzinfo cannot be None).")
        return v


class BoundingBox(BaseModel):
    """Normalized spatial coordinates on a page in the range [0.0, 1.0].

    The canonical coordinate frame assumes:
    - Origin (0.0, 0.0) at top-left.
    - x increases horizontally to the right: 0.0 <= x0 <= x1 <= 1.0.
    - y increases vertically downwards: 0.0 <= y0 <= y1 <= 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    x0: float = Field(..., ge=0.0, le=1.0, description="Normalized left edge.")
    y0: float = Field(..., ge=0.0, le=1.0, description="Normalized top edge.")
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized right edge.")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom edge.")

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.x0 > self.x1:
            raise ValueError(f"Invalid bounding box: x0 ({self.x0}) cannot exceed x1 ({self.x1}).")
        if self.y0 > self.y1:
            raise ValueError(f"Invalid bounding box: y0 ({self.y0}) cannot exceed y1 ({self.y1}).")
        return self

    def to_absolute(
        self, page_width: float, page_height: float
    ) -> tuple[float, float, float, float]:
        """Convert normalized [0.0, 1.0] bounds to absolute coordinates.

        Returns (abs_x0, abs_y0, abs_x1, abs_y1).
        """
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Page dimensions must be strictly positive.")
        return (
            self.x0 * page_width,
            self.y0 * page_height,
            self.x1 * page_width,
            self.y1 * page_height,
        )


class PageLocation(BaseModel):
    """Spatial and page provenance for an extracted element or chunk."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number within the document.",
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Optional normalized bounding box on the page.",
    )
    coordinate_system: str = Field(
        default="normalized_top_left",
        description="Coordinate space convention (default: normalized [0,1] top-left).",
    )
    page_width: float | None = Field(
        default=None,
        gt=0,
        description="Original page width (in points/pixels) to facilitate absolute conversion.",
    )
    page_height: float | None = Field(
        default=None,
        gt=0,
        description="Original page height (in points/pixels) to facilitate absolute conversion.",
    )


class TableData(BaseModel):
    """Minimal JSON-compatible representation of extracted tabular data.

    Limitations:
    - Assumes a simple 2D rectangular grid of cell text.
    - Does not represent merged cells (colspan/rowspan), complex hierarchies,
      or embedded cell styling.
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[list[str]] = Field(
        ...,
        description="2D array of cell string values (row-major).",
    )
    headers: list[str] | None = Field(
        default=None,
        description="Optional list of column header labels.",
    )
    caption: str | None = Field(
        default=None,
        description="Optional title or caption associated with the table.",
    )


class ExtractedElement(BaseModel):
    """An atomic content element extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier within the scope of this document.",
    )
    element_type: ElementType = Field(
        ...,
        description="Semantic category of the extracted element.",
    )
    content: str = Field(
        ...,
        description="Raw or plain-text string representation of the element.",
    )
    order_index: int = Field(
        ...,
        ge=0,
        description="0-indexed sequence position indicating reading order in source.",
    )
    location: PageLocation | None = Field(
        default=None,
        description="Optional spatial provenance and page position.",
    )
    table_data: TableData | None = Field(
        default=None,
        description="Structured table representation if element_type is TABLE.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser-specific attributes or optical confidence metrics.",
    )

    @field_validator("element_id")
    @classmethod
    def validate_element_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("element_id cannot be whitespace-only.")
        return v


class NormalizedDocument(BaseModel):
    """Canonical representation of an ingested document."""

    model_config = ConfigDict(extra="forbid")

    metadata: DocumentMetadata = Field(
        ...,
        description="Document-level metadata and source provenance.",
    )
    elements: list[ExtractedElement] = Field(
        default_factory=list,
        description="Ordered sequence of extracted content elements.",
    )
    page_count: int | None = Field(
        default=None,
        ge=1,
        description="Total number of pages if known.",
    )

    @model_validator(mode="after")
    def validate_document_structure(self) -> Self:
        seen_ids: set[str] = set()
        for elem in self.elements:
            if elem.element_id in seen_ids:
                raise ValueError(f"Duplicate element_id detected in document: {elem.element_id!r}")
            seen_ids.add(elem.element_id)
        return self


class Chunk(BaseModel):
    """Discrete text segment derived from a NormalizedDocument for embedding/retrieval.

    Chunks do not store embedding vectors or retrieval scores; those belong to
    vector-index or retrieval-candidate models.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Stable unique identifier for this chunk.",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the parent NormalizedDocument.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Cleaned text content of the chunk.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="0-indexed position of chunk within the document sequence.",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Computed token count if known; must be non-negative.",
    )
    source_element_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of source ExtractedElements contributing to this chunk.",
    )
    locations: list[PageLocation] = Field(
        default_factory=list,
        description="Spatial provenance collected from contributing elements.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-level metadata (e.g. chunking strategy, section path).",
    )

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID cannot be empty or whitespace-only.")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only.")
        return v
