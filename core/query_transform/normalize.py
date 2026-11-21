"""Deterministic query normalization transformer."""

from __future__ import annotations

import re
import unicodedata

from core.query_transform.base import QuerySet

# Patterns that should be preserved as-is (technical identifiers)
_TECH_PATTERNS = [
    r"RFC\s*\d+",  # RFC 8259, RFC8259
    r"UTF-\d+",  # UTF-8, UTF-16
    r"JSON",  # JSON
    r"ABNF",  # ABNF
    r"MIME",  # MIME
    r"HTTP",  # HTTP, HTTPS
    r"TLS",  # TLS
    r"SHA-\d+",  # SHA-256
    r"ECMA-\d+",  # ECMA-404
    r"ISO-\d+",  # ISO-8601
    r"BOM",  # BOM
    r"URI",  # URI, URL
    r"API",  # API
    r"CPU",  # CPU
    r"GPU",  # GPU
    r"RAM",  # RAM
    r"SSD",  # SSD
    r"NVMe",  # NVMe
    r"SQL",  # SQL
    r"HTML",  # HTML
    r"XML",  # XML
    r"YAML",  # YAML
    r"TOML",  # TOML
    r"Docker",  # Docker
    r"Kubernetes",  # Kubernetes
    r"Prometheus",  # Prometheus
    r"OpenTelemetry",  # OpenTelemetry
    r"W3C",  # W3C
    r"IETF",  # IETF
    r"IANA",  # IANA
    r"BCP\s*\d+",  # BCP 178
    r"STD\s*\d+",  # STD 90
    r"OTLP",  # OTLP
    r"gRPC",  # gRPC
    r"%\w+",  # %x20, %x09, etc.
    r"\\[[^\]]+\\]",  # [S1], [S2] citation labels
]

_TECH_RE = re.compile("|".join(f"({p})" for p in _TECH_PATTERNS), re.IGNORECASE)


def _normalize_text(text: str) -> str:
    """Normalize text while preserving technical identifiers."""
    if not text:
        return text

    # Protect technical terms
    protected: list[str] = []
    placeholder_map: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        term = match.group(0)
        placeholder = f"__TECH_{len(protected)}__"
        protected.append(term)
        placeholder_map[placeholder] = term
        return placeholder

    protected_text = _TECH_RE.sub(protect, text)

    # Normalize: Unicode NFKC, collapse whitespace
    normalized = unicodedata.normalize("NFKC", protected_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Restore technical terms
    for placeholder, term in placeholder_map.items():
        normalized = normalized.replace(placeholder, term)

    return normalized


class NormalizeTransformer:
    """Applies deterministic normalization to queries while preserving technical terms."""

    @property
    def name(self) -> str:
        return "normalize"

    def transform(self, query: str) -> QuerySet:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        normalized = _normalize_text(query)
        if normalized == query.strip():
            # No change needed
            return QuerySet(
                original=query.strip(),
                transformed=(),
                strategy="normalize",
                metadata={"changed": False},
            )

        return QuerySet(
            original=query.strip(),
            transformed=(normalized,),
            strategy="normalize",
            metadata={"changed": True},
        )
