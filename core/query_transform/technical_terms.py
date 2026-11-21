"""Technical term extraction for query expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.query_transform.base import QuerySet

# Patterns for high-value technical terms
_TECH_TERM_PATTERNS = [
    # RFC and standard references
    (r"RFC\s*\d+", "rfc"),
    (r"BCP\s*\d+", "bcp"),
    (r"STD\s*\d+", "std"),
    (r"ECMA-\d+", "ecma"),
    (r"ISO[-\s]?\d+", "iso"),
    # Encoding/format identifiers
    (r"UTF[-\s]?\d+", "utf"),
    (r"ASCII", "ascii"),
    (r"Unicode", "unicode"),
    (r"BOM", "bom"),
    # Protocol/format names
    (r"JSON", "json"),
    (r"XML", "xml"),
    (r"YAML", "yaml"),
    (r"TOML", "toml"),
    (r"HTML", "html"),
    (r"HTTP", "http"),
    (r"HTTPS", "https"),
    (r"TLS", "tls"),
    (r"MIME", "mime"),
    (r"ABNF", "abnf"),
    # Technical acronyms
    (r"API", "api"),
    (r"CPU", "cpu"),
    (r"GPU", "gpu"),
    (r"RAM", "ram"),
    (r"SSD", "ssd"),
    (r"NVMe", "nvme"),
    (r"SQL", "sql"),
    (r"BOM", "bom"),
    (r"URI", "uri"),
    (r"URL", "url"),
    # Hex/percent notation
    (r"%[xX][0-9a-fA-F]{2}", "hex_code"),
    # Quoted technical terms
    (r"'[A-Za-z0-9_/-]+'", "quoted"),
    (r'"[A-Za-z0-9_/-]+"', "quoted"),
]


@dataclass
class TechnicalTerms:
    """Extracted technical terms from a query."""

    terms: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]

    def all_terms(self) -> tuple[str, ...]:
        return self.terms


def extract_technical_terms(query: str) -> TechnicalTerms:
    """Extract high-value technical terms from a query."""
    if not query or not query.strip():
        return TechnicalTerms(terms=(), categories={})

    query.lower()
    found_terms: list[str] = []
    categories: dict[str, list[str]] = {}

    for pattern, category in _TECH_TERM_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            for match in matches:
                term = match.strip().strip("'\"")
                if term and term.lower() not in (t.lower() for t in found_terms):
                    found_terms.append(term)
                    categories.setdefault(category, []).append(term)

    # Also extract capitalized words (potential proper nouns/acronyms)
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9_/-]{2,}\b", query)
    for term in capitalized:
        if term.lower() not in (t.lower() for t in found_terms):
            found_terms.append(term)
            categories.setdefault("capitalized", []).append(term)

    # Extract numbers that look like references (section numbers, versions, etc.)
    numbers = re.findall(r"\b\d+(?:\.\d+)*\b", query)
    for num in numbers:
        if num not in found_terms:
            found_terms.append(num)
            categories.setdefault("numeric", []).append(num)

    return TechnicalTerms(
        terms=tuple(found_terms),
        categories={k: tuple(v) for k, v in categories.items()},
    )


class TechnicalTermExtractor:
    """Extracts technical terms from queries for lexical query expansion."""

    @property
    def name(self) -> str:
        return "technical_terms"

    def transform(self, query: str) -> QuerySet:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        terms = extract_technical_terms(query)
        if not terms.terms:
            return QuerySet(
                original=query.strip(),
                transformed=(),
                strategy="technical_terms",
                metadata={"extracted_terms": 0},
            )

        # Create expanded query: original + extracted terms
        expanded = query.strip() + " " + " ".join(terms.terms)

        return QuerySet(
            original=query.strip(),
            transformed=(expanded,),
            strategy="technical_terms",
            metadata={
                "extracted_terms": len(terms.terms),
                "terms": terms.terms,
                "categories": {k: list(v) for k, v in terms.categories.items()},
            },
        )
