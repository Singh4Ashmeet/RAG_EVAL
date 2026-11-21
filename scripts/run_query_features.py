"""Analyze query features for routing decisions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure project root is in sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.datasets.loader import load_jsonl_dataset
from eval.datasets.schema import CorpusTrack


def extract_features(query: str) -> dict:
    """Extract production-observable features from a query."""
    query = query.strip()
    tokens = query.split()

    features = {
        "token_count": len(tokens),
        "char_count": len(query),
        "contains_number": bool(re.search(r"\d", query)),
        "contains_version": bool(re.search(r"\bv?\d+\.\d+", query)),
        "contains_number_range": bool(re.search(r"\d+\.\.\d+", query)),
        "contains_hex": bool(re.search(r"0x[0-9a-fA-F]+", query)),
        "contains_uuid": bool(
            re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                query,
                re.IGNORECASE,
            )
        ),
        "contains_hash": bool(re.search(r"\b[0-9a-f]{32,}\b", query, re.IGNORECASE)),
        "contains_path": bool(re.search(r"[./\\]", query)),
        "contains_quotes": bool(re.search(r'["\']', query)),
        "contains_error_code": bool(
            re.search(r"\b(error|exception|fail|timeout|404|500|502|503)\b", query, re.IGNORECASE)
        ),
        "contains_comparison": bool(
            re.search(
                r"\b(vs|versus|compare|difference|better|worse|faster|slower)\b",
                query,
                re.IGNORECASE,
            )
        ),
        "contains_table_terms": bool(
            re.search(
                r"\b(table|column|row|schema|database|query|select|join)\b", query, re.IGNORECASE
            )
        ),
        "contains_source_terms": bool(
            re.search(
                r"\b(source|provenance|document|page|section|paragraph)\b", query, re.IGNORECASE
            )
        ),
        "contains_how_to": bool(
            re.search(r"\b(how to|how do|how can|steps to|guide)\b", query, re.IGNORECASE)
        ),
        "contains_what_is": bool(
            re.search(r"\b(what is|what are|define|definition)\b", query, re.IGNORECASE)
        ),
        "contains_why": bool(re.search(r"\b(why|reason|cause)\b", query, re.IGNORECASE)),
        "contains_abbreviation": bool(re.search(r"\b[A-Z]{2,}\b", query)),
        "contains_technical_term": bool(
            re.search(
                r"\b(json|xml|http|api|rest|grpc|sql|utf|ascii|utf-8|utf-16|rfc|iso|ecma|iana|mime|base64|sha|md5|sha256|sha512|rsa|aes|ssl|tls|oauth|jwt|jwt)\b",
                query,
                re.IGNORECASE,
            )
        ),
        "ends_with_question": query.strip().endswith("?"),
        "starts_with_wh": bool(
            re.match(r"^(what|who|where|when|why|how|which)\b", query, re.IGNORECASE)
        ),
    }

    # Technical token ratio
    technical_terms = [
        "json",
        "xml",
        "http",
        "api",
        "rest",
        "grpc",
        "sql",
        "utf",
        "ascii",
        "utf-8",
        "utf-16",
        "rfc",
        "iso",
        "ecma",
        "iana",
        "mime",
        "base64",
        "sha",
        "md5",
        "sha256",
        "sha512",
        "rsa",
        "aes",
        "ssl",
        "tls",
        "oauth",
        "jwt",
        "rfc",
        "json",
        "xml",
        "yaml",
        "toml",
    ]
    tech_count = sum(1 for t in technical_terms if t.lower() in query.lower())
    features["technical_token_ratio"] = tech_count / max(len(tokens), 1)

    return features


def analyze_query(query: str) -> str:
    """Classify query type based on features."""
    features = extract_features(query)

    # Heuristic classification
    if (
        features["contains_error_code"]
        or features["contains_version"]
        or features["contains_number_range"]
        or features["contains_hex"]
        or features["contains_uuid"]
        or features["contains_hash"]
    ):
        return "LEXICAL_EXACT"

    if features["contains_number"]:
        return "NUMERICAL"

    if features["contains_table_terms"]:
        return "TABLE_LOOKUP"

    if features["contains_source_terms"]:
        return "PROVENANCE"

    if features["contains_technical_term"] > 0:
        return "TECHNICAL_IDENTIFIER"

    if features["contains_comparison"]:
        return "COMPARISON"

    if features["ends_with_question"] and (
        features["starts_with_wh"]
        or features["contains_what_is"]
        or features["contains_how_to"]
        or features["contains_why"]
    ):
        return "SEMANTIC"

    return "AMBIGUOUS"


def main():
    parser = argparse.ArgumentParser(description="Analyze query features for routing.")
    parser.add_argument(
        "--track", type=str, default="track_b", choices=["track_a", "track_b", "track_c"]
    )
    args = parser.parse_args()

    track = CorpusTrack(args.track)
    dataset = load_jsonl_dataset(f"eval/golden/{track.value}.jsonl")

    print(f"=== Query Feature Analysis: {track.value.upper()} ===\n")
    print(
        f"{'Query ID':<20} {'Type':<20} {'Tokens':<8} {'Chars':<8} {'HasNum':<6} {'HasVer':<6} {'HasCode':<6} {'HasPath':<6} {'HasQuote':<6} {'HasErr':<6} {'HasComp':<6} {'HasTable':<6} {'HasSource':<6} {'TechRatio':<8} {'Type':<20}"
    )
    print("-" * 160)

    type_counts = {}

    for case in dataset.cases:
        query = case.query
        features = extract_features(query)
        qtype = analyze_query(query)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        print(
            f"{case.query_id:<20} {qtype:<20} {features['token_count']:<8} {features['char_count']:<8} "
            f"{str(features['contains_number']):<6} {str(features['contains_version']):<6} "
            f"{str(bool(re.search(r'\b[A-Z]{2,}\b', case.query))):<6} {str(features['contains_path']):<6} "
            f"{str(features['contains_quotes']):<6} {str(features['contains_error_code']):<6} "
            f"{str(features['contains_comparison']):<6} {str(features['contains_table_terms']):<6} "
            f"{str(features['contains_source_terms']):<6} {features['technical_token_ratio']:.2f} "
            f"{analyze_query(case.query):<20}"
        )

    print("\n=== Type Distribution ===")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {qtype:<20}: {count}")


if __name__ == "__main__":
    sys.exit(main())
