"""Query feature extraction for routing decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class QueryFeatures:
    """Production-observable query features for routing."""

    # Basic metrics
    token_count: int
    char_count: int

    # Lexical patterns
    contains_number: bool
    contains_version: bool
    contains_hex: bool
    contains_uuid: bool
    contains_hash: bool
    contains_path: bool
    contains_quotes: bool
    contains_error_code: bool
    contains_comparison: bool
    contains_table_terms: bool
    contains_source_terms: bool
    contains_code_pattern: bool
    contains_acronym: bool
    contains_technical_term: bool
    ends_with_question: bool
    starts_with_wh: bool
    technical_token_ratio: float

    # Technical terms for detection
    _TECHNICAL_TERMS: ClassVar[frozenset[str]] = frozenset(
        {
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
            "yaml",
            "toml",
            "html",
            "css",
            "javascript",
            "python",
            "java",
            "go",
            "rust",
            "cpp",
            "c++",
            "c#",
            "f#",
            "regex",
            "nosql",
            "mongodb",
            "postgres",
            "redis",
            "kafka",
            "rabbitmq",
            "docker",
            "kubernetes",
            "helm",
            "terraform",
            "ansible",
            "git",
            "github",
            "gitlab",
            "ci",
            "cd",
            "ci/cd",
            "devops",
            "sre",
            "microservice",
            "monolith",
            "serverless",
            "lambda",
            "function",
            "container",
            "pod",
            "service",
            "deployment",
            "ingress",
            "egress",
            "dns",
            "tcp",
            "udp",
            "ip",
            "vpn",
            "proxy",
            "load",
            "balance",
            "cache",
            "memcached",
            "cdn",
            "websocket",
            "graphql",
            "soap",
            "protobuf",
            "avro",
            "parquet",
            "orc",
            "csv",
            "tsv",
            "unicode",
            "utf-32",
            "iso-8859",
            "w3c",
            "ietf",
            "wg",
            "wg21",
            "cpp11",
            "cpp14",
            "cpp17",
            "cpp20",
            "cpp23",
            "c99",
            "c11",
            "c18",
            "es5",
            "es6",
            "es2015",
            "es2016",
            "es2017",
            "es2018",
            "es2019",
            "es2020",
            "es2021",
            "es2022",
            "es2023",
            "es2024",
            "node",
            "npm",
            "yarn",
            "pnpm",
            "webpack",
            "vite",
            "rollup",
            "esbuild",
            "swc",
            "babel",
            "typescript",
            "ts",
            "js",
            "jsx",
            "tsx",
            "vue",
            "react",
            "angular",
            "svelte",
            "solid",
            "preact",
            "lit",
            "django",
            "flask",
            "fastapi",
            "starlette",
            "quart",
            "sanic",
            "tornado",
            "aiohttp",
            "httpx",
            "requests",
            "urllib",
            "curl",
            "wget",
            "ssh",
            "scp",
            "rsync",
            "tar",
            "gzip",
            "zip",
            "bzip2",
            "xz",
            "lz4",
            "zstd",
            "brotli",
            "deflate",
            "inflate",
            "compress",
            "encrypt",
            "decrypt",
            "sign",
            "verify",
            "hash",
            "hmac",
            "pbkdf2",
            "bcrypt",
            "scrypt",
            "argon2",
            "sha1",
            "blake2",
            "blake3",
            "ripemd",
            "whirlpool",
            "crc32",
            "adler32",
            "crc64",
            "xxhash",
            "murmurhash",
            "farmhash",
            "cityhash",
            "spookyhash",
            "metrohash",
            "highwayhash",
            "seahash",
        }
    )

    @classmethod
    def from_query(cls, query: str) -> QueryFeatures:
        """Extract features from a query string."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()
        tokens = query.split()
        query_lower = query.lower()

        # Basic metrics
        len(tokens)
        len(query)

        # Lexical patterns
        bool(re.search(r"\d", query))
        bool(re.search(r"\bv?\d+\.\d+", query))
        bool(re.search(r"0x[0-9a-fA-F]+", query))
        bool(
            re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                query,
                re.IGNORECASE,
            )
        )
        bool(re.search(r"\b[0-9a-f]{32,}\b", query, re.IGNORECASE))
        bool(re.search(r"[./\\]", query))
        bool(re.search(r'["\']', query))
        bool(
            re.search(
                r"\b(error|exception|fail|timeout|404|500|502|503|errno)\b", query, re.IGNORECASE
            )
        )
        bool(
            re.search(
                r"\b(vs|versus|compare|difference|better|worse|faster|slower)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(table|column|row|schema|database|query|select|join|index)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(source|provenance|document|page|section|paragraph|line)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(def|class|function|const|let|var|import|export|return|if|else|for|while)\b",
                query,
            )
        )
        bool(re.search(r"\b[A-Z]{2,}\b", query))
        query.strip().endswith("?")
        bool(re.match(r"^(what|who|where|when|why|how|which)\b", query, re.IGNORECASE))

        # Technical term detection
        query_lower = query.lower()
        tech_count = sum(1 for term in QueryFeatures._TECHNICAL_TERMS if term in query_lower)
        tech_count / max(len(query.split()), 1)

        # Other features
        bool(re.search(r"\d", query))
        bool(re.search(r"\bv?\d+\.\d+", query))
        bool(re.search(r"0x[0-9a-fA-F]+", query))
        bool(
            re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                query,
                re.IGNORECASE,
            )
        )
        bool(re.search(r"\b[0-9a-f]{32,}\b", query, re.IGNORECASE))
        bool(re.search(r"[./\\]", query))
        bool(re.search(r'["\']', query))
        bool(
            re.search(
                r"\b(error|exception|fail|timeout|404|500|502|503|errno)\b", query, re.IGNORECASE
            )
        )
        bool(
            re.search(
                r"\b(vs|versus|compare|difference|better|worse|faster|slower)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(table|column|row|schema|database|query|select|join|index)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(source|provenance|document|page|section|paragraph|line)\b",
                query,
                re.IGNORECASE,
            )
        )
        bool(
            re.search(
                r"\b(def|class|function|const|let|var|import|export|return|if|else|for|while)\b",
                query,
            )
        )
        bool(re.search(r"\b[A-Z]{2,}\b", query))
        query.strip().endswith("?")
        bool(re.match(r"^(what|who|where|when|why|how|which)\b", query, re.IGNORECASE))
        sum(1 for term in QueryFeatures._TECHNICAL_TERMS if term in query_lower) / max(
            len(query.split()), 1
        )

        return QueryFeatures(
            token_count=len(query.split()),
            char_count=len(query),
            contains_number=bool(re.search(r"\d", query)),
            contains_version=bool(re.search(r"\bv?\d+\.\d+", query)),
            contains_hex=bool(re.search(r"0x[0-9a-fA-F]+", query)),
            contains_uuid=bool(
                re.search(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    query,
                    re.IGNORECASE,
                )
            ),
            contains_hash=bool(re.search(r"\b[0-9a-f]{32,}\b", query, re.IGNORECASE)),
            contains_path=bool(re.search(r"[./\\]", query)),
            contains_quotes=bool(re.search(r'["\']', query)),
            contains_error_code=bool(
                re.search(
                    r"\b(error|exception|fail|timeout|404|500|502|503|errno)\b",
                    query,
                    re.IGNORECASE,
                )
            ),
            contains_comparison=bool(
                re.search(
                    r"\b(vs|versus|compare|difference|better|worse|faster|slower)\b",
                    query,
                    re.IGNORECASE,
                )
            ),
            contains_table_terms=bool(
                re.search(
                    r"\b(table|column|row|schema|database|query|select|join|index)\b",
                    query,
                    re.IGNORECASE,
                )
            ),
            contains_source_terms=bool(
                re.search(
                    r"\b(source|provenance|document|page|section|paragraph|line)\b",
                    query,
                    re.IGNORECASE,
                )
            ),
            contains_code_pattern=bool(
                re.search(
                    r"\b(def|class|function|const|let|var|import|export|return|if|else|for|while)\b",
                    query,
                )
            ),
            contains_acronym=bool(re.search(r"\b[A-Z]{2,}\b", query)),
            contains_technical_term=any(
                term in query_lower
                for term in [
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
                    "yaml",
                    "toml",
                    "html",
                    "css",
                    "javascript",
                ]
            ),
            ends_with_question=query.strip().endswith("?"),
            starts_with_wh=bool(
                re.match(r"^(what|who|where|when|why|how|which)\b", query, re.IGNORECASE)
            ),
            technical_token_ratio=sum(
                1 for term in QueryFeatures._TECHNICAL_TERMS if term in query_lower
            )
            / max(len(query.split()), 1),
        )


def extract_features(query: str) -> dict[str, Any]:
    """Extract features as a dictionary for easier manipulation."""
    features = QueryFeatures.from_query(query)
    return {
        "token_count": features.token_count,
        "char_count": features.char_count,
        "contains_number": features.contains_number,
        "contains_version": features.contains_version,
        "contains_hex": features.contains_hex,
        "contains_uuid": features.contains_uuid,
        "contains_hash": features.contains_hash,
        "contains_path": features.contains_path,
        "contains_quotes": features.contains_quotes,
        "contains_error_code": features.contains_error_code,
        "contains_comparison": features.contains_comparison,
        "contains_table_terms": features.contains_table_terms,
        "contains_source_terms": features.contains_source_terms,
        "contains_code_pattern": features.contains_code_pattern,
        "contains_acronym": features.contains_acronym,
        "contains_technical_term": features.contains_technical_term,
        "ends_with_question": features.ends_with_question,
        "starts_with_wh": features.starts_with_wh,
        "technical_token_ratio": features.technical_token_ratio,
    }
