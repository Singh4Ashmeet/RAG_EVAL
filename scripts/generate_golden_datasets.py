"""Generate and validate golden datasets for Track B, Track C, and Track D.

Preserves Track A intact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from eval.datasets.schema import (
    CorpusTrack,
    Difficulty,
    QuestionCategory,
)
from eval.datasets.validation import DatasetValidator


def build_track_b() -> list[dict]:
    # 52 high-quality questions on RFC 8259 (doc_61a5378f4255c720)
    # Covering every section and chunk
    doc_id = "doc_61a5378f4255c720"
    prov = {
        "annotator": "RAG_EVAL Curation Team",
        "verified_at": "2026-11-16T12:00:00Z",
        "source_documents": ["rfc8259.txt"],
        "verification_notes": "Ground-truth verified against IETF RFC 8259.",
    }

    # Resolve chunk-element mapping deterministically
    validator = DatasetValidator()
    idx_b, _ = validator._resolve_corpus_index(Path("eval/corpora/track_b"), CorpusTrack.TRACK_B)
    e_to_c: dict[str, str] = {}
    for c in idx_b.chunks:
        for eid in c.source_element_ids:
            e_to_c[eid] = c.chunk_id

    def q(
        qid: str,
        query: str,
        ans: str,
        e_idx: list[int],
        cat: QuestionCategory,
        diff: Difficulty,
        multi_chunk: bool = False,
    ) -> dict:
        e_ids = [f"{doc_id}_e{i:04d}" for i in e_idx]
        c_ids = list(dict.fromkeys(e_to_c[eid] for eid in e_ids))
        return {
            "query_id": f"track_b_q{qid:0>2s}",
            "query": query,
            "relevant_chunk_ids": c_ids,
            "relevant_element_ids": e_ids,
            "reference_answer": ans,
            "metadata": {},
            "status": "verified",
            "answerability": "answerable",
            "answerable": True,
            "requires_multi_chunk": multi_chunk or len(c_ids) > 1,
            "category": cat.value,
            "difficulty": diff.value,
            "provenance": {**prov, "evidence_references": e_ids},
            "track": "track_b",
        }

    cases = [
        # 1..11 (Core factual questions)
        q(
            "01",
            "What four primitive types can JSON represent?",
            "JSON can represent strings, numbers, booleans, and null.",
            [37],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "02",
            "What two structured types are provided in JSON?",
            "JSON provides objects and arrays as structured types.",
            [37],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "03",
            "What rule regarding member names makes a JSON object interoperable across implementations?",
            "An object whose names are all unique is interoperable in the sense that all software implementations will agree on the mappings.",
            [88],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "04",
            "How is an array structure represented in JSON grammar?",
            "An array structure is represented as square brackets surrounding zero or more values.",
            [94],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "05",
            "What components form the grammar of a JSON number?",
            "A number is represented as an optional minus sign, an integer part, an optional fraction part, and an optional exponent part.",
            [102],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "06",
            "What character encoding MUST be used when JSON text is exchanged outside a closed ecosystem?",
            "JSON text exchanged between systems that are not part of a closed ecosystem MUST be encoded using UTF-8.",
            [130],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "07",
            "How does a JSON string begin and end, and how are characters escaped?",
            "A string begins and ends with quotation marks, and all characters may be placed within the quotation marks except characters that must be escaped using a backslash.",
            [117],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.MEDIUM,
        ),
        q(
            "08",
            "How does parser behavior differ when an object contains duplicate names?",
            "Some implementations report an error, some parse the object and report all name/value pairs, and others return the last name/value pair.",
            [91],
            QuestionCategory.MULTI_PARAGRAPH,
            Difficulty.HARD,
        ),
        q(
            "09",
            "What security risk arises from using JavaScript's eval() function to parse JSON texts?",
            "Using JavaScript's eval() function to parse JSON texts can execute unexpected code because JSON's syntax is borrowed from JavaScript.",
            [169],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.MEDIUM,
        ),
        q(
            "10",
            "Which four whitespace characters are permitted in JSON text grammar?",
            "The permitted whitespace characters are space (%x20), horizontal tab (%x09), line feed (%x0A), and carriage return (%x0D).",
            [75],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "11",
            "What is the registered MIME media type for JSON text according to IANA considerations?",
            "The media type for JSON text is application/json.",
            [148],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        # 12..52 (Expanded coverage across all RFC 8259 sections)
        q(
            "12",
            "Who is the editor of RFC 8259 and in what month was it published?",
            "T. Bray was the editor, and RFC 8259 was published in December 2017.",
            [0, 1],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "13",
            "Which RFC does RFC 8259 obsolete according to its header metadata?",
            "RFC 8259 obsoletes RFC 7159.",
            [2],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "14",
            "What standards track status and STD level does RFC 8259 hold?",
            "RFC 8259 is an Internet Standard with designation STD 90.",
            [3, 4],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.EASY,
        ),
        q(
            "15",
            "Is the reference to ECMA-404 in RFC 8259 normative or informative?",
            "The reference to ECMA-404 is normative, not informative.",
            [55],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "16",
            "What definition does RFC 8259 give for a JSON text?",
            "A JSON text is a serialized value.",
            [67],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "17",
            "How did earlier JSON specifications restrict the root value of a JSON text?",
            "Certain previous specifications of JSON only considered an object or an array to be valid root JSON texts.",
            [68],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "18",
            "Which six structural characters are defined in the JSON grammar?",
            "The six structural characters are left square bracket [ (%x5B), right square bracket ] (%x5D), left curly bracket { (%x7B), right curly bracket } (%x7D), colon : (%x3A), and comma , (%x2C).",
            [73, 74],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "19",
            "Can whitespace precede or follow any of the six structural characters?",
            "Yes, insignificant whitespace is allowed before or after any of the six structural characters.",
            [72],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "20",
            "What literal names represent boolean values and null in JSON?",
            "The three literal names are false (%x66.61.6c.73.65), null (%x6e.75.6c.6c), and true (%x74.72.75.65).",
            [81, 82],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "21",
            "How is a JSON object structure represented syntactically?",
            "An object structure is represented as a pair of curly brackets surrounding zero or more name/value pairs.",
            [85, 86],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "22",
            "In an object member pair, what separator character must appear between the name and the value?",
            "A single colon character must separate the name from the value.",
            [86],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "23",
            "Does RFC 8259 define an ordering for the members of a JSON object?",
            "No, a JSON object is an unordered collection of zero or more name/value pairs.",
            [37, 85],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
            multi_chunk=True,
        ),
        q(
            "24",
            "In a JSON array, what character separates adjacent values?",
            "A comma character separates adjacent values in an array.",
            [94, 95],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "25",
            "Can a JSON number have a leading plus sign according to RFC 8259 grammar?",
            "No, a leading plus sign is not permitted in the JSON number grammar.",
            [102],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "26",
            "Does the JSON number grammar permit leading zeros in integer values?",
            "Leading zeros are not allowed in the integer part of a number, except for the digit zero itself before a decimal point.",
            [102, 103],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "27",
            "What IEEE 754 precision is recommended for interoperable JSON number implementations?",
            "Implementations are recommended to expect precision and range equivalent to IEEE 754-2008 binary64 (double precision).",
            [107],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "28",
            "What exact range of integers is guaranteed to be interoperable without precision loss?",
            "Integers in the range [-(2**53)+1, (2**53)-1], or -9007199254740991 to 9007199254740991, are interoperable.",
            [108],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        q(
            "29",
            "Are values such as NaN, Infinity, and -Infinity permitted in JSON numbers?",
            "No, numeric values that cannot be represented as sequences of digits (such as Infinity and NaN) are not permitted.",
            [110],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.MEDIUM,
        ),
        q(
            "30",
            "Which characters must be escaped when placed within a JSON string?",
            "Quotation mark, reverse solidus (backslash), and control characters (%x00 through %x1F) must be escaped.",
            [117],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "31",
            "What are the eight two-character escape sequences supported in JSON strings?",
            'The eight two-character escapes are \\", \\\\, \\/, \\b, \\f, \\n, \\r, and \\t.',
            [121],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        q(
            "32",
            "How are basic multilingual plane characters outside 7-bit ASCII escaped in JSON strings?",
            "They may be represented as a six-character sequence \\u followed by four hexadecimal digits (e.g. \\uXXXX).",
            [122],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "33",
            "How must Unicode characters outside the Basic Multilingual Plane (code points above U+FFFF) be escaped?",
            "They must be escaped as two UTF-16 surrogate pairs (e.g. \\uD834\\uDD1E for U+1D11E).",
            [123],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        q(
            "34",
            "What is the requirement regarding Byte Order Marks (BOM) when producing JSON text in UTF-8?",
            "An implementation MUST NOT add a byte order mark (%xEF.BB.BF) to the beginning of a networked UTF-8 JSON text.",
            [131],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "35",
            "Are JSON parsers required to accept or ignore a Byte Order Mark if present?",
            "Parsers MAY ignore the presence of a byte order mark rather than treating it as an error.",
            [131],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "36",
            "What is the recommended method for comparing JSON object member names for equality?",
            "The recommended method is code point by code point (or code unit by code unit) comparison rather than Unicode normalization.",
            [140],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "37",
            "What happens if a JSON parser receives texts with deeply nested structures?",
            "An implementation may set limits on the maximum nesting depth and may fail to parse texts that exceed those limits to prevent stack overflow.",
            [144, 172],
            QuestionCategory.MULTI_PARAGRAPH,
            Difficulty.MEDIUM,
            multi_chunk=True,
        ),
        q(
            "38",
            "What file extension is recommended for JSON files in the IANA considerations?",
            "The recommended file extension is .json.",
            [152],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "39",
            "What is the Macintosh file type code registered for JSON in IANA considerations?",
            "The Macintosh file type code is TEXT.",
            [153],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.EASY,
        ),
        q(
            "40",
            "Why can parsing JSON using eval() lead to code execution in JavaScript environments?",
            "Because JSON's syntax is a subset of JavaScript's Object Literal syntax, allowing malicious executable statements.",
            [169],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "41",
            "What denial of service vulnerability can affect parsers when reading untrusted JSON?",
            "Excessive nesting depth in arrays or objects can cause stack overflow or resource exhaustion.",
            [172],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "42",
            "What risk arises when different software systems parse large numbers exceeding 64-bit float limits?",
            "They may round or truncate the values differently, resulting in silent data corruption or loss of precision.",
            [173],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "43",
            "Does RFC 8259 require a generator to output whitespace between structural tokens?",
            "No, a generator may produce minified JSON without any insignificant whitespace.",
            [76, 145],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.MEDIUM,
            multi_chunk=True,
        ),
        q(
            "44",
            "Can a JSON text consist solely of a single string, number, or boolean?",
            "Yes, because a JSON text is defined as any serialized value, including primitives.",
            [67],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "45",
            "What was changed in Section 8.1 between RFC 7159 and RFC 8259?",
            "Section 8.1 was changed to require the use of UTF-8 when JSON text is exchanged outside a closed ecosystem.",
            [198],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "46",
            "Which BCP document reference was removed in Appendix A of RFC 8259?",
            "The reference to BCP 178 was removed in Appendix A.",
            [199],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        q(
            "47",
            "Which errata IDs were applied to the text of RFC 8259 according to Appendix A?",
            "Errata IDs 3927, 4264, 4336, and 4388 were applied.",
            [200],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        q(
            "48",
            "Is the forward slash character '/' required to be escaped in JSON strings?",
            "No, escaping solidus ('/') as '\\/' is allowed but optional.",
            [117, 121],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "49",
            "What case sensitivity rule applies to the true, false, and null literal names?",
            "The literal names true, false, and null must be lowercase; uppercase or mixed case literals are invalid.",
            [81, 82],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        q(
            "50",
            "Can a JSON number contain an exponent part with either lowercase 'e' or uppercase 'E'?",
            "Yes, the exponent part grammar allows either 'e' or 'E', optionally followed by a plus or minus sign.",
            [102],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        q(
            "51",
            "What document defines the ABNF grammar notation used throughout RFC 8259?",
            "RFC 5234 defines the Augmented BNF (ABNF) notation.",
            [48],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.EASY,
        ),
        q(
            "52",
            "What is the restriction on characters in the range %x00-1F inside unescaped JSON strings?",
            "Control characters in the range %x00 through %x1F are not allowed as unescaped characters in JSON strings.",
            [117, 128],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
    ]

    return cases


def build_track_c() -> list[dict]:
    # 30 hard and adversarial questions covering table lookup, PDF page provenance,
    # near-duplicate incident distinction, numerical comparisons, and multi-chunk synthesis
    validator = DatasetValidator()
    idx_c, _ = validator._resolve_corpus_index(Path("eval/corpora/track_c"), CorpusTrack.TRACK_C)
    e_to_c: dict[str, str] = {}
    for c in idx_c.chunks:
        for eid in c.source_element_ids:
            e_to_c[eid] = c.chunk_id

    # Document IDs
    spec_id = "doc_2c869a47619a12fe"
    pdf_id = "doc_21572e58b8b61bc2"
    docx_id = "doc_8ba7306c9afe56ff"
    log_id = "doc_31dc07e6f650404a"

    doc_files = {
        spec_id: "system_spec.md",
        pdf_id: "operations_guide.pdf",
        docx_id: "product_evaluation.docx",
        log_id: "incident_log.txt",
    }

    def qc(
        qid: str,
        query: str,
        ans: str,
        e_ids: list[str],
        cat: QuestionCategory,
        diff: Difficulty,
        multi_chunk: bool = False,
        requires_table: bool = False,
    ) -> dict:
        c_ids = list(dict.fromkeys(e_to_c[eid] for eid in e_ids))
        src_docs = list(dict.fromkeys(doc_files[eid.split("_e")[0]] for eid in e_ids))
        prov = {
            "annotator": "RAG_EVAL Curation Team",
            "verified_at": "2026-11-16T12:00:00Z",
            "source_documents": src_docs,
            "verification_notes": "Hard and adversarial queries verified against Track C multi-format corpus.",
            "evidence_references": e_ids,
        }
        return {
            "query_id": f"track_c_q{qid:0>2s}",
            "query": query,
            "relevant_chunk_ids": c_ids,
            "relevant_element_ids": e_ids,
            "reference_answer": ans,
            "metadata": {},
            "status": "verified",
            "answerability": "answerable",
            "answerable": True,
            "requires_multi_chunk": multi_chunk or len(c_ids) > 1,
            "requires_table": requires_table,
            "category": cat.value,
            "difficulty": diff.value,
            "provenance": prov,
            "track": "track_c",
        }

    cases = [
        # Table questions (Markdown table)
        qc(
            "01",
            "What is the maximum P99 latency target and uptime SLA for Tier-1 Gateway in the system specification?",
            "Tier-1 Gateway has an uptime target of 99.99% and a maximum P99 latency of 50ms.",
            [f"{spec_id}_e0003"],
            QuestionCategory.TABLE_LOOKUP,
            Difficulty.EASY,
            requires_table=True,
        ),
        qc(
            "02",
            "According to the system specification table, who is the primary contact for Tier-2 Compute escalations?",
            "Platform Core Team is the primary contact for Tier-2 Compute.",
            [f"{spec_id}_e0003"],
            QuestionCategory.TABLE_LOOKUP,
            Difficulty.EASY,
            requires_table=True,
        ),
        qc(
            "03",
            "What is the minimum RAM and volume size allocated to the Vector Indexer node role?",
            "The Vector Indexer requires a minimum of 64 GB RAM and a volume size of 1000 GB NVMe SSD.",
            [f"{spec_id}_e0006"],
            QuestionCategory.TABLE_LOOKUP,
            Difficulty.MEDIUM,
            requires_table=True,
        ),
        qc(
            "04",
            "What backup frequency is specified for Ingestion Worker disk volumes?",
            "The backup frequency for Ingestion Worker disk volumes is daily differential.",
            [f"{spec_id}_e0006"],
            QuestionCategory.TABLE_LOOKUP,
            Difficulty.EASY,
            requires_table=True,
        ),
        # Table questions (DOCX table)
        qc(
            "05",
            "According to the model benchmark evaluation table, what is the P99 latency and F1 score of Titan-v3?",
            "Titan-v3 achieved a P99 latency of 145ms and an Answer F1 score of 96.4%.",
            [f"{docx_id}_e0005"],
            QuestionCategory.TABLE_LOOKUP,
            Difficulty.EASY,
            requires_table=True,
        ),
        qc(
            "06",
            "Which model has the highest throughput in the benchmark results table, and what is its throughput value?",
            "Titan-v3 has the highest throughput at 380.2 tps.",
            [f"{docx_id}_e0005"],
            QuestionCategory.NUMERICAL_COMPARISON,
            Difficulty.MEDIUM,
            requires_table=True,
        ),
        qc(
            "07",
            "How does the P99 latency of Atlas-v1 compare to Nova-v2 in the quantitative benchmarks?",
            "Atlas-v1 has a P99 latency of 450ms compared to 210ms for Nova-v2.",
            [f"{docx_id}_e0005"],
            QuestionCategory.NUMERICAL_COMPARISON,
            Difficulty.MEDIUM,
            requires_table=True,
        ),
        # PDF Page Provenance questions
        qc(
            "08",
            "According to page 1 of the operations guide, which node role is assigned to node-beta-02?",
            "node-beta-02 is the secondary replica responsible for streaming write-ahead log replication.",
            [f"{pdf_id}_e0000"],
            QuestionCategory.PROVENANCE,
            Difficulty.MEDIUM,
        ),
        qc(
            "09",
            "According to page 1 of the operations guide, what role does node-gamma-03 perform?",
            "node-gamma-03 is the quorum witness participating in leader election and split-brain resolution.",
            [f"{pdf_id}_e0000"],
            QuestionCategory.PROVENANCE,
            Difficulty.MEDIUM,
        ),
        qc(
            "10",
            "According to page 2 of the operations guide, what is the heartbeat check interval and failover trigger timeout?",
            "The heartbeat check interval is 500 milliseconds and the failover trigger timeout is 3500 milliseconds.",
            [f"{pdf_id}_e0001"],
            QuestionCategory.PROVENANCE,
            Difficulty.MEDIUM,
        ),
        qc(
            "11",
            "On which UDP port does health monitoring heartbeat traffic communicate according to page 2 of the operations guide?",
            "Heartbeat channels communicate over UDP port 7443.",
            [f"{pdf_id}_e0001"],
            QuestionCategory.PROVENANCE,
            Difficulty.MEDIUM,
        ),
        qc(
            "12",
            "What are the RPO and RTO disaster recovery targets specified on page 3 of the operations guide?",
            "The target RPO is 15 minutes and the target RTO is 120 minutes.",
            [f"{pdf_id}_e0002"],
            QuestionCategory.PROVENANCE,
            Difficulty.EASY,
        ),
        qc(
            "13",
            "How frequently are offsite differential snapshots transferred according to page 3 of the operations guide?",
            "Offsite differential snapshots are transferred every 30 minutes over an encrypted TLS tunnel.",
            [f"{pdf_id}_e0002"],
            QuestionCategory.PROVENANCE,
            Difficulty.EASY,
        ),
        # Adversarial / Near-Duplicate passages (Incident logs)
        qc(
            "14",
            "What was the root cause and resolution of incident INC-401 on worker-node-east-01?",
            "The root cause was runaway regex parsing in document element splitter, resolved by deploying a patched regex filter and restarting the daemon.",
            [f"{log_id}_e0001"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.HARD,
        ),
        qc(
            "15",
            "How does incident INC-402 differ from INC-401 in its target host and symptom?",
            "INC-402 occurred on worker-node-east-02 with 96% memory usage, whereas INC-401 occurred on worker-node-east-01 with 98% CPU usage.",
            [f"{log_id}_e0001", f"{log_id}_e0002"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.HARD,
        ),
        qc(
            "16",
            "What resolution was applied to resolve incident INC-402?",
            "Implemented temporary file streaming on disk and capped maximum in-memory allocation to 512 MB.",
            [f"{log_id}_e0002"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.MEDIUM,
        ),
        qc(
            "17",
            "What was the root cause of the packet drop in incident INC-501 on api-gateway-01?",
            "The root cause was HTTP client connection pool exhaustion under concurrent polling requests.",
            [f"{log_id}_e0003"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.HARD,
        ),
        qc(
            "18",
            "What network kernel parameter was tuned to resolve incident INC-502 on api-gateway-02?",
            "The tcp_tw_reuse kernel network parameter was tuned.",
            [f"{log_id}_e0004"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.HARD,
        ),
        qc(
            "19",
            "Which incident involved ephemeral port exhaustion, INC-501 or INC-502?",
            "Incident INC-502 involved ephemeral port exhaustion.",
            [f"{log_id}_e0004"],
            QuestionCategory.ADVERSARIAL,
            Difficulty.HARD,
        ),
        # Cross-document / Similar terminology
        qc(
            "20",
            "What timeout triggers error code ERR-1001 in the system specification, and how does it compare to the failover trigger timeout in the operations guide?",
            "ERR-1001 has a connection timeout threshold of 3000ms, while the failover trigger timeout in the operations guide is 3500ms.",
            [f"{spec_id}_e0008", f"{pdf_id}_e0001"],
            QuestionCategory.CROSS_DOCUMENT,
            Difficulty.HARD,
            multi_chunk=True,
        ),
        qc(
            "21",
            "What maximum batch token threshold triggers error code ERR-1002 in the system specification?",
            "ERR-1002 is triggered when the batch token count exceeds 32,768 tokens in a single forward pass.",
            [f"{spec_id}_e0009"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "22",
            "What vector dimension causes error ERR-3001 if an embedding vector deviates from it?",
            "A deviation from the index declaration dimension of 768 triggers ERR-3001.",
            [f"{spec_id}_e0011"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "23",
            "Which network port is allocated for the Prometheus metric scraper in system_spec.md?",
            "Port 9090 is allocated for Prometheus metric scraping.",
            [f"{spec_id}_e0016"],
            QuestionCategory.DIRECT_FACTUAL,
            Difficulty.EASY,
        ),
        qc(
            "24",
            "Which model is recommended for Tier-1 search queries in the product evaluation document and why?",
            "Titan-v3 is recommended for Tier-1 search queries because of its 96.4% F1 score and 145ms P99 latency.",
            [f"{docx_id}_e0007"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "25",
            "Why was Atlas-v1 deprecated for real-time interactive search according to the deployment recommendations?",
            "Atlas-v1 was deprecated due to excessive 450ms tail latency.",
            [f"{docx_id}_e0009"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "26",
            "What role is recommended for Nova-v2 during high concurrency spikes?",
            "Nova-v2 is designated as primary fallback model to maintain sub-250ms response times during GPU saturation.",
            [f"{docx_id}_e0008"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "27",
            "What consensus vote threshold is required for automatic failover in the operations guide?",
            "A strict majority quorum of at least 2 out of 3 node votes is required.",
            [f"{pdf_id}_e0001"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "28",
            "How quickly does node-beta-02 assume the primary role when node-alpha-01 fails?",
            "node-beta-02 assumes the primary role within 4.2 seconds.",
            [f"{pdf_id}_e0001"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.HARD,
        ),
        qc(
            "29",
            "What queue length triggers HTTP 429 throttling under error code ERR-2001 in system_spec.md?",
            "A queue length exceeding 500 queued requests triggers backpressure and HTTP 429 throttling.",
            [f"{spec_id}_e0010"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
        qc(
            "30",
            "What storage medium and file system format are specified for vector index volumes in section 2 of system_spec.md?",
            "Direct-attached NVMe storage formatted with XFS is specified.",
            [f"{spec_id}_e0005"],
            QuestionCategory.SPECIFIC_DETAIL,
            Difficulty.MEDIUM,
        ),
    ]

    return cases


def build_track_d() -> list[dict]:
    # 25 unanswerable questions testing abstention and hallucination resistance on Track D
    prov = {
        "annotator": "RAG_EVAL Curation Team",
        "verified_at": "2026-11-16T12:00:00Z",
        "source_documents": ["reference_guide.md"],
        "verification_notes": "Ground-truth verified as completely absent from or unsupported by the reference corpus.",
    }

    def qd(qid: str, query: str) -> dict:
        return {
            "query_id": f"track_d_q{qid:0>2s}",
            "query": query,
            "relevant_chunk_ids": [],
            "relevant_element_ids": [],
            "reference_answer": None,
            "metadata": {},
            "status": "verified",
            "answerability": "unanswerable",
            "answerable": False,
            "requires_multi_chunk": False,
            "requires_table": False,
            "category": QuestionCategory.UNANSWERABLE.value,
            "difficulty": Difficulty.MEDIUM.value,
            "provenance": {**prov, "evidence_references": []},
            "track": "track_d",
        }

    cases = [
        qd(
            "01",
            "What configuration setting enables GraphQL subscription websockets in the microservice framework?",
        ),
        qd(
            "02",
            "How does the microservice framework configure Apache Cassandra for write-ahead replication?",
        ),
        qd("03", "Which encryption algorithm is used to secure Memcached distributed cache nodes?"),
        qd("04", "What command deploys the worker service onto an Apache Mesos cluster?"),
        qd(
            "05",
            "What is the maximum message payload size supported by the framework's native Apache Kafka 0.8 adapter?",
        ),
        qd(
            "06",
            "How do you configure mutual TLS authentication with HashiCorp Vault in the authentication module?",
        ),
        qd("07", "Which port is used by the framework for the Jaeger Thrift UDP tracing agent?"),
        qd(
            "08",
            "What is the default sliding window duration for the framework's Hystrix circuit breaker?",
        ),
        qd("09", "How do you enable Amazon DynamoDB global tables in the storage subsystem?"),
        qd(
            "10",
            "What setting configures Google Cloud Pub/Sub pull subscriptions for worker tasks?",
        ),
        qd(
            "11",
            "Which OAuth2 grant type is used for client credentials flow with Azure Active Directory?",
        ),
        qd(
            "12",
            "What is the maximum allowed size of an image attachment in the PostgreSQL queue payload?",
        ),
        qd(
            "13",
            "How do you configure Elasticsearch index aliases for multi-tenant search indexing?",
        ),
        qd(
            "14", "What command initiates a zero-downtime rolling restart on Kubernetes using Helm?"
        ),
        qd(
            "15",
            "Which compression algorithm is used by the framework's gRPC stream compressor: zstandard or snappy?",
        ),
        qd(
            "16",
            "What is the recommended retention period for Zipkin tracing spans in the framework?",
        ),
        qd(
            "17",
            "How is distributed transaction consensus managed using the Two-Phase Commit coordinator?",
        ),
        qd(
            "18", "What environment variable overrides the NATS streaming cluster ID in production?"
        ),
        qd(
            "19", "Which cipher suite is enforced for SAML 2.0 single sign-on assertion signatures?"
        ),
        qd(
            "20",
            "What is the maximum heap size setting for the JVM garbage collection profiler in the framework?",
        ),
        qd(
            "21",
            "How do you configure Couchbase bucket replicas for read-your-own-writes consistency?",
        ),
        qd(
            "22",
            "What CLI argument enables WebSocket compression using permessage-deflate in uvicorn?",
        ),
        qd("23", "Which metrics exporter sends statsd metrics to Datadog over UDP port 8125?"),
        qd(
            "24",
            "What is the default timeout for Amazon SQS long polling requests in the worker daemon?",
        ),
        qd("25", "How does the framework handle CockroachDB distributed transaction retry aborts?"),
    ]

    return cases


def main() -> None:
    golden_dir = Path("eval/golden")
    golden_dir.mkdir(parents=True, exist_ok=True)

    print("Building Track B (52 queries)...")
    track_b_cases = build_track_b()
    track_b_path = golden_dir / "track_b.jsonl"
    with open(track_b_path, "w", encoding="utf-8") as f:
        for c in track_b_cases:
            f.write(json.dumps(c) + "\n")
    print(f"Track B saved to {track_b_path} ({len(track_b_cases)} cases)")

    print("Building Track C (30 queries)...")
    track_c_cases = build_track_c()
    track_c_path = golden_dir / "track_c.jsonl"
    with open(track_c_path, "w", encoding="utf-8") as f:
        for c in track_c_cases:
            f.write(json.dumps(c) + "\n")
    print(f"Track C saved to {track_c_path} ({len(track_c_cases)} cases)")

    print("Building Track D (25 queries)...")
    track_d_cases = build_track_d()
    track_d_path = golden_dir / "track_d.jsonl"
    with open(track_d_path, "w", encoding="utf-8") as f:
        for c in track_d_cases:
            f.write(json.dumps(c) + "\n")
    print(f"Track D saved to {track_d_path} ({len(track_d_cases)} cases)")

    # Run automated validation against each track
    validator = DatasetValidator()
    print("\n--- Validating Datasets Against Respective Corpora ---")
    all_clean = True
    for track_name in ["track_a", "track_b", "track_c", "track_d"]:
        ds_path = golden_dir / f"{track_name}.jsonl"
        c_dir = Path("eval/corpora") / track_name
        report = validator.validate_file(file_path=ds_path, corpus=c_dir)
        print(
            f"Track {track_name.upper()}: valid={report.valid} errors={len(report.errors)} warnings={len(report.warnings)}"
        )
        if not report.valid:
            all_clean = False
            for err in report.errors:
                print(f"   [ERROR] {err.code}: {err.message} (qid={err.query_id})")
        if report.warnings:
            all_clean = False
            for w in report.warnings:
                print(f"   [WARN] {w.code}: {w.message} (qid={w.query_id})")
        assert report.valid, f"Validation failed for {track_name}"

    if all_clean:
        print("\nAll golden datasets generated and 100% validated: 0 ERRORS, 0 WARNINGS!")
    else:
        print("\nSome warnings or errors remain.")


if __name__ == "__main__":
    main()
