"""Deterministic, leakage-safe query generator for independent routing dataset."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.ingest import create_default_ingestion_pipeline
from core.normalize.schema import Chunk
from eval.routing.leakage import check_query_leakage, load_all_protected_queries, normalize_query
from eval.routing.schema import QueryCategory, RoutingExample, RoutingSplit


@dataclass(frozen=True)
class DocumentSpec:
    """Specification of a source document for routing dataset generation."""

    relative_path: str
    filename: str
    topic: str
    split: RoutingSplit


# Strict document-level assignment: Train (8), Validation (5), Test (4)
DEFAULT_DOCUMENT_SPECS: tuple[DocumentSpec, ...] = (
    # Train partition
    DocumentSpec(
        "eval/corpora/track_b/rfc8259.txt",
        "rfc8259.txt",
        "RFC 8259 JSON Specification",
        RoutingSplit.TRAIN,
    ),
    DocumentSpec(
        "eval/corpora/track_a/architecture.md",
        "architecture.md",
        "System Architecture",
        RoutingSplit.TRAIN,
    ),
    DocumentSpec(
        "docs/render-deployment.md",
        "render-deployment.md",
        "Render Cloud Deployment",
        RoutingSplit.TRAIN,
    ),
    DocumentSpec(
        "eval/corpora/track_a/hosted-beta.md",
        "hosted-beta.md",
        "Hosted Beta Platform",
        RoutingSplit.TRAIN,
    ),
    DocumentSpec(
        "docs/benchmarks.md", "benchmarks.md", "Evaluation Benchmarks and SLA", RoutingSplit.TRAIN
    ),
    DocumentSpec(
        "eval/corpora/track_c/incident_log.txt",
        "incident_log.txt",
        "Production Incident Log",
        RoutingSplit.TRAIN,
    ),
    DocumentSpec("docs/hosted-terms.md", "hosted-terms.md", "Terms of Service", RoutingSplit.TRAIN),
    DocumentSpec(
        "eval/corpora/track_c/product_evaluation.docx",
        "product_evaluation.docx",
        "Product Evaluation Assessment",
        RoutingSplit.TRAIN,
    ),
    # Validation partition
    DocumentSpec(
        "eval/corpora/track_a/runbook.md",
        "runbook.md",
        "Operational Runbook",
        RoutingSplit.VALIDATION,
    ),
    DocumentSpec(
        "docs/findings.md", "findings.md", "Architectural Findings", RoutingSplit.VALIDATION
    ),
    DocumentSpec(
        "docs/hosted-privacy.md",
        "hosted-privacy.md",
        "Hosted Privacy Policy",
        RoutingSplit.VALIDATION,
    ),
    DocumentSpec(
        "eval/corpora/track_c/operations_guide.pdf",
        "operations_guide.pdf",
        "Operations Guide",
        RoutingSplit.VALIDATION,
    ),
    DocumentSpec(
        "eval/corpora/track_d/reference_guide.md",
        "reference_guide.md",
        "API Reference Guide",
        RoutingSplit.VALIDATION,
    ),
    # Test partition
    DocumentSpec("README.md", "README.md", "System Overview and Architecture", RoutingSplit.TEST),
    DocumentSpec(
        "eval/corpora/track_c/system_spec.md",
        "system_spec.md",
        "System Infrastructure Specification",
        RoutingSplit.TEST,
    ),
    DocumentSpec("docs/ci-cd.md", "ci-cd.md", "CI/CD Pipeline Architecture", RoutingSplit.TEST),
    DocumentSpec("docs/privacy.md", "privacy.md", "General Privacy Policy", RoutingSplit.TEST),
)


class RoutingDatasetGenerator:
    """Generates independent, leakage-free routing query examples from source documents."""

    def __init__(
        self,
        document_specs: Sequence[DocumentSpec] = DEFAULT_DOCUMENT_SPECS,
        project_root: Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        seed: int = 42,
    ) -> None:
        self.document_specs = tuple(document_specs)
        self.project_root = project_root or Path(".")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.seed = seed
        self.pipeline = create_default_ingestion_pipeline(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.protected_queries = load_all_protected_queries(self.project_root)

    def generate(self, target_per_chunk: int = 5) -> list[RoutingExample]:
        """Generate all routing examples deterministically."""
        rng = random.Random(self.seed)
        seen_queries: set[str] = set()
        examples: list[RoutingExample] = []

        seq_id = 1

        for spec in self.document_specs:
            doc_path = self.project_root / spec.relative_path
            if not doc_path.exists():
                continue

            res = self.pipeline.ingest(doc_path)
            if not res.chunks:
                continue

            doc_id = (
                res.document.metadata.document_id
                if res.document
                else hashlib.sha256(spec.filename.encode()).hexdigest()[:16]
            )

            for chunk in res.chunks:
                chunk_examples = self._generate_for_chunk(
                    chunk=chunk,
                    doc_id=doc_id,
                    spec=spec,
                    seen_queries=seen_queries,
                    rng=rng,
                    start_seq=seq_id,
                    target_count=target_per_chunk,
                )
                examples.extend(chunk_examples)
                seq_id += len(chunk_examples)

        return examples

    def _generate_for_chunk(
        self,
        chunk: Chunk,
        doc_id: str,
        spec: DocumentSpec,
        seen_queries: set[str],
        rng: random.Random,
        start_seq: int,
        target_count: int,
    ) -> list[RoutingExample]:
        text = chunk.text
        examples: list[RoutingExample] = []
        current_seq = start_seq

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("```")
        ]
        clean_text = " ".join(lines)
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`#])", clean_text)
        sentences = [
            s.strip()
            for s in raw_sentences
            if len(s.strip()) > 25 and not s.strip().startswith("|")
        ]

        backticked = re.findall(r"`([^`]{2,40})`", text)
        valid_idents = [
            t.strip() for t in backticked if not t.startswith("http") and len(t.split()) <= 4
        ]

        def add_candidate(query: str, category: QueryCategory, answer: str) -> bool:
            nonlocal current_seq
            q_clean = query.strip()
            if not q_clean.endswith("?"):
                q_clean += "?"
            norm = normalize_query(q_clean)
            if norm in seen_queries:
                return False

            is_leaking, _, _ = check_query_leakage(
                q_clean, self.protected_queries, fuzzy_threshold=0.70
            )
            if is_leaking:
                return False

            seen_queries.add(norm)
            ex = RoutingExample(
                example_id=f"route_p11_{spec.split.value}_{current_seq:04d}",
                source_document_id=doc_id,
                source_document=spec.filename,
                source_element_ids=chunk.source_element_ids,
                source_chunk_ids=[chunk.chunk_id],
                generation_method="deterministic_chunk_synthesis",
                query=q_clean,
                query_category=category,
                expected_answer=answer.strip(),
                candidate_retrieval_strategies=["bm25", "dense", "hybrid"],
                generation_seed=self.seed,
                dataset_version="1.0.0",
                split=spec.split,
                metadata={
                    "doc_topic": spec.topic,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
            )
            examples.append(ex)
            current_seq += 1
            return True

        # Strategy 1: Technical Identifier Questions
        for ident in valid_idents[:2]:
            matching_s = next((s for s in sentences if ident in s), lines[0] if lines else ident)
            template_choices = [
                f"What role or function does `{ident}` serve in {spec.topic}?",
                f"How is `{ident}` configured or referenced in {spec.topic}?",
                f"Which component or section defines `{ident}` in {spec.topic}?",
            ]
            q_text = rng.choice(template_choices)
            add_candidate(q_text, QueryCategory.TECHNICAL_IDENTIFIER, matching_s)

        # Strategy 2: Numerical & Quantitative Questions
        num_patterns = [
            (r"\b(\d+(?:\.\d+)?)\s*(ms|seconds|minutes|hours)\b", "latency or duration"),
            (r"\b(\d+)\s*(characters|chars|tokens|elements)\b", "size or token count"),
            (r"\b(\d+)\s*(vCPU|RAM|GB|MB|KB)\b", "hardware resource allocation"),
            (r"\b(P99|P95|P50|SLA|\d+%\s*uptime)\b", "operational SLA target"),
            (r"\b(port\s*\d+|\d{4,5})\b", "network port assignment"),
            (r"\b(\d+\.\d+(?:\.\d+)?)\b", "version or threshold"),
        ]
        for pattern, metric_label in num_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                s_ctx = next((s for s in sentences if m.group(0) in s), None)
                if s_ctx:
                    add_candidate(
                        f"What {metric_label} is specified for {spec.topic} ({m.group(0)})?",
                        QueryCategory.NUMERICAL,
                        s_ctx,
                    )
                    break

        # Strategy 3: Configuration & Environment Questions
        config_keywords = [
            "config",
            "environment",
            "env",
            "setting",
            "parameter",
            "variable",
            "default",
            "flag",
        ]
        for s in sentences:
            if any(re.search(rf"\b{re.escape(kw)}\b", s, re.IGNORECASE) for kw in config_keywords):
                add_candidate(
                    f"What configuration setting or parameter is specified in {spec.topic}?",
                    QueryCategory.CONFIGURATION,
                    s,
                )
                break

        # Strategy 4: Direct Factual Questions
        factual_verbs = [
            "is",
            "represents",
            "defines",
            "provides",
            "consists of",
            "includes",
            "supports",
            "uses",
            "enables",
        ]
        for s in sentences:
            words = s.split()
            if len(words) >= 6:
                subject_phrase = " ".join(words[:4])
                if any(f" {v} " in s.lower() for v in factual_verbs):
                    add_candidate(
                        f"According to {spec.topic}, what is documented about {subject_phrase}?",
                        QueryCategory.DIRECT_FACTUAL,
                        s,
                    )
                    break

        # Strategy 5: Procedural Questions
        proc_keywords = [
            "must",
            "should",
            "ensure",
            "steps",
            "to deploy",
            "to run",
            "command",
            "process",
            "workflow",
            "action",
            "verify",
        ]
        for s in sentences:
            if any(re.search(rf"\b{re.escape(kw)}\b", s, re.IGNORECASE) for kw in proc_keywords):
                add_candidate(
                    f"What operational procedure or requirement is defined in {spec.topic}?",
                    QueryCategory.PROCEDURAL,
                    s,
                )
                break

        # Strategy 6: Multi-Step / Relationship Questions
        if len(sentences) >= 2:
            multi_ans = f"{sentences[0]} In addition: {sentences[1]}"
            add_candidate(
                f"In {spec.topic}, how do concepts in this section coordinate system operation?",
                QueryCategory.MULTI_STEP,
                multi_ans,
            )

        # Strategy 7: Specific Details & Factual Fallbacks to reach target
        for _idx, s in enumerate(sentences):
            if len(examples) >= target_count:
                break
            short_ref = " ".join(s.split()[:5])
            add_candidate(
                f"What specific detail does {spec.topic} outline concerning: {short_ref}?",
                QueryCategory.SPECIFIC_DETAIL,
                s,
            )

        return examples
