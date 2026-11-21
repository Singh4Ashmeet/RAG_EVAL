"""Audit utilities for leakage detection and split isolation."""

from __future__ import annotations

import csv
import difflib
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval.datasets.loader import load_jsonl_dataset
from eval.routing.schema import LeakageAuditReport, RoutingExample

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Normalize a query for strict leakage comparison."""
    cleaned = _PUNCT_RE.sub(" ", query.lower())
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def query_token_signature(query: str) -> str:
    """Produce a canonical sorted-token string for order-invariant comparison."""
    tokens = normalize_query(query).split()
    return " ".join(sorted(tokens))


def calculate_similarity(query_a: str, query_b: str) -> float:
    """Calculate SequenceMatcher similarity ratio between two normalized queries."""
    norm_a = normalize_query(query_a)
    norm_b = normalize_query(query_b)
    if norm_a == norm_b:
        return 1.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


def load_all_protected_queries(project_root: Path | None = None) -> list[str]:
    """Load all protected evaluation queries from Tracks A-D and M4 artifacts."""
    root = project_root or Path(".")
    queries: set[str] = set()

    # Load from Golden Tracks A, B, C, D
    for track_name in ["track_a", "track_b", "track_c", "track_d"]:
        golden_path = root / f"eval/golden/{track_name}.jsonl"
        if golden_path.exists():
            dataset = load_jsonl_dataset(str(golden_path))
            for case in dataset.cases:
                q = case.query.strip()
                if q:
                    queries.add(q)

    # Load from M4 reports if present
    for report_file in [
        "docs/reports/m4_1_query_analysis.csv",
        "docs/reports/m4_2_per_query_results.csv",
    ]:
        csv_path = root / report_file
        if csv_path.exists():
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = row.get("query", "").strip()
                    if q:
                        queries.add(q)

    return sorted(queries)


def check_query_leakage(
    candidate: str,
    protected_queries: Sequence[str],
    fuzzy_threshold: float = 0.70,
) -> tuple[bool, str | None, float]:
    """Check if a candidate query leaks or closely resembles any protected query.

    Returns:
        (is_leaking, matched_protected_query, max_similarity)
    """
    candidate_norm = normalize_query(candidate)
    candidate_sig = query_token_signature(candidate)
    candidate_tokens = set(candidate_norm.split())

    max_sim = 0.0
    best_match: str | None = None

    for protected in protected_queries:
        prot_norm = normalize_query(protected)

        # 1. Exact match
        if candidate_norm == prot_norm:
            return True, protected, 1.0

        # 2. Normalized token signature match
        prot_sig = query_token_signature(protected)
        if candidate_sig == prot_sig and len(candidate_sig.split()) >= 3:
            return True, protected, 1.0

        # 3. Fuzzy similarity (optimized: check token overlap first)
        prot_tokens = set(prot_norm.split())
        overlap = len(candidate_tokens & prot_tokens)
        max_possible_sim = (2.0 * overlap) / max(1, len(candidate_tokens) + len(prot_tokens))
        if max_possible_sim < fuzzy_threshold * 0.75:
            continue

        sim = difflib.SequenceMatcher(None, candidate_norm, prot_norm).ratio()
        if sim > max_sim:
            max_sim = sim
            best_match = protected

        if sim >= fuzzy_threshold:
            return True, protected, sim

    return False, best_match, max_sim


def audit_dataset_leakage(
    examples: Sequence[RoutingExample],
    protected_queries: Sequence[str],
    fuzzy_threshold: float = 0.70,
) -> LeakageAuditReport:
    """Run comprehensive leakage audit on generated routing examples."""
    exact_matches = 0
    normalized_matches = 0
    fuzzy_matches = 0
    max_sim = 0.0
    diagnostics: list[str] = []

    protected_norm_set = {normalize_query(q): q for q in protected_queries}
    protected_sig_set = {query_token_signature(q): q for q in protected_queries}

    for ex in examples:
        q_norm = normalize_query(ex.query)
        q_sig = query_token_signature(ex.query)

        if q_norm in protected_norm_set:
            exact_matches += 1
            match_str = protected_norm_set[q_norm]
            diagnostics.append(
                f"EXACT_LEAKAGE: Example '{ex.example_id}' matches protected: '{match_str}'"
            )
            continue

        if q_sig in protected_sig_set and len(q_sig.split()) >= 3:
            normalized_matches += 1
            sig_match = protected_sig_set[q_sig]
            diagnostics.append(
                f"NORMALIZED_LEAKAGE: Example '{ex.example_id}' token-matches: '{sig_match}'"
            )
            continue

        # Check fuzzy match
        q_tokens = set(q_norm.split())
        for _prot, prot_norm in protected_norm_set.items():
            prot_tokens = set(prot_norm.split())
            overlap = len(q_tokens & prot_tokens)
            max_possible_sim = (2.0 * overlap) / max(1, len(q_tokens) + len(prot_tokens))
            if max_possible_sim < fuzzy_threshold * 0.75:
                continue

            sim = difflib.SequenceMatcher(None, q_norm, prot_norm).ratio()
            if sim > max_sim:
                max_sim = sim
            if sim >= fuzzy_threshold:
                fuzzy_matches += 1
                diagnostics.append(
                    f"FUZZY_LEAKAGE: Example '{ex.example_id}' sim={sim:.3f} to '{prot_norm}'"
                )
                break

    # Audit source document separation
    split_docs: dict[str, set[str]] = {}
    for ex in examples:
        split_docs.setdefault(ex.split.value, set()).add(ex.source_document)

    train_docs = split_docs.get("train", set())
    val_docs = split_docs.get("validation", set())
    test_docs = split_docs.get("test", set())

    train_val_overlap = train_docs & val_docs
    train_test_overlap = train_docs & test_docs
    val_test_overlap = val_docs & test_docs

    source_sep_ok = True
    if train_val_overlap:
        source_sep_ok = False
        diagnostics.append(
            f"SOURCE_OVERLAP: Documents shared between train and validation: {train_val_overlap}"
        )
    if train_test_overlap:
        source_sep_ok = False
        diagnostics.append(
            f"SOURCE_OVERLAP: Documents shared between train and test: {train_test_overlap}"
        )
    if val_test_overlap:
        source_sep_ok = False
        diagnostics.append(
            f"SOURCE_OVERLAP: Documents shared between validation and test: {val_test_overlap}"
        )

    is_clean = (
        exact_matches == 0 and normalized_matches == 0 and fuzzy_matches == 0 and source_sep_ok
    )

    return LeakageAuditReport(
        audit_timestamp=datetime.now(UTC).isoformat(),
        protected_queries_count=len(protected_queries),
        evaluated_queries_count=len(examples),
        exact_matches_found=exact_matches,
        normalized_matches_found=normalized_matches,
        fuzzy_matches_found=fuzzy_matches,
        max_fuzzy_similarity=round(max_sim, 4),
        fuzzy_threshold=fuzzy_threshold,
        source_separation_verified=source_sep_ok,
        protected_dataset_isolation_verified=(
            exact_matches == 0 and normalized_matches == 0 and fuzzy_matches == 0
        ),
        status="PASS" if is_clean else "FAIL",
        diagnostics=diagnostics,
    )
