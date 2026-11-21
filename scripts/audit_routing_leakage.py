"""Audit routing dataset for leakage against protected evaluation benchmarks.

Verifies:
1. Zero exact query overlap with Tracks A, B, C, D and M4 reports.
2. Zero normalized token signature overlap.
3. Zero near-duplicate similarity above threshold (default 0.70).
4. Strict source document separation across train, validation, and test splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval.routing.leakage import audit_dataset_leakage, load_all_protected_queries
from eval.routing.schema import RoutingExample


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit routing dataset for leakage against protected evaluation sets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/routing/phase11"),
        help="Directory containing the routing dataset artifacts (default: data/routing/phase11).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Fuzzy similarity leakage rejection threshold (default: 0.70).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to output leakage report JSON (default: <dataset-dir>/leakage_report.json).",
    )
    return parser.parse_args(args)


def load_dataset_examples(dataset_dir: Path) -> list[RoutingExample]:
    examples: list[RoutingExample] = []
    # Prefer partition files if available, otherwise full dataset
    partition_files = [
        dataset_dir / "train.jsonl",
        dataset_dir / "validation.jsonl",
        dataset_dir / "test.jsonl",
    ]
    if all(p.exists() for p in partition_files):
        files_to_read = partition_files
    else:
        full_file = dataset_dir / "routing_dataset.jsonl"
        if not full_file.exists():
            raise FileNotFoundError(f"No routing dataset files found in {dataset_dir}")
        files_to_read = [full_file]

    for file_path in files_to_read:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    data = json.loads(line_str)
                    examples.append(RoutingExample.model_validate(data))

    return examples


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    dataset_dir = parsed.dataset_dir
    report_path = parsed.output_report or (dataset_dir / "leakage_report.json")

    print(f"\n{'=' * 70}")
    print(f"AUDITING DATASET LEAKAGE: {dataset_dir}")
    print(f"{'=' * 70}")

    protected_queries = load_all_protected_queries(_project_root)
    print(f"Loaded {len(protected_queries)} protected evaluation queries (Tracks A-D + M4).")

    try:
        examples = load_dataset_examples(dataset_dir)
        print(f"Loaded {len(examples)} independent routing examples from {dataset_dir}.")
    except Exception as exc:
        print(f"ERROR loading dataset: {exc}", file=sys.stderr)
        return 1

    report = audit_dataset_leakage(
        examples=examples,
        protected_queries=protected_queries,
        fuzzy_threshold=parsed.threshold,
    )

    # Save report only if explicitly requested
    if parsed.output_report is not None:
        report_path = parsed.output_report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Audit report saved to: {report_path}")

    # Display audit findings
    print("\n--- Leakage Audit Findings ---")
    print(f"  Total Protected Queries     : {report.protected_queries_count}")
    print(f"  Total Evaluated Queries     : {report.evaluated_queries_count}")
    print(f"  Exact Matches Found         : {report.exact_matches_found}")
    print(f"  Normalized Matches Found    : {report.normalized_matches_found}")
    print(f"  Fuzzy Matches Found (>= {report.fuzzy_threshold}): {report.fuzzy_matches_found}")
    print(f"  Maximum Fuzzy Similarity    : {report.max_fuzzy_similarity:.4f}")
    print(f"  Source Separation Verified  : {report.source_separation_verified}")
    print(f"  Protected Dataset Isolation : {report.protected_dataset_isolation_verified}")
    print(f"  Overall Status              : {report.status}")

    if report.diagnostics:
        print("\nDiagnostics:")
        for diag in report.diagnostics:
            print(f"  - {diag}")

    print(f"{'=' * 70}\n")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
