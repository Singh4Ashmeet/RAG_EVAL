"""Validate structural, semantic, and manifest integrity of Phase 11 routing dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval.routing.schema import (
    LeakageAuditReport,
    OracleLabelRecord,
    RoutingDatasetManifest,
    RoutingExample,
)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate structural and manifest integrity of Phase 11 routing dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/routing/phase11"),
        help="Directory containing routing dataset artifacts (default: data/routing/phase11).",
    )
    return parser.parse_args(args)


def validate_dataset(dataset_dir: Path) -> tuple[bool, list[str]]:
    diagnostics: list[str] = []

    # 1. Required files check
    manifest_path = dataset_dir / "routing_dataset_manifest.json"
    full_path = dataset_dir / "routing_dataset.jsonl"
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "validation.jsonl"
    test_path = dataset_dir / "test.jsonl"
    oracle_path = dataset_dir / "oracle_labels.jsonl"
    leakage_path = dataset_dir / "leakage_report.json"

    required_files = [
        manifest_path,
        full_path,
        train_path,
        val_path,
        test_path,
        oracle_path,
        leakage_path,
    ]
    for rf in required_files:
        if not rf.exists():
            diagnostics.append(f"MISSING_FILE: Required artifact '{rf.name}' does not exist.")

    if diagnostics:
        return False, diagnostics

    # 2. Manifest Schema & Hashes
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = RoutingDatasetManifest.model_validate(manifest_data)
    except Exception as exc:
        diagnostics.append(f"MANIFEST_ERROR: Manifest failed validation: {exc}")
        return False, diagnostics

    for fname, expected_hash in manifest.files.items():
        target = dataset_dir / fname
        if not target.exists():
            diagnostics.append(f"HASH_VERIFY_ERROR: Manifest file '{fname}' not found.")
            continue
        actual_hash = compute_sha256(target)
        if actual_hash != expected_hash:
            diagnostics.append(
                f"HASH_MISMATCH: File '{fname}' hash {actual_hash[:16]}... != manifest {expected_hash[:16]}..."
            )

    # 3. Leakage Report status
    try:
        leakage_data = json.loads(leakage_path.read_text(encoding="utf-8"))
        leakage_report = LeakageAuditReport.model_validate(leakage_data)
        if leakage_report.status != "PASS":
            diagnostics.append(f"LEAKAGE_FAIL: Leakage report status is '{leakage_report.status}'.")
    except Exception as exc:
        diagnostics.append(f"LEAKAGE_REPORT_ERROR: Failed to read leakage report: {exc}")

    # 4. JSONL Schema & Integrity
    def load_examples(file_path: Path) -> list[RoutingExample]:
        exs: list[RoutingExample] = []
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    exs.append(RoutingExample.model_validate(json.loads(s)))
                except Exception as ex:
                    diagnostics.append(
                        f"SCHEMA_ERROR: {file_path.name}:{line_no} failed validation: {ex}"
                    )
        return exs

    full_examples = load_examples(full_path)
    train_examples = load_examples(train_path)
    val_examples = load_examples(val_path)
    test_examples = load_examples(test_path)

    # Check counts
    if len(full_examples) < 500:
        diagnostics.append(
            f"COUNT_ERROR: Total examples {len(full_examples)} is below minimum requirement of 500."
        )

    if len(full_examples) != manifest.total_examples:
        diagnostics.append(
            f"COUNT_MISMATCH: Full dataset count {len(full_examples)} != manifest {manifest.total_examples}"
        )

    split_sum = len(train_examples) + len(val_examples) + len(test_examples)
    if split_sum != len(full_examples):
        diagnostics.append(
            f"SPLIT_SUM_MISMATCH: Sum of splits ({split_sum}) != full dataset count ({len(full_examples)})"
        )

    # Check ID uniqueness
    seen_ids: set[str] = set()
    for ex in full_examples:
        if ex.example_id in seen_ids:
            diagnostics.append(
                f"DUPLICATE_ID: Example ID '{ex.example_id}' appears more than once."
            )
        seen_ids.add(ex.example_id)

    # Check document split separation
    train_docs = {ex.source_document for ex in train_examples}
    val_docs = {ex.source_document for ex in val_examples}
    test_docs = {ex.source_document for ex in test_examples}

    if train_docs & val_docs:
        diagnostics.append(
            f"DOCUMENT_LEAKAGE: Documents shared between train and validation: {train_docs & val_docs}"
        )
    if train_docs & test_docs:
        diagnostics.append(
            f"DOCUMENT_LEAKAGE: Documents shared between train and test: {train_docs & test_docs}"
        )
    if val_docs & test_docs:
        diagnostics.append(
            f"DOCUMENT_LEAKAGE: Documents shared between validation and test: {val_docs & test_docs}"
        )

    # 5. Oracle Labels Check
    oracle_records: list[OracleLabelRecord] = []
    with open(oracle_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                oracle_records.append(OracleLabelRecord.model_validate(json.loads(s)))
            except Exception as ex:
                diagnostics.append(
                    f"ORACLE_SCHEMA_ERROR: {oracle_path.name}:{line_no} failed validation: {ex}"
                )

    if len(oracle_records) != len(full_examples):
        diagnostics.append(
            f"ORACLE_COUNT_MISMATCH: Oracle records count {len(oracle_records)} != examples {len(full_examples)}"
        )

    is_valid = len(diagnostics) == 0
    return is_valid, diagnostics


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    dataset_dir = parsed.dataset_dir

    print(f"\n{'=' * 70}")
    print(f"VALIDATING ROUTING DATASET ARTIFACTS: {dataset_dir}")
    print(f"{'=' * 70}")

    is_valid, diagnostics = validate_dataset(dataset_dir)

    if is_valid:
        print("Dataset validation: PASS")
        manifest_path = dataset_dir / "routing_dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"  Total Examples      : {manifest.get('total_examples')}")
        print(f"  Split Counts        : {manifest.get('split_counts')}")
        print(f"  Oracle Distribution : {manifest.get('oracle_distribution')}")
        print(f"  Category Counts     : {manifest.get('category_distribution')}")
        print(f"  Source Documents    : {manifest.get('source_document_count')}")
        print(f"{'=' * 70}\n")
        return 0
    else:
        print("Dataset validation: FAIL")
        print("\nDiagnostics:")
        for diag in diagnostics:
            print(f"  - {diag}")
        print(f"{'=' * 70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
