"""Build, oracle-label, audit, and serialize Phase 11 independent routing dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.embed.ollama_embedder import OllamaEmbedder
from eval.routing.generator import DEFAULT_DOCUMENT_SPECS, RoutingDatasetGenerator
from eval.routing.leakage import audit_dataset_leakage, load_all_protected_queries
from eval.routing.oracle import OracleEvaluator
from eval.routing.schema import (
    OracleLabelRecord,
    RoutingDatasetManifest,
    RoutingExample,
    RoutingSplit,
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
        description="Construct and validate independent routing dataset for retrieval strategy selection."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/routing/phase11"),
        help="Directory to save generated dataset artifacts (default: data/routing/phase11).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for ingestion pipeline (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for ingestion pipeline (default: 200).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host endpoint URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip dense embedding and oracle labeling (for dry run or lightweight generation).",
    )
    parser.add_argument(
        "--target-per-chunk",
        type=int,
        default=5,
        help="Target number of queries to synthesize per chunk (default: 5).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    output_dir = parsed.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 75}")
    print("BUILDING INDEPENDENT ROUTING DATASET (PHASE 11)")
    print(f"  Output Directory : {output_dir}")
    print(f"  Random Seed      : {parsed.seed}")
    print(f"  Chunk Size       : {parsed.chunk_size} / Overlap: {parsed.chunk_overlap}")
    print(f"{'=' * 75}\n")

    # Step 1: Query generation
    print("Step 1: Generating independent queries from source corpus...")
    generator = RoutingDatasetGenerator(
        document_specs=DEFAULT_DOCUMENT_SPECS,
        project_root=_project_root,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        seed=parsed.seed,
    )
    examples = generator.generate(target_per_chunk=parsed.target_per_chunk)
    print(f"  Generated {len(examples)} independent queries across 17 documents.")

    # Split counts
    split_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    len_counts = {"short": 0, "medium": 0, "long": 0}
    doc_splits: dict[str, set[str]] = {}

    for ex in examples:
        split_counts[ex.split.value] = split_counts.get(ex.split.value, 0) + 1
        cat_counts[ex.query_category.value] = cat_counts.get(ex.query_category.value, 0) + 1
        doc_splits.setdefault(ex.split.value, set()).add(ex.source_document)

        words = len(ex.query.split())
        if words < 8:
            len_counts["short"] += 1
        elif words <= 16:
            len_counts["medium"] += 1
        else:
            len_counts["long"] += 1

    print(f"  Partition counts: {split_counts}")
    print(f"  Query taxonomy  : {cat_counts}")
    print(f"  Query lengths   : {len_counts}")

    # Step 2: Oracle Labeling
    oracle_records: list[OracleLabelRecord] = []
    oracle_dist: dict[str, int] = {}

    if not parsed.skip_oracle:
        print("\nStep 2: Running Oracle evaluation (BM25 vs Dense vs Hybrid)...")
        embedder = OllamaEmbedder(model=parsed.embed_model, host=parsed.ollama_host)
        evaluator = OracleEvaluator(
            embedder=embedder,
            chunk_size=parsed.chunk_size,
            chunk_overlap=parsed.chunk_overlap,
        )

        all_doc_paths = [_project_root / spec.relative_path for spec in DEFAULT_DOCUMENT_SPECS]
        print("  Indexing documents into BM25 and vector store...")
        evaluator.index_documents(all_doc_paths)

        print(f"  Pre-warming query embedding cache for {len(examples)} queries...")
        evaluator.warm_query_cache([ex.query for ex in examples])

        print("  Evaluating retrieval effectiveness per query...")
        updated_examples: list[RoutingExample] = []
        for idx, ex in enumerate(examples, start=1):
            if idx % 100 == 0 or idx == len(examples):
                print(f"    Evaluated {idx}/{len(examples)} queries...")
            rec = evaluator.evaluate_example(ex)
            oracle_records.append(rec)
            oracle_dist[rec.oracle_strategy.value] = (
                oracle_dist.get(rec.oracle_strategy.value, 0) + 1
            )

            # Update example with oracle strategy
            updated_ex = ex.model_copy(update={"oracle_strategy": rec.oracle_strategy})
            updated_examples.append(updated_ex)

        examples = updated_examples
        evaluator.close()
        print(f"  Oracle distribution: {oracle_dist}")
    else:
        print("\nStep 2: Skipping Oracle labeling (--skip-oracle requested).")
        for ex in examples:
            oracle_dist[ex.oracle_strategy.value] = oracle_dist.get(ex.oracle_strategy.value, 0) + 1

    # Step 3: Leakage Audit
    print("\nStep 3: Auditing leakage against protected evaluation sets...")
    protected_queries = load_all_protected_queries(_project_root)
    leakage_report = audit_dataset_leakage(
        examples=examples,
        protected_queries=protected_queries,
        fuzzy_threshold=0.70,
    )
    print(f"  Leakage Audit Status: {leakage_report.status}")
    if leakage_report.status != "PASS":
        print(f"  ERROR: Leakage audit failed with diagnostics: {leakage_report.diagnostics}")
        return 1

    # Step 4: Serialize files
    print("\nStep 4: Serializing dataset artifacts...")
    full_path = output_dir / "routing_dataset.jsonl"
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    oracle_path = output_dir / "oracle_labels.jsonl"
    leakage_path = output_dir / "leakage_report.json"
    manifest_path = output_dir / "routing_dataset_manifest.json"

    # Write full dataset
    with open(full_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    # Write split datasets
    with (
        open(train_path, "w", encoding="utf-8") as f_train,
        open(val_path, "w", encoding="utf-8") as f_val,
        open(test_path, "w", encoding="utf-8") as f_test,
    ):
        for ex in examples:
            line = ex.model_dump_json() + "\n"
            if ex.split == RoutingSplit.TRAIN:
                f_train.write(line)
            elif ex.split == RoutingSplit.VALIDATION:
                f_val.write(line)
            elif ex.split == RoutingSplit.TEST:
                f_test.write(line)

    # Write oracle labels
    with open(oracle_path, "w", encoding="utf-8") as f_oracle:
        for rec in oracle_records:
            f_oracle.write(rec.model_dump_json() + "\n")

    # Write leakage report
    leakage_path.write_text(leakage_report.model_dump_json(indent=2), encoding="utf-8")

    # Step 5: Generate manifest
    file_hashes = {
        "routing_dataset.jsonl": compute_sha256(full_path),
        "train.jsonl": compute_sha256(train_path),
        "validation.jsonl": compute_sha256(val_path),
        "test.jsonl": compute_sha256(test_path),
        "oracle_labels.jsonl": compute_sha256(oracle_path),
        "leakage_report.json": compute_sha256(leakage_path),
    }

    manifest = RoutingDatasetManifest(
        dataset_name="Phase 11 Independent Routing Dataset",
        dataset_version="1.0.0",
        created_at=datetime.now(UTC).isoformat(),
        generation_seed=parsed.seed,
        total_examples=len(examples),
        split_counts=split_counts,
        document_splits={k: sorted(v) for k, v in doc_splits.items()},
        oracle_distribution=oracle_dist,
        category_distribution=cat_counts,
        length_distribution=len_counts,
        source_document_count=len(DEFAULT_DOCUMENT_SPECS),
        files=file_hashes,
        leakage_summary={
            "status": leakage_report.status,
            "protected_queries_count": leakage_report.protected_queries_count,
            "exact_matches_found": leakage_report.exact_matches_found,
            "normalized_matches_found": leakage_report.normalized_matches_found,
            "fuzzy_matches_found": leakage_report.fuzzy_matches_found,
            "max_fuzzy_similarity": leakage_report.max_fuzzy_similarity,
            "source_separation_verified": leakage_report.source_separation_verified,
        },
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(f"  Manifest saved to: {manifest_path}")

    print(f"\n{'=' * 75}")
    print("PHASE 11 DATASET CONSTRUCTION & VALIDATION COMPLETE")
    print(f"  Total Validated Examples : {manifest.total_examples}")
    print(f"  Train / Val / Test       : {manifest.split_counts}")
    print(f"  Oracle Distribution      : {manifest.oracle_distribution}")
    print(f"  Leakage Status           : {leakage_report.status}")
    print(f"{'=' * 75}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
