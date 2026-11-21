"""Train and validate Phase 12 lightweight learned query router."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from core.retrieval.router.learned import LearnedQueryRouter


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train learned retrieval query router on independent routing dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/routing/phase11"),
        help="Directory containing Phase 11 dataset partitions (default: data/routing/phase11).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/routing/phase12"),
        help="Directory to save trained model artifacts (default: models/routing/phase12).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for routing before fallback to Hybrid baseline (default: 0.70).",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=5.0,
        help="Regularization parameter C for Logistic Regression (default: 5.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args(args)


def load_labeled_queries(partition_file: Path) -> tuple[list[str], list[str], int, int]:
    queries: list[str] = []
    labels: list[str] = []
    total = 0
    excluded = 0

    with open(partition_file, encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            strat = rec.get("oracle_strategy", "")
            if strat in ("bm25", "dense", "hybrid"):
                queries.append(rec["query"])
                labels.append(strat)
            else:
                excluded += 1

    return queries, labels, total, excluded


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    data_dir = parsed.data_dir
    output_dir = parsed.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 75}")
    print("TRAINING LEARNED QUERY ROUTER (PHASE 12)")
    print(f"  Data Directory        : {data_dir}")
    print(f"  Output Directory      : {output_dir}")
    print(f"  Confidence Threshold  : {parsed.confidence_threshold}")
    print(f"  Regularization C      : {parsed.c_param}")
    print(f"  Seed                  : {parsed.seed}")
    print(f"{'=' * 75}\n")

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"

    if not train_file.exists() or not val_file.exists():
        print(f"ERROR: Dataset files missing in {data_dir}", file=sys.stderr)
        return 1

    train_q, train_y, train_total, train_excluded = load_labeled_queries(train_file)
    val_q, val_y, val_total, val_excluded = load_labeled_queries(val_file)

    print(
        f"Train Partition      : {train_total} total ({len(train_q)} confident, {train_excluded} excluded)"
    )
    print(
        f"Validation Partition : {val_total} total ({len(val_q)} confident, {val_excluded} excluded)"
    )

    print("\nTraining Logistic Regression classifier (TF-IDF word n-grams)...")
    router = LearnedQueryRouter.train(
        queries=train_q,
        labels=train_y,
        confidence_threshold=parsed.confidence_threshold,
        c_param=parsed.c_param,
        random_state=parsed.seed,
    )

    # Validation evaluation
    val_preds: list[str] = []
    val_confidences: list[float] = []

    for q in val_q:
        dec = router.route(q, threshold=0.0)  # raw prediction
        val_preds.append(dec.strategy.value)
        val_confidences.append(dec.confidence)

    classes = ["bm25", "dense", "hybrid"]
    report = classification_report(
        val_y, val_preds, labels=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(val_y, val_preds, labels=classes)

    acc = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])

    print("\n--- Validation Classification Results ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("\n  Per-Class Metrics:")
    for cls_name in classes:
        p = report[cls_name]["precision"]
        r = report[cls_name]["recall"]
        f = report[cls_name]["f1-score"]
        supp = report[cls_name]["support"]
        print(f"    {cls_name:<8} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f} (N={supp})")

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"    {'':<8} {'bm25':<8} {'dense':<8} {'hybrid':<8}")
    for idx, row in enumerate(cm):
        print(f"    {classes[idx]:<8} {row[0]:<8} {row[1]:<8} {row[2]:<8}")

    print("\n--- Confidence Distribution on Validation ---")
    print(f"  Min    : {np.min(val_confidences):.4f}")
    print(f"  Median : {np.median(val_confidences):.4f}")
    print(f"  Mean   : {np.mean(val_confidences):.4f}")
    print(f"  Max    : {np.max(val_confidences):.4f}")

    # Gating analysis on validation
    print("\n--- Confidence Gating Analysis on Validation ---")
    for th in [0.50, 0.60, 0.70, 0.80]:
        fb_count = sum(1 for c in val_confidences if c < th)
        fb_rate = fb_count / len(val_confidences)
        print(
            f"  Threshold {th:.2f} -> Fallback Rate: {fb_rate:.1%} ({fb_count}/{len(val_confidences)})"
        )

    # Serialize model & metadata
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "train_examples_count": len(train_q),
        "train_total_count": train_total,
        "val_examples_count": len(val_q),
        "validation_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm.tolist(),
            "classes": classes,
        },
        "confidence_stats": {
            "min": round(float(np.min(val_confidences)), 4),
            "median": round(float(np.median(val_confidences)), 4),
            "mean": round(float(np.mean(val_confidences)), 4),
            "max": round(float(np.max(val_confidences)), 4),
        },
    }
    router.save(output_dir, metadata=metadata)
    print(f"\nModel and metadata successfully saved to: {output_dir}")
    print(f"{'=' * 75}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
