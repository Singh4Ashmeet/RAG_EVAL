#!/usr/bin/env python
"""M4.2 Chunk-Size Ablation Comparison Script.

Loads experiment results and generates comparison tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.runners.baseline import BaselineRunManifest
from eval.runners.evaluate import EvaluationReport

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_manifest(run_dir: Path) -> BaselineRunManifest | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return BaselineRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_evaluation_report(run_dir: Path) -> EvaluationReport | None:
    report_path = run_dir / "evaluation_report.json"
    if not report_path.exists():
        return None
    try:
        return EvaluationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_experiment_runs(output_root: Path) -> list[Path]:
    """Find all experiment run directories."""
    runs = []
    if not output_root.exists():
        return runs

    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir():
            continue
        for track_dir in sorted(config_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            for run_dir in sorted(track_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "run_manifest.json").exists():
                    runs.append(run_dir)
    return runs


def extract_metrics(report: EvaluationReport) -> dict:
    """Extract key metrics from evaluation report."""
    metrics = report.metrics
    return {
        "recall_at_1": metrics.get("retrieval.recall@1"),
        "recall_at_3": metrics.get("retrieval.recall@3"),
        "recall_at_5": metrics.get("retrieval.recall@5"),
        "mrr_at_5": metrics.get("retrieval.mrr@5"),
        "precision_at_5": metrics.get("retrieval.precision@5"),
        "source_element_recall_at_5": metrics.get("retrieval.source_element_recall@5"),
        "exact_match": metrics.get("answer.exact_match"),
        "token_f1": metrics.get("answer.token_f1"),
        "retrieval_latency_ms": metrics.get("latency.retrieval_ms"),
        "generation_latency_ms": metrics.get("latency.generation_ms"),
    }


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare M4.2 chunk-size ablation experiment results."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="eval/runs/experiments/m4_2_chunk_ablation",
        help="Root output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout).",
    )
    parsed = parser.parse_args(args)

    output_root = Path(parsed.output_root)
    runs = find_experiment_runs(output_root)

    if not runs:
        print(f"No experiment runs found in {output_root}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} experiment runs")

    results = []
    for run_dir in runs:
        manifest = load_manifest(run_dir)
        report = load_evaluation_report(run_dir)

        if not manifest or not report:
            print(f"Warning: Could not load {run_dir}", file=sys.stderr)
            continue

        # Parse config from directory structure
        # output_root/config_name/track/run_id
        try:
            config_name = run_dir.parent.parent.name
            track = run_dir.parent.name
        except Exception:
            config_name = "unknown"
            track = "unknown"

        metrics = extract_metrics(report)
        results.append(
            {
                "config_name": config_name,
                "track": track,
                "run_id": manifest.run_id,
                "status": manifest.run_status,
                "duration_seconds": manifest.duration_seconds,
                "total_cases": manifest.total_cases,
                **metrics,
            }
        )

    if not results:
        print("No valid results found", file=sys.stderr)
        return 1

    # Sort for consistent output
    results.sort(key=lambda r: (r["track"], r["config_name"]))

    # Output
    if parsed.format == "json":
        output = json.dumps(results, indent=2)
    elif parsed.format == "csv":
        import csv
        import io

        output_io = io.StringIO()
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(output_io, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        output = output_io.getvalue()
    else:
        # Table format
        lines = []
        # Header
        header = (
            f"{'Config':<18} {'Track':<8} {'Status':<10} "
            f"{'R@1':>6} {'R@3':>6} {'R@5':>6} "
            f"{'MRR@5':>6} {'P@5':>6} {'SE_R@5':>6} "
            f"{'EM':>6} {'F1':>6} "
            f"{'RetLat(ms)':>10} {'GenLat(ms)':>10} {'Duration(s)':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in results:

            def fmt(v, digits=3):
                if v is None:
                    return "   N/A"
                return f"{v:>6.{digits}f}"

                # Build line parts separately to avoid long line
                part1 = f"{r['config_name']:<18} {r['track']:<8} {r['status']:<10} "  # noqa: B023
                part2 = (
                    f"{fmt(r.get('recall_at_1'))} {fmt(r.get('recall_at_3'))} "  # noqa: B023
                    f"{fmt(r.get('recall_at_5'))} {fmt(r.get('mrr_at_5'))} "  # noqa: B023
                )
                part3 = (
                    f"{fmt(r.get('precision_at_5'))} "  # noqa: B023
                    f"{fmt(r.get('source_element_recall_at_5'))} "  # noqa: B023
                )
                part4 = f"{fmt(r.get('exact_match'))} {fmt(r.get('token_f1'))} "  # noqa: B023
                part5 = (
                    f"{r.get('retrieval_latency_ms', 0):>10.1f} "  # noqa: B023
                    f"{(r.get('generation_latency_ms') or 0):>10.1f} "  # noqa: B023
                    f"{r['duration_seconds']:>10.1f}"  # noqa: B023
                )
                line = part1 + part2 + part3 + part4 + part5  # noqa: F841
                lines.append(line)

        output = "\n".join(lines)

    if parsed.output:
        Path(parsed.output).write_text(output, encoding="utf-8")
        print(f"Output written to {parsed.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
