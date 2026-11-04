"""CLI script to execute a reproducible baseline evaluation for Track A or Track B.

Usage:
    uv run python scripts/run_baseline.py --track track_a
    uv run python scripts/run_baseline.py --track track_b
    uv run python scripts/run_baseline.py --track track_a --no-generation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.datasets.schema import CorpusTrack  # noqa: E402
from eval.runners.baseline import BaselineConfig, run_baseline  # noqa: E402
from eval.runners.evaluate import QueryEvaluationResult  # noqa: E402


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible golden evaluation baseline for Track A or Track B."
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["track_a", "track_b"],
        help="Golden evaluation track to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation JSONL dataset (default: eval/golden/{track}.jsonl).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus directory (default: eval/corpora/{track}).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="llama3.2:3b",
        help="Ollama chat model name (default: llama3.2:3b).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable generation phase and evaluate retrieval only.",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Retrieval evaluation cutoffs (default: 1 3 5).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="eval/runs/baseline",
        help="Root output directory for run artifacts (default: eval/runs/baseline).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for Ollama LLM requests (default: 180.0).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    track = CorpusTrack(parsed.track)

    dataset_path = parsed.dataset or f"eval/golden/{track.value}.jsonl"
    corpus_dir = parsed.corpus or f"eval/corpora/{track.value}"

    config = BaselineConfig(
        track=track,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        embed_model=parsed.embed_model,
        chat_model=parsed.chat_model if not parsed.no_generation else None,
        ollama_host=parsed.ollama_host,
        cutoffs=tuple(parsed.cutoffs),
        run_generation=not parsed.no_generation,
        request_timeout=parsed.timeout,
    )

    print(f"=== Baseline Evaluation: {track.value.upper()} ===")
    print(f"Dataset:     {dataset_path}")
    print(f"Corpus:      {corpus_dir}")
    print(f"Embed model: {config.embed_model}")
    print(f"Chat model:  {config.chat_model or 'None (retrieval only)'}")
    print(f"Cutoffs:     {config.cutoffs}")
    print(f"Output root: {parsed.output_root}")
    print()

    def on_query(idx: int, total: int, res: QueryEvaluationResult) -> None:
        ret_ms = f"{res.retrieval_duration_ms:.0f}ms" if res.retrieval_duration_ms else "n/a"
        gen_ms = f"{res.generation_duration_ms:.0f}ms" if res.generation_duration_ms else "n/a"
        err_str = f" err={res.errors}" if res.errors else ""
        print(
            f"  [{idx:02d}/{total:02d}] {res.query_id:<22} "
            f"status={res.status:<7} ret={ret_ms:<8} gen={gen_ms:<8}{err_str}"
        )

    try:
        manifest = run_baseline(
            config, output_root=Path(parsed.output_root), on_query_complete=on_query
        )
    except Exception as exc:
        print(f"\nERROR: Baseline execution failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Baseline Run Complete ===")
    print(f"Run ID:        {manifest.run_id}")
    print(f"Status:        {manifest.run_status}")
    print(f"Duration:      {manifest.duration_seconds:.2f}s")
    print(f"Total Cases:   {manifest.total_cases}")
    print(f"Artifacts dir: {manifest.artifact_dir}")
    print("Artifacts generated:")
    for name, path in sorted(manifest.artifacts.items()):
        print(f"  - {name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
