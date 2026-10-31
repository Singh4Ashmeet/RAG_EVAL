"""Minimal local CLI for ingesting documents via the default ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.ingest import IngestionStatus, create_default_ingestion_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for document ingestion."""
    parser = argparse.ArgumentParser(
        prog="rag-eval-ingest",
        description="Ingest a local document through the RAG_EVAL ingestion pipeline.",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the local source document file to ingest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ingestion CLI.

    Args:
        argv: Command-line arguments. If None, sys.argv[1:] is parsed.

    Returns:
        0 on IngestionStatus.SUCCESS, 1 on IngestionStatus.FAILED or PARTIAL_SUCCESS.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = create_default_ingestion_pipeline()
    result = pipeline.ingest(Path(args.file_path))

    print(f"Source Path:       {result.source_path}")
    print(f"Ingestion Status:  {result.status.value}")
    print(f"Document ID:       {result.document_id}")
    print(f"Number of Chunks:  {len(result.chunks)}")
    print(f"Duration:          {result.duration_ms:.2f} ms")

    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")

    # Condition 2: Return 0 ONLY for SUCCESS; return 1 for FAILED and PARTIAL_SUCCESS
    if result.status == IngestionStatus.SUCCESS:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
