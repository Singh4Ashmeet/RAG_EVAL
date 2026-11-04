"""Strict UTF-8 JSONL evaluation dataset loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from core.errors import EvaluationError
from eval.datasets.schema import EvaluationCase, EvaluationDataset


def load_jsonl_dataset(file_path: str | Path) -> EvaluationDataset:
    """Load and validate one JSON object per non-empty line.

    The returned digest covers the exact file bytes, including line endings and
    blank lines, so dataset revisions can be identified reproducibly.
    """
    path = Path(file_path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Failed to read evaluation dataset '{path}': {exc}") from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"Evaluation dataset '{path}' must be UTF-8: {exc}") from exc

    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise EvaluationError(
                f"Invalid evaluation record at {path}:{line_number}: {exc}"
            ) from exc

    dataset_track = (
        cases[0].track if cases and all(c.track == cases[0].track for c in cases) else None
    )

    try:
        return EvaluationDataset(
            dataset_id=path.stem,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            cases=cases,
            track=dataset_track,
        )
    except ValidationError as exc:
        raise EvaluationError(f"Invalid evaluation dataset '{path}': {exc}") from exc
