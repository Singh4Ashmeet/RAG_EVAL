"""Automated Phase 0 validation entrypoint for RAG_EVAL.

Executes static checks, type checking, test suite, import verifications,
and live service checks to validate repository readiness for Phase 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_step(name: str, cmd: list[str]) -> bool:
    print("\n==================================================")
    print(f"--> [Phase 0] {name}")
    print(f"    Command: {' '.join(cmd)}")
    print("==================================================")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False
    print(f"PASSED: {name}")
    return True


def check_python_version() -> bool:
    print(f"\nChecking Python version... Currently running {sys.version}")
    if sys.version_info[:2] != (3, 12):
        print(
            f"FAILED: Python version must be 3.12, got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        return False
    print("PASSED: Python 3.12 verified.")
    return True


def check_imports() -> bool:
    print("\nVerifying component imports and initialization...")
    try:
        from alembic.config import Config

        from apps.api.main import app
        from apps.worker.pg_queue import claim_next_job, enqueue_job
        from apps.worker.tasks import evaluate_dataset, index_document, run_experiment

        _ = app.title
        _ = (index_document, evaluate_dataset, run_experiment, enqueue_job, claim_next_job)
        _ = Config("alembic.ini")
        print("PASSED: API, Worker, Queue, and Migration configurations imported successfully.")
        return True
    except Exception as exc:
        print(f"FAILED: Component imports failed: {exc}")
        return False


def is_ollama_online() -> bool:
    import urllib.request

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def unload_ollama_models() -> None:
    """Free memory by releasing inactive models prior to live verification."""
    import json
    import urllib.request

    for model in ["nomic-embed-text:latest", "llama3.2:3b"]:
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5.0):
                pass
        except Exception:
            pass


def main() -> int:
    print("=" * 60)
    print("RAG_EVAL PHASE 0 AUTOMATED VALIDATION SUITE")
    print("=" * 60)

    steps = [
        ("Python Version", lambda: check_python_version()),
        ("Component Imports", lambda: check_imports()),
        (
            "Ruff Lint",
            lambda: run_step("Ruff Lint", [sys.executable, "-m", "ruff", "check", "."]),
        ),
        (
            "Ruff Format",
            lambda: run_step(
                "Ruff Format",
                [sys.executable, "-m", "ruff", "format", "--check", "."],
            ),
        ),
        (
            "Mypy Type Check",
            lambda: run_step(
                "Mypy Type Check",
                [sys.executable, "-m", "mypy", "core", "apps", "eval"],
            ),
        ),
        (
            "Unit / Offline Tests",
            lambda: run_step("Pytest Unit Suite", [sys.executable, "-m", "pytest"]),
        ),
    ]

    failures: list[str] = []
    for step_name, step_fn in steps:
        if not step_fn():
            failures.append(step_name)

    # Optional Live checks if Ollama is available
    if is_ollama_online():
        unload_ollama_models()
        print("\nOllama service detected on http://localhost:11434. Executing live verification...")
        cmd = [sys.executable, "-m", "pytest", "-m", "live", "--live"]
        if not run_step("Pytest Live Integration", cmd):
            failures.append("Pytest Live Integration")
    else:
        print("\nOllama service not detected on localhost:11434 (skipping live service test).")

    print("\n" + "=" * 60)
    if failures:
        print("PHASE 0 GATE: BLOCKED")
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PHASE 0 GATE: PASS")
    print("Phase 0 is complete and the repository is ready for Phase 1 implementation.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
