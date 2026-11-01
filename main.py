"""Run the RAG_EVAL API and Vite development server together."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"

dotenv.load_dotenv(ROOT / ".env")


def _npm(*args: str, cwd: Path = FRONTEND) -> int:
    """Run npm on Windows (where npm is a .cmd file) and Unix."""
    command = ["npm", *args]
    return subprocess.call(command, cwd=cwd, shell=os.name == "nt")


def _install_frontend() -> None:
    vite_entry = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
    if vite_entry.exists():
        return
    print("Installing frontend dependencies with npm ci...", flush=True)
    if _npm("ci") != 0:
        raise SystemExit("npm ci failed; check the Node.js and npm installation.")


def _commands() -> tuple[list[str], list[str]]:
    uv = shutil.which("uv")
    python = shutil.which("python") or sys.executable
    server = [
        "uvicorn",
        "apps.api.main:app",
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    backend = [uv, "run", *server] if uv else [python, "-m", *server]
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js is required to run the frontend.")
    vite = str(FRONTEND / "node_modules" / "vite" / "bin" / "vite.js")
    frontend = [node, vite, "--host", "127.0.0.1", "--port", "5173"]
    return backend, frontend


def main() -> int:
    if not (ROOT / "pyproject.toml").exists() or not (FRONTEND / "package.json").exists():
        raise SystemExit("Run main.py from a complete RAG_EVAL checkout.")
    if not shutil.which("uv") and not shutil.which("python"):
        raise SystemExit("Install uv or Python 3.12 before starting RAG_EVAL.")
    if not shutil.which("node") or not shutil.which("npm"):
        raise SystemExit("Install Node.js 20.19+ (or 22.12+) and npm before starting RAG_EVAL.")

    _install_frontend()
    backend_command, frontend_command = _commands()
    children: list[subprocess.Popen[bytes]] = []
    try:
        children.append(subprocess.Popen(backend_command, cwd=ROOT))
        children.append(subprocess.Popen(frontend_command, cwd=FRONTEND))
        print("RAG_EVAL is starting:", flush=True)
        print("  Frontend: http://127.0.0.1:5173", flush=True)
        print("  API docs: http://127.0.0.1:8000/docs", flush=True)
        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    return code
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
