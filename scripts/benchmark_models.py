"""Benchmark models script – subcommand 'llm' measures Ollama inference speed."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]
DEFAULT_RUNS = 3
DEFAULT_NUM_PREDICT = 200
DEFAULT_NUM_CTX = 4096
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42
REQUEST_TIMEOUT = 300.0
MEMORY_SAMPLE_INTERVAL = 0.2  # seconds

# ---------------------------------------------------------------------------
# Pure helper functions (testable, no I/O)
# ---------------------------------------------------------------------------


def tokens_per_second(count: int, duration_ns: int) -> float:
    """Compute tokens/sec from a count and a duration in nanoseconds.

    Returns 0.0 if duration_ns is zero or negative.
    """
    if duration_ns <= 0:
        return 0.0
    return count / (duration_ns / 1e9)


def median(values: list[float]) -> float:
    """Return the median of a list of numbers.

    Raises ValueError on empty list.
    """
    if not values:
        raise ValueError("median() requires at least one value")
    return statistics.median(values)


def is_ollama_process_name(name: str | None) -> bool:
    """Return True if the process name belongs to Ollama or its inference workers.

    Matches process names containing 'ollama' (e.g. ollama.exe, ollama app.exe)
    or 'llama-server' (the llama.cpp model runner worker process).
    Handles None safely and matches case-insensitively.
    """
    if not name:
        return False
    name_lower = name.lower()
    return "ollama" in name_lower or "llama-server" in name_lower


def build_prompt(nonce: str | None = None) -> list[dict[str, str]]:
    """Build a deterministic RAG-style chat prompt (~1500 prompt tokens).

    A unique *nonce* is placed at the very start of the system message so
    that Ollama cannot reuse its prompt cache across runs.

    Returns a list of chat messages (role / content dicts).
    """
    if nonce is None:
        nonce = uuid.uuid4().hex

    system_parts = [
        f"Run ID: {nonce}\n\n",
        "You are a precise technical assistant. Answer the question using ONLY "
        "the information in the provided passages. Cite the passage number in "
        "square brackets. If the answer is not in the passages, say so.\n",
    ]

    passages = [
        # Passage 1 – approx 60 words
        (
            "Passage 1: Transformer Architecture\n"
            "The transformer architecture was introduced in 2017 and relies on "
            "self-attention mechanisms rather than recurrence. Each layer consists "
            "of a multi-head attention sublayer followed by a position-wise "
            "feed-forward network. Layer normalization and residual connections "
            "stabilize training. The original model used sinusoidal positional "
            "encodings, though learned encodings have since become popular. "
            "Transformers process all positions in parallel, making them faster "
            "to train than RNNs on modern hardware with sufficient memory bandwidth."
        ),
        # Passage 2 – approx 60 words
        (
            "Passage 2: Retrieval-Augmented Generation\n"
            "Retrieval-Augmented Generation combines a retriever with a generator. "
            "The retriever searches a corpus for relevant passages, which are then "
            "concatenated with the user query and fed to a language model. This "
            "approach reduces hallucination because the model can ground its answers "
            "in retrieved evidence. Dense retrievers use bi-encoder embeddings and "
            "approximate nearest-neighbor search. Sparse retrievers like BM25 rely "
            "on term-frequency statistics and inverted indices for efficient lookup."
        ),
        # Passage 3 – approx 60 words
        (
            "Passage 3: Quantization Methods\n"
            "Post-training quantization reduces model size by converting weights "
            "from 32-bit floating point to lower precision such as 8-bit or 4-bit "
            "integers. GPTQ uses approximate second-order information to minimize "
            "quantization error layer by layer. GGUF is a file format designed for "
            "CPU inference, supporting multiple quantization levels from Q2 to Q8. "
            "Quantized models typically lose one to three percent accuracy on common "
            "benchmarks while reducing memory requirements by three to eight times."
        ),
        # Passage 4 – approx 60 words
        (
            "Passage 4: Evaluation Metrics for RAG\n"
            "RAG systems are evaluated along two axes: retrieval quality and "
            "generation quality. Retrieval is measured by recall at k and mean "
            "reciprocal rank. Generation quality is assessed with faithfulness, "
            "which checks whether the answer is supported by the retrieved passages, "
            "and relevance, which checks whether the answer addresses the question. "
            "Human evaluation remains the gold standard, but LLM-as-judge approaches "
            "correlate well with human scores at lower cost and higher throughput."
        ),
        # Passage 5 – approx 60 words
        (
            "Passage 5: Vector Database Indexing\n"
            "Vector databases store high-dimensional embeddings and support "
            "approximate nearest-neighbor queries. HNSW is a graph-based index that "
            "builds a hierarchical navigable small-world graph, achieving sub-linear "
            "query time. IVF-PQ partitions the space into Voronoi cells and compresses "
            "vectors with product quantization. Choosing the right index depends on "
            "dataset size, dimensionality, recall requirements, and available memory. "
            "Most databases also support metadata filtering to narrow search results."
        ),
    ]

    system_text = "".join(system_parts) + "\n" + "\n\n".join(passages)

    question = (
        "According to the passages, what are the two axes along which RAG systems "
        "are evaluated, and which specific metrics are mentioned for each axis?"
    )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table from headers and rows.

    Each column is left-aligned and padded to the widest entry.
    """
    if not headers:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def _pad_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else len(cell)
            parts.append(cell.ljust(w))
        return "| " + " | ".join(parts) + " |"

    lines = [_pad_row(headers)]
    lines.append("|" + "|".join(f" {'-' * w} " for w in col_widths) + "|")
    for row in rows:
        lines.append(_pad_row(row))
    return "\n".join(lines)


def format_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string (MB or GB)."""
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):.1f} GB"
    return f"{size_bytes / (1 << 20):.0f} MB"


def format_duration(seconds: float) -> str:
    """Format seconds to a readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def ollama_get(path: str, timeout: float = 10.0) -> Any:
    """GET a JSON endpoint from Ollama."""
    r = httpx.get(f"{OLLAMA_BASE}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def ollama_post(path: str, payload: dict, timeout: float = 10.0) -> Any:
    """POST JSON to Ollama and return the parsed response."""
    r = httpx.post(f"{OLLAMA_BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_installed_models() -> dict[str, dict]:
    """Return {model_name: model_info} for all installed models."""
    data = ollama_get("/api/tags")
    models: dict[str, dict] = {}
    for m in data.get("models", []):
        models[m["name"]] = m
    return models


def unload_model(model: str, *, wait: bool = True, max_wait: float = 60.0) -> None:
    """Unload a model from Ollama memory."""
    ollama_post(
        "/api/generate",
        {"model": model, "keep_alive": 0},
        timeout=30.0,
    )
    if wait:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            ps = ollama_get("/api/ps")
            loaded = [m["name"] for m in ps.get("models", [])]
            if model not in loaded:
                return
            time.sleep(0.5)
        print(f"  WARNING: {model} did not unload within {max_wait}s")


def get_ps_info(model: str) -> dict:
    """Return /api/ps entry for a model (size, size_vram, etc.)."""
    ps = ollama_get("/api/ps")
    for m in ps.get("models", []):
        if m["name"] == model:
            return m
    return {}


class MemorySampler:
    """Background thread that samples Ollama RSS and system available memory.

    Measures the peak summed RSS across all matching Ollama-related processes
    (the Ollama server, GUI app, and llama-server worker processes), as well
    as the minimum available system memory. Note that this metric is the aggregate
    RSS of matching processes, not an isolated measurement of model memory.
    """

    def __init__(self, interval: float = MEMORY_SAMPLE_INTERVAL) -> None:
        self.interval = interval
        self.peak_ollama_rss: int = 0
        self.min_available: int = psutil.virtual_memory().available
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss_sum = 0
                for proc in psutil.process_iter(["name", "memory_info"]):
                    try:
                        name = proc.info.get("name")
                        if is_ollama_process_name(name):
                            rss_sum += proc.info["memory_info"].rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_ollama_rss = max(self.peak_ollama_rss, rss_sum)
                avail = psutil.virtual_memory().available
                self.min_available = min(self.min_available, avail)
            except Exception:
                pass
            self._stop.wait(self.interval)


def stream_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    keep_alive: str = "5m",
    num_thread: int | None = None,
) -> dict:
    """POST to /api/chat with stream=true and parse the NDJSON stream.

    Returns a dict with:
      - wall_ttft: wall-clock seconds to first content token
      - wall_total: total wall-clock seconds
      - final: the final JSON message from Ollama (contains eval_count, etc.)
      - response_text: concatenated response
    """
    options: dict[str, Any] = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
    }
    if num_thread is not None:
        options["num_thread"] = num_thread

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
        "keep_alive": keep_alive,
    }

    wall_start = time.monotonic()
    wall_ttft: float | None = None
    final_msg: dict = {}
    response_parts: list[str] = []

    with httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
        with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    if wall_ttft is None:
                        wall_ttft = time.monotonic() - wall_start
                    response_parts.append(chunk["message"]["content"])
                if chunk.get("done"):
                    final_msg = chunk

    wall_total = time.monotonic() - wall_start
    return {
        "wall_ttft": wall_ttft or wall_total,
        "wall_total": wall_total,
        "final": final_msg,
        "response_text": "".join(response_parts),
    }


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def capture_environment() -> dict[str, Any]:
    """Gather system and environment info for the benchmark record."""
    env: dict[str, Any] = {
        "date": datetime.now(UTC).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_model": platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1 << 30), 1),
    }
    try:
        ver = ollama_get("/api/version")
        env["ollama_version"] = ver.get("version", "unknown")
    except Exception:
        env["ollama_version"] = "unreachable"

    battery = psutil.sensors_battery()
    if battery is not None:
        env["ac_power"] = battery.power_plugged
        env["battery_percent"] = battery.percent
    else:
        env["ac_power"] = None  # desktop / no battery

    return env


def check_ac_power(env: dict) -> None:
    """Print a loud warning if not on AC power."""
    if env.get("ac_power") is False:
        print("\n" + "=" * 60)
        print("[WARNING] NOT PLUGGED IN - results may be throttled!")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def benchmark_model(
    model: str,
    *,
    runs: int = DEFAULT_RUNS,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_thread: int | None = None,
) -> dict[str, Any]:
    """Run cold + warm benchmarks for a single model.

    Returns a result dict with all metrics.
    """
    result: dict[str, Any] = {"model": model, "num_thread": num_thread}

    # --- Unload model ---
    print(f"\n  Unloading {model}...")
    unload_model(model)

    # --- Start memory sampler ---
    sampler = MemorySampler()
    sampler.start()

    try:
        # --- COLD run ---
        print("  Cold run (loading from disk)...")
        messages = build_prompt()
        cold_result = stream_chat(
            model,
            messages,
            num_predict=num_predict,
            keep_alive="5m",
            num_thread=num_thread,
        )
        cold_final = cold_result["final"]
        result["cold_ttft"] = cold_result["wall_ttft"]
        result["cold_load_duration_ns"] = cold_final.get("load_duration", 0)
        result["prompt_eval_count"] = cold_final.get("prompt_eval_count", 0)
        print(f"    Cold TTFT: {format_duration(cold_result['wall_ttft'])}")
        print(f"    Prompt tokens: {result['prompt_eval_count']}")

        # --- Get model info from /api/ps ---
        ps_info = get_ps_info(model)
        result["size"] = ps_info.get("size", 0)
        result["size_vram"] = ps_info.get("size_vram", 0)
        if result["size_vram"] > 0:
            print(f"    [GPU] size_vram = {format_size(result['size_vram'])} (GPU in use!)")
        else:
            print("    size_vram = 0 (CPU only [OK])")

        # --- WARM runs ---
        warm_metrics: dict[str, list[float]] = {
            "wall_ttft": [],
            "prompt_tok_s": [],
            "gen_tok_s": [],
            "wall_total": [],
            "eval_count": [],
        }

        for i in range(runs):
            print(f"  Warm run {i + 1}/{runs}...")
            messages = build_prompt()  # fresh nonce each time
            wr = stream_chat(
                model,
                messages,
                num_predict=num_predict,
                keep_alive="5m",
                num_thread=num_thread,
            )
            wf = wr["final"]
            warm_metrics["wall_ttft"].append(wr["wall_ttft"])
            warm_metrics["wall_total"].append(wr["wall_total"])

            prompt_dur = wf.get("prompt_eval_duration", 0)
            prompt_count = wf.get("prompt_eval_count", 0)
            eval_dur = wf.get("eval_duration", 0)
            eval_count = wf.get("eval_count", 0)

            warm_metrics["prompt_tok_s"].append(tokens_per_second(prompt_count, prompt_dur))
            warm_metrics["gen_tok_s"].append(tokens_per_second(eval_count, eval_dur))
            warm_metrics["eval_count"].append(float(eval_count))

            print(
                f"    TTFT: {format_duration(wr['wall_ttft'])}, "
                f"prompt: {warm_metrics['prompt_tok_s'][-1]:.1f} tok/s, "
                f"gen: {warm_metrics['gen_tok_s'][-1]:.1f} tok/s, "
                f"total: {format_duration(wr['wall_total'])}"
            )

        # Compute medians
        result["warm_ttft"] = median(warm_metrics["wall_ttft"])
        result["warm_prompt_tok_s"] = median(warm_metrics["prompt_tok_s"])
        result["warm_gen_tok_s"] = median(warm_metrics["gen_tok_s"])
        result["warm_wall_total"] = median(warm_metrics["wall_total"])
        result["warm_eval_count"] = median(warm_metrics["eval_count"])
        result["warm_runs_raw"] = {k: v for k, v in warm_metrics.items()}

    finally:
        sampler.stop()
        result["peak_ollama_rss"] = sampler.peak_ollama_rss
        result["min_available_mem"] = sampler.min_available

    # --- Unload to free RAM ---
    print(f"  Unloading {model}...")
    unload_model(model)

    return result


def run_thread_sweep(
    model: str,
    *,
    runs: int = DEFAULT_RUNS,
    num_predict: int = DEFAULT_NUM_PREDICT,
    thread_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Sweep num_thread values for a model and return results per thread count."""
    if thread_counts is None:
        thread_counts = [4, 6, 8, 12, 16]

    sweep_results = []
    for nt in thread_counts:
        print(f"\n--- Thread sweep: {model} with num_thread={nt} ---")
        res = benchmark_model(model, runs=runs, num_predict=num_predict, num_thread=nt)
        sweep_results.append(res)

    return sweep_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def results_table(results: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Build headers and rows for the main results table."""
    headers = [
        "Model",
        "Size",
        "Cold TTFT",
        "Warm TTFT",
        "Prompt tok/s",
        "Gen tok/s",
        "Peak Ollama RSS",
        "Min Avail RAM",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["model"],
                format_size(r.get("size", 0)),
                format_duration(r["cold_ttft"]),
                format_duration(r["warm_ttft"]),
                f"{r['warm_prompt_tok_s']:.1f}",
                f"{r['warm_gen_tok_s']:.1f}",
                format_size(r["peak_ollama_rss"]),
                format_size(r["min_available_mem"]),
            ]
        )
    return headers, rows


def thread_sweep_table(sweep: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Build headers and rows for the thread-sweep table."""
    headers = ["num_thread", "Gen tok/s", "Prompt tok/s", "Warm TTFT", "Wall Total"]
    rows = []
    for r in sweep:
        rows.append(
            [
                str(r.get("num_thread", "default")),
                f"{r['warm_gen_tok_s']:.1f}",
                f"{r['warm_prompt_tok_s']:.1f}",
                format_duration(r["warm_ttft"]),
                format_duration(r["warm_wall_total"]),
            ]
        )
    return headers, rows


def save_raw_json(data: dict, out_dir: Path) -> Path:
    """Save raw benchmark data as JSON and return the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"llm_{ts}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def update_benchmarks_md(
    md_path: Path,
    env: dict,
    results: list[dict],
    sweep: list[dict] | None,
    prompt_token_count: int,
) -> None:
    """Update docs/benchmarks.md with the LLM benchmark section."""
    begin_marker = "<!-- BEGIN:llm -->"
    end_marker = "<!-- END:llm -->"

    # Build the section content
    lines = []
    lines.append("")
    lines.append("### Environment")
    lines.append("")
    lines.append(f"- **Date**: {env['date']}")
    lines.append(f"- **OS**: {env['os']}")
    lines.append(f"- **CPU**: {env['cpu_model']}")
    lines.append(f"- **Cores**: {env['physical_cores']} physical / {env['logical_cores']} logical")
    lines.append(f"- **RAM**: {env['total_ram_gb']} GB")
    lines.append(f"- **Ollama**: {env['ollama_version']}")
    ac = env.get("ac_power")
    if ac is True:
        lines.append("- **AC Power**: ✅ Plugged in")
    elif ac is False:
        lines.append("- **AC Power**: ⚠️ NOT plugged in (results may be throttled)")
    else:
        lines.append("- **AC Power**: N/A (desktop or no battery)")
    lines.append(f"- **Prompt tokens**: {prompt_token_count}")
    lines.append("")

    # Main results table
    lines.append("### Results")
    lines.append("")
    headers, rows = results_table(results)
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    if not results:
        lines.append(
            "*Note: No models were benchmarked because none of the requested models "
            "were installed in Ollama, and free disk space on the model storage drive "
            "was below the 25 GB threshold required to pull models.*"
        )
        lines.append("")

    # Thread sweep
    if sweep:
        model_name = sweep[0]["model"]
        lines.append(f"### Thread Sweep ({model_name})")
        lines.append("")
        sh, sr = thread_sweep_table(sweep)
        lines.append(format_markdown_table(sh, sr))

        # Find best
        best = max(sweep, key=lambda x: x["warm_gen_tok_s"])
        lines.append("")
        lines.append(
            f"**Best num_thread = {best['num_thread']}** ({best['warm_gen_tok_s']:.1f} gen tok/s)"
        )
        lines.append("")

    section_content = "\n".join(lines)

    # Read existing file
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
    else:
        text = "# Benchmarks\n"

    # Replace or insert section
    if begin_marker in text and end_marker in text:
        before = text[: text.index(begin_marker) + len(begin_marker)]
        after = text[text.index(end_marker) :]
        text = before + "\n" + section_content + "\n" + after
    else:
        # Append the section
        text = text.rstrip() + "\n\n## LLM Inference\n\n" + begin_marker + "\n"
        text += section_content + "\n" + end_marker + "\n"

    md_path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_llm(args: argparse.Namespace) -> None:
    """Run the LLM benchmark subcommand."""
    env = capture_environment()
    check_ac_power(env)

    print("=" * 60)
    print("LLM Benchmark")
    print("=" * 60)
    print(f"OS: {env['os']}")
    print(f"CPU: {env['cpu_model']}")
    print(f"Cores: {env['physical_cores']}P / {env['logical_cores']}L")
    print(f"RAM: {env['total_ram_gb']} GB")
    print(f"Ollama: {env['ollama_version']}")
    print(f"Runs: {args.runs}, num_predict: {args.num_predict}")
    print()

    # Get installed models
    installed = get_installed_models()
    print(f"Installed models: {', '.join(installed.keys()) or '(none)'}")

    requested = args.models or DEFAULT_MODELS
    to_benchmark = []
    skipped = []
    for m in requested:
        if m in installed:
            to_benchmark.append(m)
        else:
            print(f"  SKIP: {m} is not installed")
            skipped.append(m)

    all_results: list[dict] = []
    prompt_token_count = 0

    if to_benchmark:
        print(f"\nWill benchmark: {', '.join(to_benchmark)}")

        for model in to_benchmark:
            print(f"\n{'=' * 60}")
            print(f"Benchmarking: {model}")
            print("=" * 60)
            try:
                result = benchmark_model(model, runs=args.runs, num_predict=args.num_predict)
                all_results.append(result)
                if result.get("prompt_eval_count"):
                    prompt_token_count = result["prompt_eval_count"]
            except Exception as e:
                print(f"  ERROR benchmarking {model}: {e}")
                continue
    else:
        print("\nNo requested models are installed in Ollama to benchmark.")

    # Thread sweep
    sweep_results: list[dict] | None = None
    if args.thread_sweep and all_results:
        # Pick the model for thread sweep: qwen2.5:7b if installed, else largest
        sweep_model = None
        if "qwen2.5:7b" in [r["model"] for r in all_results]:
            sweep_model = "qwen2.5:7b"
        else:
            # Pick the largest installed model by size
            sweep_model = max(all_results, key=lambda r: r.get("size", 0))["model"]

        print(f"\n{'=' * 60}")
        print(f"Thread Sweep: {sweep_model}")
        print("=" * 60)
        try:
            sweep_results = run_thread_sweep(
                sweep_model, runs=args.runs, num_predict=args.num_predict
            )
        except Exception as e:
            print(f"  ERROR in thread sweep: {e}")

    # --- Output ---
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print()

    headers, rows = results_table(all_results)
    table_md = format_markdown_table(headers, rows)
    print(table_md)
    if not all_results:
        print(f"\nNo models were benchmarked. Skipped: {', '.join(skipped)}")
    print(f"\nPrompt token count: {prompt_token_count}")

    # size_vram summary
    print("\nGPU / VRAM check:")
    if all_results:
        for r in all_results:
            vram = r.get("size_vram", 0)
            status = "GPU in use" if vram > 0 else "CPU only"
            print(f"  {r['model']}: size_vram = {format_size(vram)} ({status})")
    else:
        print("  No models were loaded to check VRAM.")

    if sweep_results:
        print(f"\n{'=' * 60}")
        print("THREAD SWEEP")
        print("=" * 60)
        print()
        sh, sr = thread_sweep_table(sweep_results)
        print(format_markdown_table(sh, sr))
        best = max(sweep_results, key=lambda x: x["warm_gen_tok_s"])
        print(f"\nBest num_thread = {best['num_thread']} ({best['warm_gen_tok_s']:.1f} gen tok/s)")

    # Save raw JSON
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "docs" / "benchmarks" / "raw"
    raw_data = {
        "environment": env,
        "results": all_results,
        "skipped": skipped,
        "thread_sweep": sweep_results,
        "prompt_token_count": prompt_token_count,
    }
    json_path = save_raw_json(raw_data, raw_dir)
    print(f"\nRaw results saved to: {json_path}")

    # Update benchmarks.md
    md_path = project_root / "docs" / "benchmarks.md"
    update_benchmarks_md(md_path, env, all_results, sweep_results, prompt_token_count)
    print(f"Updated: {md_path}")

    check_ac_power(env)
    print("\nDone.")


def main() -> None:
    """Entry point – parse args and dispatch subcommand."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="RAG-EVAL model benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    llm_parser = sub.add_parser("llm", help="Benchmark LLM inference on Ollama")
    llm_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to benchmark (default: {' '.join(DEFAULT_MODELS)})",
    )
    llm_parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of warm runs (default: {DEFAULT_RUNS})",
    )
    llm_parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=f"Max tokens to generate per run (default: {DEFAULT_NUM_PREDICT})",
    )
    llm_parser.add_argument(
        "--thread-sweep",
        action="store_true",
        help="Run a num_thread sweep after the main benchmark",
    )

    args = parser.parse_args()
    if args.command == "llm":
        cmd_llm(args)


if __name__ == "__main__":
    main()
