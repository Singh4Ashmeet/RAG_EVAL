"""Unit tests for benchmark helper functions – no network required."""

from __future__ import annotations

import uuid

from scripts.benchmark_models import (
    build_prompt,
    format_markdown_table,
    is_ollama_process_name,
    median,
    tokens_per_second,
)

# ---------------------------------------------------------------------------
# tokens_per_second
# ---------------------------------------------------------------------------


class TestTokensPerSecond:
    def test_known_values(self):
        # 100 tokens in 2 seconds (2_000_000_000 ns) = 50 tok/s
        assert tokens_per_second(100, 2_000_000_000) == 50.0

    def test_one_token_one_second(self):
        assert tokens_per_second(1, 1_000_000_000) == 1.0

    def test_zero_duration(self):
        assert tokens_per_second(100, 0) == 0.0

    def test_negative_duration(self):
        assert tokens_per_second(100, -1) == 0.0

    def test_large_count(self):
        # 1000 tokens in 0.5 seconds = 2000 tok/s
        result = tokens_per_second(1000, 500_000_000)
        assert abs(result - 2000.0) < 0.01


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------


class TestMedian:
    def test_odd_count(self):
        assert median([3.0, 1.0, 2.0]) == 2.0

    def test_even_count(self):
        assert median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_single_value(self):
        assert median([42.0]) == 42.0

    def test_two_values(self):
        assert median([1.0, 3.0]) == 2.0

    def test_already_sorted(self):
        assert median([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_empty_raises(self):
        import pytest

        with pytest.raises(ValueError, match="at least one value"):
            median([])


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_nonce_at_start(self):
        nonce = uuid.uuid4().hex
        messages = build_prompt(nonce=nonce)
        system_content = messages[0]["content"]
        assert system_content.startswith(f"Run ID: {nonce}")

    def test_different_nonces_produce_different_prompts(self):
        m1 = build_prompt(nonce="aaa")
        m2 = build_prompt(nonce="bbb")
        assert m1[0]["content"] != m2[0]["content"]

    def test_deterministic_apart_from_nonce(self):
        m1 = build_prompt(nonce="fixed")
        m2 = build_prompt(nonce="fixed")
        assert m1 == m2

    def test_has_system_and_user_messages(self):
        messages = build_prompt(nonce="test")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_contains_passages(self):
        messages = build_prompt(nonce="test")
        system = messages[0]["content"]
        for i in range(1, 6):
            assert f"Passage {i}:" in system

    def test_auto_nonce_when_none(self):
        messages = build_prompt()
        system = messages[0]["content"]
        assert system.startswith("Run ID: ")


# ---------------------------------------------------------------------------
# format_markdown_table
# ---------------------------------------------------------------------------


class TestFormatMarkdownTable:
    def test_basic_table(self):
        headers = ["A", "B"]
        rows = [["1", "2"], ["3", "4"]]
        result = format_markdown_table(headers, rows)
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
        assert "| A" in lines[0]
        assert "| B" in lines[0]
        assert "-" in lines[1]  # separator row
        assert "| 1" in lines[2]

    def test_column_alignment(self):
        headers = ["Name", "Value"]
        rows = [["short", "1"], ["a longer name", "2"]]
        result = format_markdown_table(headers, rows)
        lines = result.split("\n")
        # All lines should have the same length
        lengths = {len(line) for line in lines}
        assert len(lengths) == 1, f"Inconsistent line lengths: {lengths}"

    def test_empty_headers(self):
        assert format_markdown_table([], []) == ""

    def test_single_column(self):
        result = format_markdown_table(["X"], [["a"], ["b"]])
        assert "| X |" in result
        assert "| a |" in result


# ---------------------------------------------------------------------------
# is_ollama_process_name
# ---------------------------------------------------------------------------


class TestIsOllamaProcessName:
    def test_ollama_exe(self):
        assert is_ollama_process_name("ollama.exe") is True

    def test_ollama_app_exe(self):
        assert is_ollama_process_name("ollama app.exe") is True

    def test_llama_server_exe(self):
        assert is_ollama_process_name("llama-server.exe") is True

    def test_mixed_case(self):
        assert is_ollama_process_name("Llama-Server.EXE") is True
        assert is_ollama_process_name("OLLAMA.EXE") is True
        assert is_ollama_process_name("Ollama App.Exe") is True

    def test_unrelated_processes_do_not_match(self):
        assert is_ollama_process_name("chrome.exe") is False
        assert is_ollama_process_name("python.exe") is False

    def test_absent_and_empty_names(self):
        assert is_ollama_process_name(None) is False
        assert is_ollama_process_name("") is False
