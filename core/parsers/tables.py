"""Table extraction and serialization utilities."""

from __future__ import annotations

import re

from core.normalize.schema import TableData


def _escape_cell(text: str) -> str:
    """Escape pipe characters and normalize newlines within a Markdown table cell."""
    # Replace literal newlines and carriage returns with space
    cleaned = re.sub(r"[\r\n]+", " ", text).strip()
    # Escape pipe characters
    return cleaned.replace("|", r"\|")


def serialize_table(table_data: TableData) -> str:
    """Serialize a TableData model into a deterministic Markdown table representation.

    Args:
        table_data: TableData model with rows and optional headers/caption.

    Returns:
        Deterministic pipe-delimited Markdown table string.
    """
    lines: list[str] = []

    if table_data.caption:
        lines.append(f"**Table: {table_data.caption.strip()}**")
        lines.append("")

    headers = table_data.headers
    rows = table_data.rows

    # Determine number of columns
    num_cols = 0
    if headers:
        num_cols = max(num_cols, len(headers))
    for row in rows:
        num_cols = max(num_cols, len(row))

    if num_cols == 0:
        return "\n".join(lines).strip()

    if headers:
        escaped_headers = [_escape_cell(h) for h in headers]
        # Pad if header row has fewer columns than maximum
        while len(escaped_headers) < num_cols:
            escaped_headers.append("")
        lines.append("| " + " | ".join(escaped_headers) + " |")
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    elif rows:
        # If no explicit headers provided, synthesize a basic divider after the first row
        # or use blank headers
        first_row = [_escape_cell(c) for c in rows[0]]
        while len(first_row) < num_cols:
            first_row.append("")
        lines.append("| " + " | ".join(first_row) + " |")
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")
        rows = rows[1:]

    for row in rows:
        escaped_cells = [_escape_cell(c) for c in row]
        while len(escaped_cells) < num_cols:
            escaped_cells.append("")
        lines.append("| " + " | ".join(escaped_cells) + " |")

    return "\n".join(lines).strip()
