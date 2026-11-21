"""Unit tests for table serialization utilities."""

from __future__ import annotations

from core.normalize.schema import TableData
from core.parsers.tables import serialize_table


class TestTableSerialization:
    """Test suite for serialize_table function."""

    def test_serialize_with_headers_and_rows(self) -> None:
        table_data = TableData(
            headers=["City", "Population"],
            rows=[["Tokyo", "37M"], ["Delhi", "32M"]],
            caption="Major Cities",
        )
        md = serialize_table(table_data)
        assert "**Table: Major Cities**" in md
        assert "| City | Population |" in md
        assert "| --- | --- |" in md
        assert "| Tokyo | 37M |" in md
        assert "| Delhi | 32M |" in md

    def test_serialize_escapes_pipe_characters(self) -> None:
        table_data = TableData(
            headers=["Pattern", "Description"],
            rows=[["a | b", "Union of a and b"]],
        )
        md = serialize_table(table_data)
        assert r"a \| b" in md
        assert "Union of a and b" in md

    def test_serialize_without_headers(self) -> None:
        table_data = TableData(
            headers=None,
            rows=[["Row 0 Col 0", "Row 0 Col 1"], ["Row 1 Col 0", "Row 1 Col 1"]],
        )
        md = serialize_table(table_data)
        assert "| Row 0 Col 0 | Row 0 Col 1 |" in md
        assert "| --- | --- |" in md
        assert "| Row 1 Col 0 | Row 1 Col 1 |" in md

    def test_serialize_empty_table(self) -> None:
        table_data = TableData(headers=None, rows=[])
        assert serialize_table(table_data) == ""
