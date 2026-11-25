"""Regression tests for import-time schema initialization controls."""

from __future__ import annotations

from unittest.mock import patch

from apps.api import main


def test_skip_schema_init_prevents_create_all(monkeypatch) -> None:
    monkeypatch.setenv("SKIP_SCHEMA_INIT", "true")

    with patch.object(main.Base.metadata, "create_all") as create_all:
        with patch("apps.api.main._sync_sqlite_columns") as sync_sqlite_columns:
            main._initialize_schema()

    create_all.assert_not_called()
    sync_sqlite_columns.assert_not_called()


def test_local_schema_init_remains_enabled(monkeypatch) -> None:
    monkeypatch.delenv("SKIP_SCHEMA_INIT", raising=False)

    with patch.object(main.Base.metadata, "create_all") as create_all:
        with patch("apps.api.main._sync_sqlite_columns") as sync_sqlite_columns:
            main._initialize_schema()

    create_all.assert_called_once_with(main.engine)
    if main.engine.dialect.name == "sqlite":
        sync_sqlite_columns.assert_called_once_with(main.engine, main.Base.metadata)
