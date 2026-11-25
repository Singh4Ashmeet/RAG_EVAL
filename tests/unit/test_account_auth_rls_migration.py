"""Regression coverage for the production account-bootstrap RLS policy."""

from __future__ import annotations

from pathlib import Path


def test_account_auth_rls_migration_requires_invited_identity() -> None:
    migration = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "20261125_0004_account_auth_rls.py"
    ).read_text(encoding="utf-8")

    assert "ENABLE ROW LEVEL SECURITY" in migration
    assert "FORCE ROW LEVEL SECURITY" in migration
    assert "FOR INSERT TO rag_api" in migration
    assert "app.auth_subject" in migration
    assert "app.auth_email" in migration
    assert "FROM public.invitations" in migration
    assert "BYPASSRLS" not in migration

    bootstrap = (
        Path(__file__).resolve().parents[2]
        / "deploy"
        / "sql"
        / "01_worker_roles_and_privileges.sql"
    ).read_text(encoding="utf-8")
    assert "ALTER TABLE public.accounts ENABLE ROW LEVEL SECURITY" in bootstrap
    assert "FOR INSERT TO rag_api" in bootstrap
    assert "FROM public.invitations" in bootstrap


def test_github_account_resolution_sets_production_rls_identity_context() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from apps.api.main import Invitation, resolve_auth_account

    db = MagicMock()
    db.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    db.scalar.side_effect = [None, None]
    db.get.return_value = Invitation(email="invited@example.test")

    account = resolve_auth_account(
        db,
        "github-verified-subject",
        "invited@example.test",
        "invited-user",
    )

    assert account.auth_subject == "github-verified-subject"
    assert account.email == "invited@example.test"
    assert db.add.called
    executed_sql = [str(call.args[0]) for call in db.execute.call_args_list]
    assert any("app.auth_subject" in statement for statement in executed_sql)
    assert any("app.auth_email" in statement for statement in executed_sql)
