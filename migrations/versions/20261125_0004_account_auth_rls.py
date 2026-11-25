"""add invited identity RLS policy for account bootstrap

Revision ID: 20261125_0004
Revises: 20261109_0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261125_0004"
down_revision = "20261109_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(sa.text("ALTER TABLE public.accounts ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE public.accounts FORCE ROW LEVEL SECURITY"))
    op.execute(sa.text("DROP POLICY IF EXISTS account_self_delete ON public.accounts"))
    op.execute(sa.text("DROP POLICY IF EXISTS account_self_update ON public.accounts"))
    op.execute(sa.text("DROP POLICY IF EXISTS account_invited_creation ON public.accounts"))
    op.execute(sa.text("DROP POLICY IF EXISTS account_identity_isolation ON public.accounts"))
    op.execute(
        sa.text("""
CREATE POLICY account_identity_isolation ON public.accounts
    FOR SELECT TO rag_api
    USING (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    )
""")
    )
    op.execute(
        sa.text("""
CREATE POLICY account_invited_creation ON public.accounts
    FOR INSERT TO rag_api
    WITH CHECK (
        github_id = current_setting('app.auth_subject', true)
        AND email = current_setting('app.auth_email', true)
        AND EXISTS (
            SELECT 1 FROM public.invitations
            WHERE invitations.email = accounts.email
        )
    )
""")
    )
    op.execute(
        sa.text("""
CREATE POLICY account_self_update ON public.accounts
    FOR UPDATE TO rag_api
    USING (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    )
    WITH CHECK (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    )
""")
    )
    op.execute(
        sa.text("""
CREATE POLICY account_self_delete ON public.accounts
    FOR DELETE TO rag_api
    USING (id = current_setting('app.workspace_id', true))
""")
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    for policy in (
        "account_self_delete",
        "account_self_update",
        "account_invited_creation",
        "account_identity_isolation",
    ):
        op.execute(sa.text(f"DROP POLICY IF EXISTS {policy} ON public.accounts"))
