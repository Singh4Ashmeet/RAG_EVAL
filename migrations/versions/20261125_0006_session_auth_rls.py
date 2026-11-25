"""scope authenticated sessions to their token or workspace

Revision ID: 20261125_0006
Revises: 20261125_0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261125_0006"
down_revision = "20261125_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(sa.text("ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE public.sessions FORCE ROW LEVEL SECURITY"))
    for policy in (
        "session_token_lookup",
        "session_owner_create",
        "session_owner_delete",
    ):
        op.execute(sa.text(f"DROP POLICY IF EXISTS {policy} ON public.sessions"))
    op.execute(
        sa.text("""
CREATE POLICY session_token_lookup ON public.sessions
    FOR SELECT TO rag_api
    USING (
        token_hash = current_setting('app.session_token_hash', true)
        OR account_id = current_setting('app.workspace_id', true)
    )
""")
    )
    op.execute(
        sa.text("""
CREATE POLICY session_owner_create ON public.sessions
    FOR INSERT TO rag_api
    WITH CHECK (account_id = current_setting('app.workspace_id', true))
""")
    )
    op.execute(
        sa.text("""
CREATE POLICY session_owner_delete ON public.sessions
    FOR DELETE TO rag_api
    USING (
        token_hash = current_setting('app.session_token_hash', true)
        OR account_id = current_setting('app.workspace_id', true)
    )
""")
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    for policy in (
        "session_owner_delete",
        "session_owner_create",
        "session_token_lookup",
    ):
        op.execute(sa.text(f"DROP POLICY IF EXISTS {policy} ON public.sessions"))
