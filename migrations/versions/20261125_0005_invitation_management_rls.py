"""allow the API role to manage invitations under RLS

Revision ID: 20261125_0005
Revises: 20261125_0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261125_0005"
down_revision = "20261125_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(sa.text("ALTER TABLE public.invitations ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE public.invitations FORCE ROW LEVEL SECURITY"))
    op.execute(sa.text("DROP POLICY IF EXISTS invitation_management ON public.invitations"))
    op.execute(
        sa.text("""
CREATE POLICY invitation_management ON public.invitations
    FOR ALL TO rag_api
    USING (true)
    WITH CHECK (true)
""")
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(sa.text("DROP POLICY IF EXISTS invitation_management ON public.invitations"))
