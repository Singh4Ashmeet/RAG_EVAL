from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261101_0002"
down_revision = "20261101_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_login_tokens",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_email_login_tokens_email", "email_login_tokens", ["email"])


def downgrade() -> None:
    op.drop_index("ix_email_login_tokens_email", table_name="email_login_tokens")
    op.drop_table("email_login_tokens")
