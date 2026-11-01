"""sesiones

Revision ID: s0002_sesiones
Revises: s0001_api_keys
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0002_sesiones"
down_revision = "s0001_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "sesiones",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("api_key_id", sa.Uuid),
        sa.Column(
            "creada_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:

    op.drop_table("sesiones")
