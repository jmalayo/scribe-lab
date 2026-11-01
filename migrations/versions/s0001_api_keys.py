"""api_keys

Revision ID: s0001_api_keys
Revises: 
Create Date: 2026-11-01
"""
import sqlalchemy as sa
from alembic import op

revision = "s0001_api_keys"
down_revision = None
branch_labels = ("servicio",)
depends_on = None


def upgrade() -> None:

    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            sa.Uuid,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "dueno",
            sa.Text,
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "hash",
            sa.Text,
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "limite_por_minuto",
            sa.Integer,
            nullable=False,
            server_default="60",
        ),
        sa.Column(
            "creada_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("revocada_en", sa.DateTime(timezone=True)),
        sa.CheckConstraint("limite_por_minuto > 0", name="api_keys_limite_por_minuto"),
    )


def downgrade() -> None:

    op.drop_table("api_keys")
