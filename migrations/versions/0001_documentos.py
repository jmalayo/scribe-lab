"""documentos

Revision ID: 0001_documentos
Revises:
Create Date: 2026-10-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0001_documentos"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "documentos",
        sa.Column(
            "id",
            sa.Uuid,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("coleccion", sa.Text, nullable=False),
        sa.Column("nombre", sa.Text, nullable=False),
        sa.Column(
            "sha256",
            sa.Text,
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "ingestado_en",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("coleccion IN ('propia', 'cliente')", name="documentos_coleccion"),
    )


def downgrade() -> None:

    op.drop_table("documentos")
