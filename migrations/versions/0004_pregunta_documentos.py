"""pregunta_documentos

Revision ID: 0004_pregunta_documentos
Revises: 0003_preguntas
Create Date: 2026-10-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0004_pregunta_documentos"
down_revision = "0003_preguntas"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "pregunta_documentos",
        sa.Column(
            "pregunta_id",
            sa.Text,
            sa.ForeignKey("preguntas.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "documento_id",
            sa.Uuid,
            sa.ForeignKey("documentos.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )


def downgrade() -> None:

    op.drop_table("pregunta_documentos")
