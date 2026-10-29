"""preguntas

Revision ID: 0003_preguntas
Revises: 0002_chunks
Create Date: 2026-10-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0003_preguntas"
down_revision = "0002_chunks"
branch_labels = None
depends_on = None

TIPOS = ("simple", "comparativa_mismo_doc", "comparativa_entre_docs", "sin_respuesta")


def upgrade() -> None:

    op.create_table(
        "preguntas",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("biblioteca", sa.Text, nullable=False),
        sa.Column("tipo", sa.Text, nullable=False),
        sa.Column("pregunta", sa.Text, nullable=False),
        sa.Column("respuesta_esperada", sa.Text, nullable=False),
        sa.Column("gold_hash", sa.Text, nullable=False),
        sa.CheckConstraint(
            "tipo IN (" + ", ".join(f"'{t}'" for t in TIPOS) + ")",
            name="preguntas_tipo",
        ),
    )


def downgrade() -> None:

    op.drop_table("preguntas")
