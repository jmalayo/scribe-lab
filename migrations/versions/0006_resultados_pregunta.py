"""resultados_pregunta

Revision ID: 0006_resultados_pregunta
Revises: 0005_corridas_eval
Create Date: 2026-10-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0006_resultados_pregunta"
down_revision = "0005_corridas_eval"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "resultados_pregunta",
        sa.Column(
            "corrida_id",
            sa.Uuid,
            sa.ForeignKey("corridas_eval.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "pregunta_id",
            sa.Text,
            sa.ForeignKey("preguntas.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("gold_hash", sa.Text, nullable=False),
        sa.Column("respuesta", sa.Text, nullable=False),
        sa.Column("latencia_ms", sa.Float, nullable=False),
        sa.Column("tokens", sa.Integer, nullable=False),
        sa.Column("gold_spans_at_10", sa.Integer),
        sa.CheckConstraint("gold_spans_at_10 >= 0", name="resultados_pregunta_gold_spans_at_10"),
    )


def downgrade() -> None:

    op.drop_table("resultados_pregunta")
