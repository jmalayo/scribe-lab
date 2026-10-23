"""veredictos

Revision ID: 0007_veredictos
Revises: 0006_resultados_pregunta
Create Date: 2026-10-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0007_veredictos"
down_revision = "0006_resultados_pregunta"
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "veredictos",
        sa.Column("corrida_id", sa.Uuid, primary_key=True),
        sa.Column("pregunta_id", sa.Text, primary_key=True),
        sa.Column("fuente", sa.Text, primary_key=True),
        sa.Column("veredicto", sa.Text, nullable=False),
        sa.Column("motivo", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(
            ["corrida_id", "pregunta_id"],
            ["resultados_pregunta.corrida_id", "resultados_pregunta.pregunta_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("fuente IN ('juez', 'humano')", name="veredictos_fuente"),
        sa.CheckConstraint(
            "veredicto IN ('correcta', 'parcial', 'incorrecta', 'abstencion')",
            name="veredictos_veredicto",
        ),
    )


def downgrade() -> None:

    op.drop_table("veredictos")
