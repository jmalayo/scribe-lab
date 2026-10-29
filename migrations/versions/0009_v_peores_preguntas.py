"""v_peores_preguntas

Revision ID: 0009_v_peores_preguntas
Revises: 0008_v_metricas_por_metodo
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0009_v_peores_preguntas"
down_revision = "0008_v_metricas_por_metodo"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_peores_preguntas.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_peores_preguntas")
