"""v_metricas_por_metodo

Revision ID: 0008_v_metricas_por_metodo
Revises: 0007_veredictos
Create Date: 2026-10-24
"""
from pathlib import Path

from alembic import op

revision = "0008_v_metricas_por_metodo"
down_revision = "0007_veredictos"
branch_labels = None
depends_on = None

VIEW = Path(__file__).resolve().parents[1] / "views" / "v_metricas_por_metodo.sql"


def upgrade() -> None:

    op.execute(VIEW.read_text(encoding="utf-8"))


def downgrade() -> None:

    op.execute("DROP VIEW v_metricas_por_metodo")
